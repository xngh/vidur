from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.utils.radix_tree import RadixCache
from vidur.scheduler.utils.kv_block_manager import KVBlockManager
from vidur.entities.full_request import FullRequest
from vidur.entities.batch import Batch

from math import ceil
from collections import defaultdict
from typing import Set, List, Dict

class LocalReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._num_running_batches = 0
        self._preempted_requests = []
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

        self.max_tokens_per_request = self._config.max_tokens_per_request
        self.bin_width = self._config.bin_width

        self.block_manager = KVBlockManager(
            int(self._config.num_blocks),
            self._config.block_size
        )
        self.tree_cache = RadixCache(self.block_manager, self._config.block_size)
    
    def can_allocate(self, request: FullRequest, num_required_blocks: int) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_blocks_left = self.block_manager._num_total_blocks - self.block_manager.num_used_blocks \
                                - num_required_blocks
            if num_blocks_left < self._watermark_blocks:
                self.tree_cache.evict(self._watermark_blocks - num_blocks_left)

                return (self.block_manager._num_total_blocks 
                        - self.block_manager.num_used_blocks
                        - num_required_blocks 
                        >= self._watermark_blocks)
            else:
                return True
            

        # vllm requires at least one block to be available
        return self.block_manager._num_total_blocks - self.block_manager.num_used_blocks >= 1
    
    # 重载 base_replica_scheduler的allocate方法
    # 传入的是需要新分配的num_blocks，因此外面需要把复用的block个数赋值给new request的allocate_map
    # - case: node1 : [a, b, c, d]   request: [a, b, c, d, e, f, g, h], num_blocks: 1, but request not in allocate_map
    # - self._allocation_map[request_id] = num_blocks is error
    # - so we add param num_matched_blocks for new request
    def allocate(self, request: FullRequest, num_blocks: int, num_matched_blocks: int = 0):
        request_id = request.id

        num_token_required = 0
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks + num_matched_blocks
            num_token_required = request.num_prefill_tokens - len(request.block_table)
        else:
            # 在这个case里的request有两种可能：
            # -- partial prefill   这种会进入这边吗? 感觉好像并不会进来，而是prefill统一allocate
            # -- decode
            if request.is_prefill_complete:
                num_token_required  = 1
            else:
                num_token_required = request.num_prefill_tokens - request.num_processed_tokens

            # 这里默认一个block只能给一个request使用
            num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
            num_blocks_required = (max(0, request.num_processed_tokens + num_token_required - num_tokens_reserved) + self._config.block_size - 1) \
                // self._config.block_size

            num_blocks = num_blocks_required
            self._allocation_map[request_id] += num_blocks

            # for decode request，修改block_table    eg. [0,0,0,0,1] -> [0,0,0,0,1,1] (block_size = 4)
            if num_blocks == 0 and num_token_required == 1:
                request.append_block(request.get_last_block_id())
                # self.block_manager.increment_slot(request.block_table[-1])

        assert self.block_manager.num_used_blocks <= self._config.num_blocks

        if num_blocks == 0:
            return 
        
        # 分配新 Block
        # 这里的逻辑比较不合理. 一个block可以存储block size大小的token kv。 对于decode阶段的request，每个token算出来都需要 1 block
        # 就会导致后续相当于每个block只存了一个token的kv eg. [0,0,0,0,1,2,3,4](block_size = 4)
        # 因此需要考虑复用已分配，slot还未满的block, 这也是sarathi的allocate_request中的 line 48 -49的逻辑
        # 我把这部分逻辑补充在上方
        new_blocks = [self.block_manager.allocate_block() for _ in range(num_blocks)]
        
        # 转成冗余的 block id 列表
        extend_block_table = []
        
        for i in range(num_token_required):
            index = i // self._config.block_size
            extend_block_table.append(new_blocks[index])
            # self.block_manager.increment_slot(new_blocks[index])

        # 设置给 request block table
        request.block_table = request.block_table + extend_block_table

    def _can_allocate_request(self, request: FullRequest) -> bool:
        reused_block_ids, last_node_match = self.tree_cache.match_prefix(
            request.input_token_ids
        )
        if request.last_node is None:
            request.last_node = last_node_match
            request.prefix_indices = reused_block_ids

        num_matched_blocks = len(set(reused_block_ids))
        request.num_matched_tokens = len(reused_block_ids)

        if request.id not in self._allocation_map:
            # new request
            num_matched_tokens = len(reused_block_ids)
            assert request.num_prefill_tokens - num_matched_tokens >= 0

            num_required_blocks = ceil(
                (request.num_prefill_tokens - num_matched_tokens) / self._config.block_size
            )
            return self.can_allocate(request, num_required_blocks)
        else:
            num_tokens_reserved = len(request.block_table)
            num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

            assert (
                    num_tokens_required == 0 or num_tokens_required == 1
            ), f"num_tokens_required: {num_tokens_required}"

            if num_tokens_required == 0:
                return True
            else:
                return self.can_allocate(request, 1)


    # 因为block manager的allocate_block中已经自动为新分配的block加了ref_count，所以这里不需要再加
    # 还有一个问题就是，这里和chunk是怎么联系起来的，理论上，request需要的token是跟chunk有关的？
    def _allocate_request(self, request: FullRequest) -> None:
        # 尝试前缀匹配 
        # 冗余 Blocks 列表, 匹配的 token 数量, 最后一个节点
        reused_block_ids, last_node_match = self.tree_cache.match_prefix(
            request.input_token_ids
        )
        if request.last_node is None:
            request.last_node = last_node_match
            request.prefix_indices = reused_block_ids

        num_matched_blocks = len(set(reused_block_ids))
        if len(request.block_table) == 0 and len(reused_block_ids) > 0:
            request.set_block_table(reused_block_ids)
            # 增加前缀树中被复用的block的引用计数
            # 遍历冗余列表，但只对唯一的 Block ID 增加引用计数
            self.block_manager.increment_ref_for_blocks(reused_block_ids)

            # 还得增加 tree node 的引用计数
            self.tree_cache.inc_lock_ref(last_node_match)
        
        # 计算新 Block 需求
        # 这个判断是对的，因为这里的block id是冗余的
        # 即 block_size = 4, 则 [a, b, c, d] -> [0, 0, 0, 0]
        num_matched_tokens = len(reused_block_ids) 

        # for new request
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens - num_matched_tokens) / self._config.block_size
            )
            if self.can_allocate(request, num_required_blocks):
                self.allocate(request, num_required_blocks, num_matched_blocks)
            else:
                print(f"Request {request.id} cannot be allocated because there are not enough blocks available.")
            return
        
        # reserved token的个数计算感觉不太对，要是某个block没满，直接乘就是有问题的
        # 然后这里我用processed_token - reserved_token 是认为 reserved_token 包括matched_token
        # 这里的case很奇怪，似乎就是decode的场景
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        # num_tokens_reserved = len(request.block_table)
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)


        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}, {len(request.block_table)=}, {request.num_processed_tokens=}"

        # yinhan:
        # if num_tokens_required == 0, it means a block allocated to the request can be
        # reused. In vidur logic, this situation doesn't need any more operations,
        # but in our logic, though there is also no need for another block, we need to
        # append the reused block id to request's block table. This logic I have implemented
        # in function allocate(request, num_blocks), so I annotate follow 2 lines

        # if num_tokens_required == 0:
        #    return

        self.allocate(request, 1)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if not request.completed:
                if request.is_prefill_complete and request.num_processed_decode_tokens == 1:
                    self._prepare_for_decode(request)
                    assert len(request.block_table) == request.num_processed_tokens == len(request.fill_ids), "error when process block table and num_processed_tokens."
                self._preempted_requests.append(request)

            self.cache_request(request)

            # 还是得先 cache 再 free
            # 这里有个case，就是某个request的总token长度为block size的倍数，因此相当于整个request都被cache
            # 这时候先free再cache，会导致output的block的ref - 1 = 0， 然后free。cache时又想给ref+1，但是
            # 找不到这个block了。 但是prepare_for_decode得在前面执行（因为计数原因），因此调换顺序
            if request.completed:
                self._free_request([request])

    # 这里有个 case，就是如果刚好给所有的 prefill token 分配完block，即prefill_completed == True
    # vidur原本的做法是在batch_end事件的时候，给request的状态作改变，eg. request.is_prefilled_completed = True
    # 然而，这个第一个decode的token所需要的block并没有分配
    # -- case1：prefill token刚好占据完整的block，需要额外的一个block
    # -- case2：prefill token所用的block还有剩余空间，这时候如何处理
    #     -- decode token复用最后一个block，这样节省block，并且比较符合逻辑
    #     -- decode token默认新开一个block，这样两种case就合并成一种，而且相当于prompt级别的cache match
    # 由于replica_scheduler的on_batch_end 在batch的
    def _prepare_for_decode(self, request: FullRequest) -> None:
        if len(request.get_block_table()) == 0:
            print(f"Request {request.id} does not contain any blocks, error.")
            return

        request.sim_output_tokens(1)

        # append block id
        # 这里调用时，num_processed_token = len(block_table) + 1
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_blocks_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        if num_blocks_required == 0:
            request.append_block(request.get_last_block_id())
        else:
            id = self.block_manager.allocate_block()
            request.append_block(id)
            self._allocation_map[request.id] += 1

        assert request.num_processed_tokens == len(
            request.block_table), "error when process block table and num_processed_tokens"

    # 这里修改time_predictor相关的逻辑
    # sklearn_time_predictor中，把当前batch的所有request的待处理的token求和去查表
    # 待处理的token就是这里的next_num_tokens
    # 对于有kv复用的场景，只需要考虑把prefill阶段的matched的token数量减去即可
    # 可以看到原始的逻辑是 num_prefill_tokens - num_processed_tokens
    # num_processed_tokens 在每次batch最后会更新，因此
    # case1： req = [a, b, c, d], 新到达，且radix tree刚好缓存了 [a, b, c, d], 这时候 next_num_tokens = num_prefill_tokens - num_processed_tokens - matched_tokens = 4 - 0 - 4 = 0
    # case2： req = [a, b, c, d], 新到达，但radix tree没有缓存 [a, b, c, d], 这时候 next_num_tokens = num_prefill_tokens - num_processed_tokens - matched_tokens = 4 - 0 - 0 = 4
    # case3： req = [a, b, c, d, e, f, g, h], 虽然radix tree缓存了所有token，但是由于chunk size的限制第一次prefill只处理了a b c d；第二次到达时
    # next_num_tokens = num_prefill_tokens - num_processed_tokens - matched_tokens = 8 - 4 - 8 = -4 这里产生了错误
    # 错误的原因在于有的时候num_processed_token包含了match_tokens，有的时候没有
    # 这里修改的方式是用 max(request.num_processed_tokens, request.matched_tokens) 代替 request.num_processed_tokens
    def _get_request_next_num_tokens(
        self, request: FullRequest, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        # decode
        if request.is_prefill_complete:
            return 1

        # prefill 
        next_num_tokens = min(
            request.num_prefill_tokens - max(request.num_processed_tokens, request.num_matched_tokens),
            self._config.chunk_size - num_batch_tokens,
        )

        next_num_tokens = max(0, next_num_tokens)

        return next_num_tokens

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        skipped_requests = []
        running_prefills = []
        contains_prefill = False
        num_batch_tokens = 0

        # preempted requests could contain multiple requests which have
        # partial prefills completed, so we need to be careful
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue
            
            # 进入decode阶段的request  这里固定返回1
            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                # 无法分配时
                # 把preempted requests的最后一个放回request队列的头部
                # 但是preempted request是什么
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    self._free_request([victim_request])  # 这里扣减了node.ref_count  也删除了 block id
                    victim_request.restart()
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    # 如果没有可抢占的request，则把当前request放回队列的开头
                    self._free_request([request.id])
                    request.restart()
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete   #确保都是decode
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # re-add the skipped requests, but make sure that we add them to the
        # front of the queue so that they are scheduled first and we maintain FIFO ordering
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )
        skipped_requests = []

        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            if not self._can_allocate_request(self._request_queue[0]):
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)

            # all new requests will have a prefill
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)
    
    # 参考 vllm 的前缀复用逻辑
    # 对于radix tree，request完成的时候对于block和node的引用-1，但是不代表驱逐
    # 驱逐的时候，应该是优先对于引用最少得进行LRU（RadixTree）
    # 这里不能采用vidur的free函数，因为他不考虑block复用的情况
    def _free_request(self, requests: List[FullRequest]) -> None:
        for request in requests:
            # 更新allocation_map
            _ = self._allocation_map.pop(request.id)

            # block.ref_count - 1
            uni_blocks = set(request.block_table)
            for block_id in uni_blocks:
                self.block_manager.free_block(block_id)

            # node.ref_count - 1
            # 会从last开始往祖先回溯，全部 - 1
            # 由于 cache_finished_req 时会dec_lock_ref一次，因此只对还未完成的request做该操作
            if not request.completed:
                self.tree_cache.dec_lock_ref(request.last_node)

    def cache_request(self, request: FullRequest) -> None:
        if request.completed:
            if request.id == 1:
                print("hook here")
            self.tree_cache.cache_finished_req(request)
        else:
            self.tree_cache.cache_unfinished_req(request)

    # ------------- RL relative methods ---------------

    def get_replica_scheduler_state(self, request: FullRequest):
        '''
            1. scheduler_id: one_hot   # 在global scheduler里面实现
            2. local_scheduler_state for every replica scheduler
                a. kv_matched_token_num / num_prefill_tokens
                b. token_distribution for running requests and pending requests
            3. decode time for current batch
        :return:
        '''
        replica_state = []
        matched_tokens, _ = self.tree_cache.match_prefix(request.input_token_ids)
        assert len(matched_tokens) <= len(request.input_token_ids)
        num_matched_tokens = len(matched_tokens)

        save_ratio = num_matched_tokens / len(request.input_token_ids)

        batch_queue = self.get_replica_stage_scheduler(0).get_batch_queue()
        running_req_distribution = self.get_running_state(batch_queue)

        pending_req_distribution = self.get_waiting_state(self._request_queue)
        assert len(running_req_distribution) == len(pending_req_distribution), f"running_req_len:{len(running_req_distribution)} and pending_req_len:{len(pending_req_distribution)}"

        replica_state.append(save_ratio)
        replica_state.extend(running_req_distribution + pending_req_distribution)
        return replica_state

    def get_running_state(self, batch_queue):
        '''
            running state目前考虑的就是 正在运行的request的剩余工作量的分布：
            -- prefill的任务就是 num_prefill_tokens - num_process_tokens
            -- decode的任务就是 num_decode_tokens - num_process_decode_tokens

        :param bin_width: 直方图的间隔大小，默认为 100
        :return: token_distribution: 一个列表，包含 (bin_start, count) 元组，
                                    例如 [(0, 5), (100, 3)] 表示有 5 个请求剩余 token 在 [0, 100) 范围内，
                                    有 3 个请求剩余 token 在 [100, 200) 范围内。
        '''
        running_state = []

        # 假定 bin_width 必须是正整数
        if not isinstance(self.bin_width, int) or self.bin_width <= 0:
            raise ValueError("bin_width must be positive integer.")

        # get running requests
        assert len(batch_queue) == 0 or len(batch_queue) == 1  # assume only one stage scheduler

        if len(batch_queue) == 0:
            # 没有正在运行的batch
            return self.cal_token_distribution([])


        for request in batch_queue:
            # TODO: 这里目前用真实的decode token数量来表示，后续改成预测的decode数量
            remaining_tokens = request.num_prefill_tokens + request.num_decode_tokens - request.num_processed_tokens

            # 只有在剩余工作量大于 0 时才计入分布
            if remaining_tokens > 0:
                running_state.append(remaining_tokens)

        return self.cal_token_distribution(running_state)

    def get_waiting_state(self, pending_requests):
        if len(pending_requests) == 0:
            return self.cal_token_distribution([])

        pending_state = []
        for request in pending_requests:
            pending_state.append(request.num_prefill_tokens + request.num_decode_tokens - request.num_processed_tokens)

        return self.cal_token_distribution(pending_state)

    def step(self, request: FullRequest):
        next_state = []

        matched_tokens, _ = self.tree_cache.match_prefix(request.input_token_ids)
        assert len(matched_tokens) <= len(request.input_token_ids)
        num_matched_tokens = len(matched_tokens)

        save_ratio = num_matched_tokens / len(request.input_token_ids)

        batch_queue = self.get_replica_stage_scheduler(0).get_batch_queue()
        if len(self._request_queue) > 0:
            pending_queue = self._request_queue + [request]
            next_waiting_state = self.get_waiting_state(pending_queue)

            next_running_state = self.get_running_state(batch_queue)
        else:
            next_waiting_state = self.get_waiting_state(self._request_queue)
            running_queue = batch_queue + [request]
            next_running_state = self.get_running_state(running_queue)

        next_state.append(save_ratio)
        next_state.extend(next_waiting_state + next_running_state)
        return next_state

    def cal_token_distribution(self, num_tokens: List[int]) -> List[int]:
        num_bins = ceil(self.max_tokens_per_request / self.bin_width)
        num_requests = len(num_tokens)

        # 使用 defaultdict 初始化分布，这样我们只需要处理有计数的 bin，
        # 后续再用一个循环确保所有 bin 都存在。
        token_distribution_counts = defaultdict(int)

        for remaining_tokens in num_tokens:
            # 计算 bin 的起始值，例如 150/100 * 100 = 100
            # remaining_tokens 已经被限制在 [1, max_token_per_request] 范围内
            bin_start = (remaining_tokens - 1) // self.bin_width * self.bin_width  # 确保剩余 1 token 落在 bin 0

            # 统计计数
            token_distribution_counts[bin_start] += 1

        # 5. 生成固定长度且有序的最终分布列表
        token_distribution = []
        for i in range(num_bins):
            bin_start = i * self.bin_width

            # 如果 bin_start 已经超过了 max_token_per_request，则停止
            if bin_start >= self.max_tokens_per_request:
                break

            # 获取计数，如果该 bin 没有请求，则计数值为 0
            count = token_distribution_counts[bin_start] / num_requests if num_requests > 0 else 0
            token_distribution.append(count)


        return token_distribution
    
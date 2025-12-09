from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.utils.radix_tree import RadixCache
from vidur.scheduler.utils.kv_block_manager import KVBlockManager
from vidur.entities.full_request import FullRequest
from vidur.entities.batch import Batch

from math import ceil
from typing import Set, List

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
        else:
            self._allocation_map[request_id] += num_blocks

        assert self.block_manager.num_used_blocks <= self._config.num_blocks

        if num_blocks == 0:
            return 
        
        # 分配新 Block
        new_blocks = [self.block_manager.allocate_block() for _ in range(num_blocks)]
        
        # 转成冗余的 block id 列表
        extend_block_table = []
        for i in range(num_token_required):
            index = i // self._config.block_size
            extend_block_table.append(new_blocks[index])
            self.block_manager.increment_slot(new_blocks[index])

        # 设置给 request block table
        request.block_table = request.block_table + extend_block_table

    def _can_allocate_request(self, request: FullRequest) -> bool:
        reused_block_ids, last_node_match = self.tree_cache.match_prefix(
            request.input_token_ids
        )

        num_matched_blocks = len(set(reused_block_ids))

        if request.id not in self._allocation_map:
            # new request
            num_matched_tokens = len(reused_block_ids)
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

        request.last_node = last_node_match
        num_matched_blocks = 0
        if len(request.block_table) == 0 and len(reused_block_ids) > 0:
            request.set_block_table(reused_block_ids)
            # 增加前缀树中被复用的block的引用计数
            # 遍历冗余列表，但只对唯一的 Block ID 增加引用计数
            seen_blocks: Set[int] = set(reused_block_ids)
            for block_id in seen_blocks:
                self.block_manager.increment_ref_count(block_id)

            num_matched_blocks = len(seen_blocks)
        
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
        # num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_reserved = len(request.block_table)
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request, 1)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self._free_request([request])
            else:
                if request.is_prefill_complete and request.num_processed_decode_tokens == 1:
                    self._prepare_for_decode(request)
                self._preempted_requests.append(request)

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

        last_block_id = request.block_table[-1] # 一般认为block table最后一个元素就是最后用的那个block的id
        if self.block_manager.has_free_slots(last_block_id):
            request.append_block(last_block_id)
            self.block_manager.increment_slot(last_block_id)
        else:
            self.allocate(request, 1)   # 这里change了 block_table，也修改了slot

    def _get_request_next_num_tokens(
        self, request: FullRequest, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        # decode
        if request.is_prefill_complete:
            return 1

        # prefill 
        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,
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
                    victim_request.restart()
                    self._free_request([victim_request])
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    # 如果没有可抢占的request，则把当前request放回队列的开头
                    request.restart()
                    self._free_request([request.id])
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
            # block.slot - n
            uni_blocks = Set()
            for block_id in request.block_table:
                self.block_manager.decrement_slot(block_id)
                if block_id not in uni_blocks:
                    self.block_manager.free_block(block_id)
                    uni_blocks.add(block_id)

            # node.ref_count - 1
            # 会从last开始往祖先回溯，全部 - 1
            self.tree_cache.dec_lock_ref(request.last_node)
            


    # TODO: 这个逻辑还需要确定一下
    def cache_request(self, request: FullRequest) -> None:
        if request.completed:
            self.tree_cache.cache_finished_req(request)
        else:
            self.tree_cache.cache_unfinished_req(request)
    
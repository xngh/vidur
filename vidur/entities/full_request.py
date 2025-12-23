from typing import List, Optional, Dict, Any
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)

import torch
from transformers import AutoTokenizer

GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

class FullRequest(Request):
    """
    Represents a complete LLM generation request, including the actual 
    input/output content and a KV cache block table.
    Inherited from Request, and adds token content and KV cache management functionality.
    """
    def __init__(
        self,
        req_id: str,
        arrived_at: float,
        input_str: str = "",
        output_str: str = "", # 原始期望的输出字符串
        parent_unified_request: Optional['UnifiedRequest'] = None, 
        num_processed_tokens: int = 0,
        max_tokens: int = 4096,
    ):
        self.req_id = req_id
        self.input_str = input_str
        self.output_str = output_str

        # 存储 token ID 列表
        self.input_token_ids: List[int] = GLOBAL_TOKENIZER.encode(
            self.input_str, add_special_tokens=False
        )
        self.output_token_ids: List[int] = GLOBAL_TOKENIZER.encode(
            self.output_str, add_special_tokens=False
        )
        self.generated_token_ids: List[int] = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        # - in chunked prefill, fill_ids will be updated chunk by chunk
        # - it means already computed tokens' id
        # - it should be updated in replica stage scheduler
        self.fill_ids: List[int] = []

        num_prefill_tokens = len(self.input_token_ids)
        num_decode_tokens = len(self.output_token_ids)
        super().__init__(arrived_at, num_prefill_tokens, num_decode_tokens, 
                         num_processed_tokens)
        
        # --- KV Cache Block ---
        # 存储分配给这个请求的 物理 Block ID
        self.block_table: List[int] = []  # 这里存储冗余的block id
        
        self.parent_unified_request = parent_unified_request

        # Prefix info
        self.prefix_indices: List[int] = []    # 记录request当前处理的 token 的 block id列表, len(prefix_indices) == len(fill_ids)

        self.last_node: Any = None

        self.num_matched_tokens = 0               # 用于replica_scheduler get_next_token时使用
        self.max_tokens = max_tokens              # 用于归一化request state
    
    # --- Reload Request Methods ---

    # 没有直接调用父类的原有逻辑，而是在其基础上修改
    # 原因是一些判断逻辑可以复用，就不判断多次了
    def on_batch_end(self, time, num_tokens_processed):
        # super().on_batch_end(time, num_tokens_processed)
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time
        self.sim_output_tokens(num_tokens_processed)   # make fill_ids matches num_tokens_processed

        assert self._num_processed_tokens <= self.total_tokens
        # for prefill request matches more tokens than it can process in this chunk
        # eg。 Assume [a b c d] has been cached。 req = [a b c d e f g h] chunk = 2
        # num_tokens_processed may be 2, but req.num_tokens_processed should be 6
        # that's because the reused token can be considered as zero cost
        # (TODO: yinhan) 这里还是有点问题 把partial prefill 和 chunk太小的问题混起来了，而且两者解决方式还不一样
        # if self.is_prefill_complete == False and self.num_processed_tokens < len(self.block_table):
        #    self.num_processed_tokens = len(self.block_table)
        # assert self._num_processed_tokens == len(self.block_table), f"{self._num_processed_tokens} != {len(self.block_table)}"

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1
            # move this logic to replica_scheduler.on_batch_end
            # because sim_output_tokens(1) without add this token to a block is incorrect
            # self.sim_output_tokens(1)

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        #logger.debug(f"Request #{self.req_id}'s progress: {self._num_processed_tokens} tokens processed")

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            # this assert is wrong for prefill-only request. At request.on_batch_end, the first output
            # token doesn't add to self.generated_token_ids, but in replica_scheduler.on_batch_end()
            # so I remove this assertation.
            # assert len(self.generated_token_ids) == len(self.output_token_ids), f"{len(self.generated_token_ids)=} == {len(self.output_token_ids)=}"
            self._completed_at = time
            self._completed = True
            self.parent_unified_request.current_step_index += 1
            logger.debug(f"Request {self._id} completed at {self._completed_at}")
        return

    def restart(self):
        logger.debug(f"Restarting request {self._id}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens
        self.fill_ids = []
        self.generated_token_ids = []
        self.block_table = []
        self.prefix_indices = []
        self.last_node = None
        self.num_matched_tokens = 0

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts += 1

    # --- KV Cache Block Methods ---
    
    def set_block_table(self, block_table: List[int]):
        """Sets the block table for this request."""
        self.block_table = block_table
    
    def get_block_table(self) -> List[int]:
        """Returns the block table for this request."""
        return self.block_table
    
    def append_block(self, block_id: int):
        """Appends a block ID to the block table."""
        self.block_table.append(block_id)
    
    def get_last_block_id(self) -> Optional[int]:
        """Get the last block ID in the block table."""
        if not self.block_table:
            return None
        return self.block_table[-1]
    
    # --- Generation Methods ---

    # 这个函数应该在ReplicaStageScheduleEvent中调用
    def append_generated_token_id(self, token_id: int) -> bool:
        """
        追加一个新生成的 token ID。
        返回 True 如果生成完成，否则返回 False。
        """
        self.generated_token_ids.append(token_id)
        # 在这里不直接修改 self._num_processed_tokens，而是依赖于外部的 on_batch_end
        # 假设 on_batch_end 会增加计数
        
        return len(self.generated_token_ids) >= self._num_decode_tokens

    # --- Utility ---

    def sim_output_tokens(self, num_tokens: int):
        last = len(self.fill_ids)
        total_token_ids = self.input_token_ids + self.output_token_ids

        # change fill ids
        self.fill_ids = self.fill_ids + total_token_ids[last : last + num_tokens]

        # change generated_token_ids
        if self._is_prefill_complete:
            last_decode = len(self.generated_token_ids)
            self.generated_token_ids = self.generated_token_ids + self.output_token_ids[
                last_decode : last_decode + num_tokens
            ]


        assert len(self.fill_ids) <= len(total_token_ids)
        assert len(self.generated_token_ids) <= len(self.output_token_ids)

    def to_dict(self) -> Dict:
        """Overrides the parent to_dict() to include new fields."""
        data = super().to_dict()
        data.update({
            "input_str": self.input_str,
            "output_str": self.output_str,
            "input_token_ids": self.input_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "block_table": self.block_table,
        })
        return data

    def get_state(self):
        state = []

        state.append(self.pd_ratio)  # pd_ratio
        state.append(self.num_prefill_tokens / self.max_tokens)
        state.append(self.num_decode_tokens / self.max_tokens)

        # 当前request在 workflow中的位次
        current_progress_of_workflow = 0 if self.parent_unified_request.total_steps == 1 \
            else self.parent_unified_request.current_step_index / (self.parent_unified_request.total_steps - 1)
        state.append(current_progress_of_workflow)

        return state
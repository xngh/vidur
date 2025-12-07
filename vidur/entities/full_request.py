from typing import List, Optional, Dict, Any
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)

import torch
from transformers import AutoTokenizer

GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")

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

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        
        self.parent_unified_request = parent_unified_request

        # Prefix info
        self.prefix_indices: List[int] = []    # 记录request当前处理的 token 的 block id列表, len(prefix_indices) == len(fill_ids)
        # Number of tokens to run prefill.
        self.extend_input_len = 0

        self.last_node: Any = None
    
    # --- Reload Request Methods ---

    # 没有直接调用父类的原有逻辑，而是在其基础上修改
    # 原因是一些判断逻辑可以复用，就不判断多次了
    def on_batch_end(self, time, num_tokens_processed):
        # super().on_batch_end(time, num_tokens_processed)
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time
        self.sim_output_tokens(num_tokens_processed)

        assert self._num_processed_tokens <= self.total_tokens

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1
            self.sim_output_tokens(1)

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            assert len(self.generated_token_ids) == len(self.output_token_ids)
            self._completed_at = time
            self._completed = True
            logger.debug(f"Request {self._id} completed at {self._completed_at}")
        return 


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

    def release_all_blocks(self, block_manager: 'KVCacheManager'):
        """
        Notify the BlockManager to release all physical blocks held by this request.
        """
        for block_id in self.block_table:
            block_manager.free_block(block_id)
        
        self.block_table = []
    
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
        last = len(self.generated_token_ids) - 1
        total_token_ids = self.input_token_ids + self.output_token_ids

        self.fill_ids = self.fill_ids + total_token_ids[last : last + num_tokens]
        if self._is_prefill_complete:
            last_decode = len(self.generated_token_ids)
            self.generated_token_ids = self.generated_token_ids + total_token_ids[
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
from collections import deque
from typing import Dict, List, Optional
import math
import sys
from typing import Set

DEFAULT_BLOCK_SIZE = 16

class KVBlockManager:
    """
    仿真环境下的 KV Cache Block 内存管理器 (Block Manager)。
    负责 Block ID 的分配、回收和引用计数。
    """
    def __init__(self, num_total_blocks: int, block_size: int = DEFAULT_BLOCK_SIZE):
        """
        初始化 Block Manager。

        Args:
            num_total_blocks: KV 缓存中总共可用的 Block 数量。
            block_size: 每个 Block 存储的 token 数量。
        """
        if num_total_blocks <= 0:
            raise ValueError("Total number of blocks must be positive.")
            
        self._num_total_blocks = num_total_blocks
        self._block_size = block_size
        
        # 空闲 Block ID 池：存储可供分配的 Block ID
        self.free_blocks: deque[int] = deque(range(self._num_total_blocks))
        
        # 引用计数器：Key=Block ID, Value=引用次数
        self.ref_counts: Dict[int, int] = {}
        
        # 跟踪当前分配给活跃请求的 Block ID 集合
        self.allocated_blocks: Set[int] = set()

        # 跟踪每个block还有多少剩余slot，最大为block_size  被遗弃
        # self.block_slots: Dict[int, int] = {}


        print(f"BlockManager initialized: {self._num_total_blocks} blocks of size {self._block_size}.")

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return len(self.ref_counts)

    def allocate_block(self) -> int:
        """
        分配一个空闲的 Block ID。
        将新分配的 Block 的引用计数设置为 1。
        
        Raises:
            Exception: 如果内存不足。
        
        Returns:
            int: 分配的 Block ID。
        """
        if not self.free_blocks:
            raise Exception("Out of Memory: No free KV cache blocks available.")


        # 从空闲池中取出 Block ID
        block_id = self.free_blocks.popleft()

        if block_id == 10:
            print("hook here.")
        # 初始化引用计数
        self.ref_counts[block_id] = 1
        self.allocated_blocks.add(block_id)
        
        return block_id

    def free_block(self, block_id: int):
        """
        释放一个 Block ID (引用计数减 1)。
        如果引用计数降至 0，则将该 Block ID 回收到空闲池。
        
        Args:
            block_id: 待释放的 Block ID。
        """
        if block_id not in self.ref_counts:
            print(f"Warning: Attempting to free Block ID {block_id} which is not tracked or already freed.")

        # 减少引用计数
        self.ref_counts[block_id] -= 1
        
        # 检查是否可以回收
        if self.ref_counts[block_id] == 0:
            del self.ref_counts[block_id]
            self.allocated_blocks.remove(block_id)
            self.free_blocks.append(block_id)

    def increment_ref_count(self, block_id: int):
        """
        增加一个 Block ID 的引用计数。
        当 RadixCache 发现前缀复用时调用。
        
        Args:
            block_id: 待增加引用的 Block ID。
        """
        if block_id not in self.ref_counts:
            # 如果 Block ID 不在引用计数中，说明它可能是一个新的或已被回收的 Block
            # 在正常流程中，不应该对未分配的 Block 增加引用计数
            raise ValueError(f"Cannot increment ref count for Block ID {block_id}: Block is not allocated.")
            
        self.ref_counts[block_id] += 1
        # print(f"Block ID {block_id} ref count increased to {self.ref_counts[block_id]}.")

    def get_ref_count(self, block_id: int) -> int:
        return self.ref_counts.get(block_id, 0)

    def increment_ref_for_blocks(self, block_ids: List[int]) -> None:
        uni_block = set(block_ids)

        for block_id in uni_block:
            self.increment_ref_count(block_id)

    def decrement_ref_for_blocks(self, block_ids: List[int]) -> None:
        uni_block = set(block_ids)

        for block_id in uni_block:
            self.free_block(block_id)

    def has_free_slots(self, block_id: int) -> bool:
        if block_id not in self.block_slots:
            raise ValueError(f"Block ID {block_id}: Block is not allocated.")

        if self.block_slots[block_id] < self._block_size:
            return True
        else:
            return False

    def increment_slot(self, block_id: int) -> None:
        if block_id not in self.block_slots:
            self.block_slots[block_id] = 1
            return

        self.block_slots[block_id] += 1

    def decrement_slot(self, block_id: int) -> None:
        if block_id not in self.block_slots:
            raise ValueError(f"Block ID {block_id}: Block is not allocated.")

        self.block_slots[block_id] -= 1
        if self.block_slots[block_id] == 0:
            del self.block_slots[block_id]
    
    def print_status(self):
        """打印 BlockManager 的当前状态摘要。"""
        print("-" * 30)
        print(f"BlockManager Status:")
        print(f"Total Blocks: {self._num_total_blocks}")
        print(f"Used Blocks (Total Refs): {self.num_used_blocks}")
        print(f"Free Blocks: {self.num_free_blocks}")
        print(f"Reference Counts (Top 5):")
        # 打印引用次数最高的 5 个 Block
        sorted_refs = sorted(self.ref_counts.items(), key=lambda item: item[1], reverse=True)
        for block_id, count in sorted_refs[:5]:
            print(f"  Block {block_id}: {count} references")
        print("-" * 30)

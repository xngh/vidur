from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from typing import Tuple, Dict

from vidur.scheduler.utils.prefix_cache import BasePrefixCache
from vidur.scheduler.utils.kv_block_manager import KVBlockManager

from vidur.entities.full_request import FullRequest
from functools import lru_cache, partial
from vidur.logger import init_logger

logger = init_logger(__name__)

# python -m vidur.scheduler.utils.radix_tree
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py#L358


# 存储的是对应于 node.key ( token_ids ) 的 KV 缓存的 物理内存索引 kv_indices，通常是一个 torch.Tensor
# 现在我想把他改成 block id，也就是每个token所对应的block id。比如[I am yinhan] -> [0, 0, 1] 假设一个
# block 存两个token
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None # 存储 token id
        self.value = None # 存储 token id 对应的 kv_indices   仿真中改成block id
        self.lock_ref = 0 # 表明当前节点存储的 token kv 正在被几个 request 使用
        self.last_access_time = time.time() # 上一次该 node 存储的 token kv 被使用的时间

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i

# 这里计算的是child节点在parent节点中查询的key
def get_child_key(key: List, page_size: int = 1):
    if page_size == 1:
        plain_key = key[0]
    else:
        plain_key = tuple(key[:page_size])

    return plain_key


# TODO(yinhan)： 当前实现是SGLang的某个版本的radixTree实现。由于SGLang是真实的物理引擎，因此他需要保存所谓的KV
# 这在仿真的情况下是不需要的，我只要判断token id是否匹配即可，也就是说所谓的value应该不需要存在了
# 这里需要修改一下。token_2_kv_pool应该也不需要存在了

class RadixCache(BasePrefixCache):
    def __init__(
        self,
        block_manager: KVBlockManager,
        page_size: int,
        disable: bool = False,
    ):
        self.block_manager = block_manager
        self.disable = disable
        self.page_size = page_size   # 等于 block size


        if self.page_size == 1:
            self.key_match_fn = _key_match
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
    
    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    def match_prefix(self, key: List, **kwargs) -> Tuple[List, TreeNode]:
        if self.disable or len(key) == 0:
            return [], self.root_node

        # 块对齐
        key = self._page_align_keys(key)
        
        if len(key) == 0:
            return [], self.root_node

        value, last_node = self._match_prefix_helper(self.root_node, key)
        
        return value, last_node

    def insert(self, key: List, value=None, chunked=False, priority: int = 0):
        if self.disable:
            return 0
        
        if value is None:
            raise TypeError("value cannot be None")

        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: FullRequest, is_insert: bool = True):
        """Cache request when it finishes."""
        if self.disable:
            return

        token_ids = (req.input_token_ids + req.generated_token_ids)
        # TODO(yinhan) 这个地方需要进行修改
        #kv_indices = self.req_to_token_pool.req_to_token[
        #    req.req_pool_idx, : len(token_ids)
        #]
        block_ids = req.get_block_table()[:len(token_ids)]

        keys = req.fill_ids
        assert len(req.fill_ids) == len(token_ids)
        keys = self._page_align_keys(keys)
        #values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        values = block_ids[: len(keys)]

        # Radix Cache takes one ref in memory pool
        if is_insert:
            new_prefix_len = self.insert(keys, values, priority=0)    # 每次 insert 都会给新的node的block的id + 1
            # Free the duplicates that were already in the tree

        # Remove req slot release the cache lock
        # 在local_scheduler 处 free block
        self.dec_lock_ref(req.last_node)


    # TODO(yinhan): 感觉这里需要对block进行操作，就像SGlang中一样
    def cache_unfinished_req(self, req: FullRequest, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        block_ids = req.get_block_table()[: len(token_ids)]

        keys = req.fill_ids
        # for len(token_ids) < page_size, SGLang doesn't insert these token into the radix tree
        keys = self._page_align_keys(keys)
        values = block_ids[: len(keys)]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            keys,
            values,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )

        # The prefix indices could be updated, reuse it
        (new_indices, new_last_node) = self.match_prefix(keys)

        logger.debug(f"new indices: {new_indices}")
        logger.debug(f"token ids: {keys}")

        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}, {len(block_ids)=}"

        #self.req_to_token_pool.req_to_token[
        #    req.req_pool_idx, len(req.prefix_indices) : len(new_indices)
        #] = new_indices[len(req.prefix_indices) :]
        # req.get_block_table()[len(req.prefix_indices) : len(new_indices)] = new_indices[len(req.prefix_indices) :]

        # if there is a node which have cache the keys, use the origin block in the node.value
        # this case only happened in decode request, when the output_len % block_size == 0
        block_table = req.get_block_table()
        if block_table[: len(new_indices)] != new_indices:
            self.block_manager.decrement_ref_for_blocks(block_table[: len(new_indices)])
            self.block_manager.increment_ref_for_blocks(new_indices)

        req.get_block_table()[: len(new_indices)] = new_indices

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        # - I don't know what's the operation means
        if len(new_indices) < len(block_ids):
            req.prefix_indices = new_indices + block_ids[len(new_indices) :]
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_blocks: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves) # heapq 是一个堆队列（heap_queue), 将列表 leaves 转换为一个最小堆（就地操作）, node 中定义了 __lt__ 函数，即最后一次被访问时间越早的 node 在堆的上面

        num_evicted = 0
        while num_evicted < num_blocks and len(leaves):
            x = heapq.heappop(leaves) # 从堆中弹出最小的元素

            if x == self.root_node:
                break
            if x.lock_ref > 0: # 表明这个节点存储的 block 正在被其他请求使用，不能删除，跳过
                continue
            
            # 这里的 evict_callback 函数，释放 node 对应的 value：block id
            self._evict_block(x.value) 
            num_evicted += len(set(x.value))
            self._delete_leaf(x) # 删除该 node 及其所有的child node

            if len(x.parent.children) == 0: # 一些叶子节点被删除后，可能会产生新的叶子节点，收集这些新的叶子节点
                heapq.heappush(leaves, x.parent)

    def _evict_block(self, block_ids: List[int]):
        uni_block_ids = set(block_ids)
        for block_id in uni_block_ids:
            assert self.block_manager.ref_counts[block_id] == 1    # 只剩下radix tree 引用该 block
            self.block_manager.free_block(block_id)



    def inc_lock_ref(self, node: TreeNode): # 将匹配到的节点 node 以及它的所有的祖先节点的 lock_ref 加一
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode): # 将匹配到的节点 node 以及它的所有的祖先节点的 lock_ref 减一
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1

            if node.parent is None:
                assert (
                        node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    # 输入 key 是 prompt_token_ids, 该函数的目的是:尽可能的找到输入 prompt_token_ids 中前 n 个已经被缓存了的 kv 的 token （姑且叫做前向最大匹配），
    # 并将前 n 已经被缓存了 kv 的 token 的 kv_indices 添加到 value 中
    # 那在仿真中，只需要知道 n 的大小即可，不需要kv indices
    # 当前修改value为node所对应的block id
    def _match_prefix_helper(self, node: TreeNode, key: List):
        access_time = time.time()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value += new_node.value
                node = new_node
                break
            else:
                value += child.value
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    # 我的设想是每个节点的value存储的是 block id，就是这个节点的kv所对应的block的序号
    # 这里应该要传入整个路径的value，一个新的 req 的 block id 应该是从 radix tree 中复用 + 新申请的
    def _insert_helper(self, node: TreeNode, key: List, value: List):
        access_time = time.time()
        node.last_access_time = access_time
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        if 7951 in value:
            print(" ")
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)

            # 增加 radix tree 对 block 的引用
            uni_value = set(value)
            for block_id in uni_value:
                self.block_manager.increment_ref_count(block_id)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key, f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node: TreeNode): # 删除 node 及其所有的child node
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.key)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self): # 收集所有的叶子节点（叶子节点:该节点没有）
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, False)

    prompt_token_ids = [1,2,3,4,5,6,7,8] # 假设第一个 prompt 的 token_ids 是 [1,2,3,4,5,6,7,8]
    tree.insert(prompt_token_ids)
    tree.pretty_print()

    prompt_token_ids_2 = [1,2,3,4,5,6,9,10] # 假设第二个 prompt 的 token_ids 是 [1,2,3,4,9,10,7,8]
    kv_indices, _ = tree.match_prefix(prompt_token_ids_2)
    print(f"\nkv_indices:{kv_indices}\n")
    tree.insert(prompt_token_ids_2)
    tree.pretty_print()

    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(4, evict_callback)
    tree.pretty_print()
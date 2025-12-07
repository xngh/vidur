
from vidur.scheduler.utils.radix_tree import RadixCache


def test_evict_case1():
    print("------------Case1------------")
    tree = RadixCache(None, 1000, False)

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

    # result
    # 8 [1, 2, 3, 4, 5, 6, 7, 8] r=0
    # tokens: 8

    # kv_indices:[[1, 2, 3, 4, 5, 6]]

    # 6 [1, 2, 3, 4, 5, 6] r=0
    #   2 [7, 8] r=0
    #   2 [9, 10] r=0
    # tokens: 10
    # 6 [1, 2, 3, 4, 5, 6] r=0
    # tokens: 6



def test_evict_case2():
    print("------------Case2------------")
    tree = RadixCache(None, 1000, False)

    prompt_token_ids = [1,2,3,4,5,6,7,8] # 假设第一个 prompt 的 token_ids 是 [1,2,3,4,5,6,7,8]
    tree.insert(prompt_token_ids)
    tree.pretty_print()

    prompt_token_ids_2 = [1,2,3,4,9,10,7,8] # 假设第二个 prompt 的 token_ids 是 [1,2,3,4,9,10,7,8]
    kv_indices, _ = tree.match_prefix(prompt_token_ids_2)
    print(f"\nkv_indices:{kv_indices}\n")
    tree.insert(prompt_token_ids_2)
    tree.pretty_print()

    # 4 [1, 2, 3, 4] r=0
    #    4 [5, 6, 7, 8] r=0
    #    4 [9, 10, 7, 8] r=0

    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(2, evict_callback)
    tree.pretty_print()

    # 4 [1, 2, 3, 4] r=0
    #   4 [9, 10, 7, 8] r=0

    # 这么看来evict是以node为单位，而不是block为单位的

def test_evict_case3():
    print("------------Case2------------")
    tree = RadixCache(None, 1000, False)

    prompt_token_ids = [1,2,3,4,5,6,7,8] # 假设第一个 prompt 的 token_ids 是 [1,2,3,4,5,6,7,8]
    tree.insert(prompt_token_ids)
    tree.pretty_print()

    prompt_token_ids_2 = [1,2,3,4,5,6,7,8] # 假设第二个 prompt 的 token_ids 是 [1,2,3,4,9,10,7,8]
    kv_indices, _ = tree.match_prefix(prompt_token_ids_2)
    print(f"\nkv_indices:{kv_indices}\n")
    tree.insert(prompt_token_ids_2)
    tree.pretty_print()

    # 4 [1, 2, 3, 4] r=0
    #    4 [5, 6, 7, 8] r=0
    #    4 [9, 10, 7, 8] r=0

    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(2, evict_callback)
    tree.pretty_print()

from vidur.scheduler.utils.kv_block_manager import KVBlockManager
from vidur.entities.full_request import FullRequest
def test_insert_case1():
    tree = RadixCache(KVBlockManager(10), 4, False)

    req1 = FullRequest(
        req_id = "req1",
        arrived_at= 0.0,
        input_str="What do you like to eat?",
        output_str="I like to eat apple.",
        parent_unified_request= None,
        num_processed_tokens= 0
    )

    save_block_ids, last_node = tree.match_prefix(req1.input_token_ids)
    req1.last_node = last_node

    req1.append_generated_token_id(req1.input_token_ids[0])

    tree.cache_unfinished_req(req1)

    tree.pretty_print()

# python -m vidur.scheduler.utils.radix_tree_test

if __name__ == "__main__":
    
    test_insert_case1()
from vidur.config import SimulationConfig
from vidur.entities import Replica
from vidur.entities.unified_request import UnifiedRequest
from vidur.scheduler.replica_scheduler.local_replica_scheduler import LocalReplicaScheduler
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from vidur.scheduler.utils.radix_tree import RadixCache
from vidur.test_agent_simulator import TestAgentSimulator
from vidur.utils.random import set_seeds


def init_test_scheduler(block_size=4, chunk_size=4096):
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    config.cluster_config.replica_scheduler_config.block_size = block_size
    config.cluster_config.replica_scheduler_config.chunk_size = chunk_size

    scheduler = ReplicaSchedulerRegistry.get(
        config.cluster_config.replica_scheduler_config.get_type(),
        replica_config=config.cluster_config.replica_config,
        replica_scheduler_config=config.cluster_config.replica_scheduler_config,
        request_generator_config=config.request_generator_config,
        replica=Replica(config.cluster_config.replica_config, config.request_generator_config),
        num_stages=1,
        execution_time_predictor=None,
    )

    assert isinstance(scheduler, LocalReplicaScheduler)
    return scheduler

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

def test_lock_ref():
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    config.cluster_config.replica_scheduler_config.block_size = 16

    set_seeds(config.seed)

    simulator = TestAgentSimulator(config)

    req1 = FullRequest(
        req_id="req1",
        arrived_at=0.0,
        input_str="1 2 3 4 5 6 7 8 9 10 11 12 13 14 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 11 12",
        output_str="1 2 3 4",
        parent_unified_request=None,
        num_processed_tokens=0
    )
    localScheduler = simulator.scheduler.get_replica_scheduler(0)
    tree = localScheduler.tree_cache

    localScheduler._allocate_request(req1)
    req1.fill_ids = req1.input_token_ids
    tree.cache_unfinished_req(req1)
    tree.pretty_print()

    req2 = FullRequest(
        req_id="req2",
        arrived_at=0.0,
        input_str="1 2 3 4 5 6 7 8 9 10 11 12 13 14 5 6",
        output_str="1 2",
        parent_unified_request=None,
        num_processed_tokens=0
    )
    req2.fill_ids.extend(req2.input_token_ids)

    localScheduler._allocate_request(req2)
    tree.cache_unfinished_req(req2)
    tree.pretty_print()

    localScheduler._free_request([req1])
    tree.pretty_print()

    localScheduler._free_request([req2])
    tree.pretty_print()


def test_block_increment():
    scheduler = init_test_scheduler(block_size=4)

    req1 = FullRequest(
        req_id="req1",
        arrived_at=0.0,
        input_str="1 2 3 4 5 6 7 8",
        output_str="1 2 3 4",
        parent_unified_request=None,
        num_processed_tokens=0
    )

    scheduler._allocate_request(req1)
    block_ids = set(req1.get_block_table())

    for block_id in block_ids:
        ref_count = scheduler.tree_cache.block_manager.get_ref_count(block_id)
        assert ref_count == 1

    req1.sim_output_tokens(8)
    scheduler.cache_request(req1)

    scheduler.tree_cache.pretty_print()

    for block_id in block_ids:
        ref_count = scheduler.tree_cache.block_manager.get_ref_count(block_id)
        assert ref_count == 2

    req2 = FullRequest(
        req_id="req2",
        arrived_at=0.0,
        input_str="1 2 3 4",
        output_str="1 2 3 4",
        parent_unified_request=None,
        num_processed_tokens=0
    )
    scheduler._allocate_request(req2)
    block_ids_2 = set(req2.get_block_table())

    for block_id in block_ids:
        ref_count = scheduler.tree_cache.block_manager.get_ref_count(block_id)
        print(f"block_id: {block_id}, ref_count：{ref_count}")

    req2.sim_output_tokens(4)
    scheduler.cache_request(req2)

    scheduler.tree_cache.pretty_print()

    scheduler._free_request([req2])

    for block_id in block_ids:
        ref_count = scheduler.tree_cache.block_manager.get_ref_count(block_id)
        print(f"block_id: {block_id}, ref_count：{ref_count}")
    print("---------------------------")
    scheduler._free_request([req1])

    scheduler.tree_cache.pretty_print()
    for block_id in block_ids:
        ref_count = scheduler.tree_cache.block_manager.get_ref_count(block_id)
        print(f"block_id: {block_id}, ref_count：{ref_count}")

    scheduler._allocate_request(req2)

    scheduler.tree_cache.evict(1)
    scheduler.tree_cache.pretty_print()

    scheduler.cache_request(req2)
    scheduler.tree_cache.pretty_print()


def test_block_ref_with_node_ref():
    '''
        测试block的引用计数与node的引用计数是否始终满足差1的关系
    '''
    scheduler = init_test_scheduler(block_size=4)

    req1 = FullRequest(
        req_id="req1",
        arrived_at=0.0,
        input_str="1 2 3 4 5 6 7 8",
        output_str="1 2",
        parent_unified_request=None,
        num_processed_tokens=0
    )
    scheduler._allocate_request(req1)
    req1.sim_output_tokens(8)
    scheduler.cache_request(req1)

    block_ids = set(req1.get_block_table())
    node = scheduler.tree_cache.root_node

    key = req1.input_token_ids
    child_key = scheduler.tree_cache.get_child_key_fn(key)


    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        prefix_len = scheduler.tree_cache.key_match_fn(child.key, key)
        if prefix_len < len(child.key):
            break
        else:
            value = child.value
            for block_id in value:
                ref_count = scheduler.block_manager.get_ref_count(block_id)
                assert child.lock_ref == ref_count - 1, f"{child.lock_ref=}, {ref_count=}."
            node = child
            key = key[prefix_len:]

            if len(key):
                child_key = scheduler.tree_cache.get_child_key_fn(key)

    unireq1 = UnifiedRequest(
        workflow_id="req2",
        arrive_at=0,
        workflow_config=[{"step": "step1", "input_str": "a b c d e f g h", "output_str": "1 2"}]
    )

    req2 = unireq1.get_next_requests(0)[0]
    scheduler.add_request(req2)
    batch = scheduler._get_next_batch()
    batch.on_batch_end(0)
    scheduler.on_batch_end(batch)

    batch = scheduler._get_next_batch()
    batch.on_batch_end(0.1)
    scheduler.on_batch_end(batch)


def test_block_ref_with_node_case2():

    # 构造一个已经有一个节点的 radix_tree
    block_manager = KVBlockManager(4, 4)
    tree = RadixCache(block_manager, 4)

    block_manager.ref_counts[0] = 2
    block_manager.allocated_blocks.add(0)
    tree.insert([1, 2, 3, 4], [0,0,0,0])

    req = FullRequest(
        req_id="req1",
        input_str="1 2 3",
        output_str="4",
        arrived_at=0.0,
    )

    block_manager.ref_counts[1] = 1
    block_manager.allocated_blocks.add(1)
    req.set_block_table([1, 1, 1, 1])
    req.fill_ids = [1, 2, 3, 4]
    req.generated_token_ids = req.output_token_ids

    tree.cache_finished_req(req)
# python -m vidur.scheduler.utils.radix_tree_test

if __name__ == "__main__":
    test_block_ref_with_node_case2()
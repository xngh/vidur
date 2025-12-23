
from vidur.scheduler.replica_scheduler.local_replica_scheduler import LocalReplicaScheduler
from vidur.entities.replica import Replica
from vidur.config.config import BaseReplicaSchedulerConfig, ReplicaConfig
from vidur.request_generator import RequestGeneratorRegistry

from vidur.entities.full_request import FullRequest 
from vidur.config import SimulationConfig

from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from typing import List

from transformers import AutoTokenizer

GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

# python -m vidur.scheduler.replica_scheduler.local_replica_scheduler_test --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 1 --replica_scheduler_config_type local



def init_test_scheduler(block_size = 4, chunk_size = 4096):
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


def fill_request_ids(req: FullRequest):
    req.fill_ids = req.input_token_ids[: len(req.block_table)] if len(req.block_table) <= req.num_prefill_tokens else req.input_token_ids

def test_allocate_for_request():
    scheduler = init_test_scheduler() 

    req1 = FullRequest(
        req_id= "req1",
        arrived_at= 0,
        input_str= "what is the meaning of life?",
        output_str= "I don't know",
    )

    scheduler._allocate_request(req1)

    # 这里手动更新req的fill ids
    fill_request_ids(req1)

    scheduler.cache_request(req1)

    scheduler.tree_cache.pretty_print()

    req2 = FullRequest(
        req_id= "req2",
        arrived_at= 0,
        input_str= "what is the meaning of life, can you tell me?",
        output_str= "I don't know",
    )

    scheduler._allocate_request(req2)
    fill_request_ids(req2)

    scheduler.cache_request(req2)
    scheduler.tree_cache.pretty_print()

def test_allocate_for_request_case2():
    scheduler = init_test_scheduler(block_size = 4)


    req1 = FullRequest(
        req_id="req1",
        arrived_at=0,
        input_str="a b c d",
        output_str="g h",
    )

    scheduler._allocate_request(req1)
    fill_request_ids(req1)

    scheduler.cache_request(req1)

    req2 = FullRequest(
        req_id="req2",
        arrived_at=0,
        input_str="a b c d e f",
        output_str="g h",
    )

    scheduler._allocate_request(req2)
    fill_request_ids(req2)
    scheduler.cache_request(req2)

    print(" ")



def test_node_split():
    scheduler = init_test_scheduler()

    req1 = FullRequest(
        req_id= "req1",
        arrived_at= 0,
        input_str= "a b c d e f g h",
        output_str= "i j k l m n",
    )
    scheduler._allocate_request(req1)
    fill_request_ids(req1)

    scheduler.cache_request(req1)

    scheduler.tree_cache.pretty_print()

    req2 = FullRequest(
        req_id= "req2",
        arrived_at= 0,
        input_str= "a b c d",
        output_str= "e f g h",
    )

    scheduler._allocate_request(req2)
    scheduler.tree_cache.pretty_print()

    fill_request_ids(req2)
    scheduler.cache_request(req2)
    scheduler.tree_cache.pretty_print()


def test_matched_tokens_update():
    '''
        Test that matched tokens are updated correctly
        在FullRequest中添加了一个成员变量叫做 num_matched_tokens, 用于记录当前请求命中的token数

    '''
    scheduler = init_test_scheduler()
    req1 = FullRequest(
        req_id= "req1",
        arrived_at= 0,
        input_str= "a b c d",
        output_str= "e",
    )
    
    scheduler._allocate_request(req1)
    print(req1.block_table)
    fill_request_ids(req1)
    scheduler.cache_request(req1)
    scheduler.tree_cache.pretty_print()

    req2 = FullRequest(
        req_id= "req2",
        arrived_at= 0,
        input_str= "a b c d",
        output_str= "f",
    )

    scheduler._can_allocate_request(req2)
    assert req2.num_matched_tokens == 4
    next_token = scheduler._get_request_next_num_tokens(req2, True, 0)
    assert next_token == 0

def test_partial_prefill():
    chunk_size = 4
    scheduler = init_test_scheduler(block_size=8, chunk_size=chunk_size)

    req1 = FullRequest(
        req_id="req1",
        arrived_at=0,
        input_str="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
        output_str="a b c",
    )
    scheduler.add_request(req1)
    batch = scheduler._get_next_batch()
    assert batch.num_tokens[0] == chunk_size, "error in init scheduler."
    batch.on_batch_end(0)

    scheduler.cache_request(req1)
    scheduler.tree_cache.pretty_print()

    scheduler.add_request(req1)
    batch = scheduler._get_next_batch()
    batch.on_batch_end(0.5)
    scheduler.cache_request(req1)
    scheduler.tree_cache.pretty_print()

if __name__ == "__main__":
    test_allocate_for_request_case2()
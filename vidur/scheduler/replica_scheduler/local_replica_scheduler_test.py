
from vidur.scheduler.replica_scheduler.local_replica_scheduler import LocalReplicaScheduler
from vidur.entities.replica import Replica
from vidur.config.config import BaseReplicaSchedulerConfig, ReplicaConfig
from vidur.request_generator import RequestGeneratorRegistry

from vidur.entities.full_request import FullRequest 
from vidur.config import SimulationConfig

from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from typing import List


# python -m vidur.scheduler.replica_scheduler.local_replica_scheduler_test --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 1 --replica_scheduler_config_type local



def init_test_scheduler():
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    config.cluster_config.replica_scheduler_config.block_size = 4

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

def test_request_free():
    pass

if __name__ == "__main__":
    test_node_split()
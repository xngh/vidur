from vidur.scheduler.utils.kv_block_manager import KVBlockManager
from vidur.config import ReplicaConfig
from vidur.entities.replica import Replica

from vidur.scheduler.replica_scheduler.local_replica_scheduler import (
    LocalReplicaScheduler,
)
from vidur.config import SimulationConfig

from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry
from typing import List
from vidur.entities.full_request import FullRequest

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

def test_ref_count():
    # 测试插入request时的ref_count，尤其是新的node
    scheduler = init_test_scheduler()

    req1 = FullRequest(
        req_id= "req1",
        arrived_at= 0,
        input_str= "a b c d",
        output_str= "",
    )

    scheduler._allocate_request(req1)
    block_manager = scheduler.block_manager
    print("request's block table: ", req1.block_table)
    print("block ref counter: ", block_manager.ref_counts)
    
    fill_request_ids(req1)
    scheduler.cache_request(req1)

    print("after cache in radix tree, block ref counter: ", block_manager.ref_counts)

    req2 = FullRequest(
        req_id= "req2",
        arrived_at= 0,
        input_str= "a b c d e f g h",
        output_str= "",
    )
    scheduler._allocate_request(req2)
    fill_request_ids(req2)
    
    scheduler.cache_request(req2)
    print("request's block table: ", req2.block_table)
    print("block ref counter: ", block_manager.ref_counts)

    assert block_manager.ref_counts[0] == 3 and block_manager.ref_counts[1] == 2

    scheduler._free_request([req2])
    print("after evict, block ref counter: ", block_manager.ref_counts)

if __name__ == "__main__":
    test_ref_count()
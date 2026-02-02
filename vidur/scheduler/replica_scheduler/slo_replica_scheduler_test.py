from vidur.entities.replica import Replica
from vidur.config import SimulationConfig
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from vidur.scheduler.replica_scheduler.slo_replica_scheduler import SLOReplicaScheduler


# Example:
# python -m vidur.scheduler.replica_scheduler.slo_replica_scheduler_test \
#   --request_generator_config_type synthetic \
#   --synthetic_request_generator_config_num_requests 1 \
#   --replica_scheduler_config_type slo


def test_registry_constructs_slo_scheduler():
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    config.cluster_config.replica_scheduler_config.block_size = 4

    scheduler = ReplicaSchedulerRegistry.get(
        config.cluster_config.replica_scheduler_config.get_type(),
        replica_config=config.cluster_config.replica_config,
        replica_scheduler_config=config.cluster_config.replica_scheduler_config,
        request_generator_config=config.request_generator_config,
        replica=Replica(
            config.cluster_config.replica_config, config.request_generator_config
        ),
        num_stages=1,
        execution_time_predictor=None,
    )

    assert isinstance(scheduler, SLOReplicaScheduler)


if __name__ == "__main__":
    test_registry_constructs_slo_scheduler()



from vidur.test_agent_simulator import TestAgentSimulator
from vidur.config import SimulationConfig
from vidur.utils.random import set_seeds
from vidur.entities.unified_request import UnifiedRequest

# 测试unified request的创建是否正常
def test_unireq_creation() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    config.cluster_config.replica_scheduler_config.block_size = 4

    set_seeds(config.seed)

    simulator = TestAgentSimulator(config)

    unified_requests = []
    
    req1 = UnifiedRequest(
            workflow_id="req1",
            arrive_at=0,
            workflow_config=[{"step": "step1", "input_str": "a b c d", "output_str": "e f g h"}, {"step": "step2", "input_str": "h i j k", "output_str": "l m n"}]
        )
    unified_requests.append(req1)

    simulator._init_app_queue(unified_requests)

    assert len(simulator._app_queue) == 1

    simulator._init_event_queue()

    assert len(simulator._event_queue) == 1


if  __name__ == "__main__":
    test_unireq_creation()
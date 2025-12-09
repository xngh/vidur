
from vidur.test_agent_simulator import TestAgentSimulator
from vidur.config import SimulationConfig
from vidur.utils.random import set_seeds
from vidur.entities.unified_request import UnifiedRequest
import heapq

# 测试unified request的创建是否正常
def test_unireq_creation() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    config.cluster_config.replica_scheduler_config.block_size = 16

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


def handle_one_event(simulator: TestAgentSimulator) -> None:
    _, event = heapq.heappop(simulator._event_queue)
    print(f"Handling one {str(event)} event")
    simulator._set_time(event._time)
    new_events = event.handle_event(simulator._scheduler, simulator._metric_store)

    simulator._add_events(new_events)

def test_event_creation() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    config.cluster_config.replica_scheduler_config.block_size = 16

    set_seeds(config.seed)

    simulator = TestAgentSimulator(config)

    unified_requests = []

    req1 = UnifiedRequest(
        workflow_id="req1",
        arrive_at=0,
        workflow_config=[{"step": "step1", "input_str": "a b c d", "output_str": "e"},
                         {"step": "step2", "input_str": "h i j k", "output_str": "l m n"}]
    )
    unified_requests.append(req1)

    simulator._init_app_queue(unified_requests)
    simulator._init_event_queue()


    # handle request_arrival_event
    handle_one_event(simulator)

    # handle global_schedule_event  当前是rr scheduler
    handle_one_event(simulator)

    # handle replica_schedule_evnet 当前是local_scheduler
    handle_one_event(simulator)

    # handle batch_stage_arrival_event
    handle_one_event(simulator)

    # handle replica_stage_schedule_event
    handle_one_event(simulator)

    # handle batch_stage_end_event
    handle_one_event(simulator)

    # handle batch_end_event
    handle_one_event(simulator)
    handle_one_event(simulator)

    # handle another replica_schedule event
    handle_one_event(simulator)


if  __name__ == "__main__":
    test_event_creation()
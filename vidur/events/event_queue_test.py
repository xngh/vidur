from vidur.entities.full_request import FullRequest
from vidur.events import RequestArrivalEvent
from vidur.utils.event_queue import EventQueue
import heapq

def test_priority():
    req1 = FullRequest(
            req_id="0",
            input_str="1",
            output_str="1 2",
            arrived_at=677.4584127,
            parent_unified_request=None,  # 链接回这个 UnifiedRequest 实例
            max_tokens=4096
    )
    event1 = RequestArrivalEvent(
        677.4584127, req1
    )

    req2 = FullRequest(
        req_id="1",
        input_str="2",
        output_str="1 2",
        arrived_at=711.753038,
        parent_unified_request=None, max_tokens=4096
    )

    event2 = RequestArrivalEvent(711.753038, req2)

    event_queue = []

    heapq.heappush(event_queue, (event1._priority_number, event1))
    heapq.heappush(event_queue, (event2._priority_number, event2))

    print(event_queue)


if __name__ == '__main__':
    test_priority()
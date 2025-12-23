from typing import List

from vidur.entities import Batch
from vidur.entities.full_request import FullRequest
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        from vidur.events.request_arrival_event import RequestArrivalEvent

        # after this, request's num_processed_token 进度与 fill_ids 相等， and should align with block table's size
        # except for prefill_completed request,which will output first decode token, this logic in replica_scheduler.prepare_for_decode
        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        # trigger next FullRequest for App
        new_request_events = []
        for request in self._batch.requests:
            if request._completed and isinstance(request, FullRequest):
                content = request.input_str + request.output_str
                next_requests = request.parent_unified_request.get_next_requests(self.time, content)

                for next_request in next_requests:
                    new_request_events.append(
                        RequestArrivalEvent(self.time, next_request)
                    )
                    logger.debug(f"Add a new request {next_request.req_id} to event queue:")


        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        return [ReplicaScheduleEvent(self.time, self._replica_id)] + new_request_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }

from typing import Dict, List, Tuple

from vidur.entities import Batch, Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class SharpGlobalScheduler(BaseGlobalScheduler):
    """
    Template custom global scheduler.
    Replace the scheduling logic in `schedule()` to fit your needs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0
        print("SharpGlobalScheduler init")

    def _estimate_batch_execution_time(self, replica_id: int, batch: Batch) -> float:
        stage_scheduler = self.get_replica_stage_scheduler(replica_id, 0)
        execution_time_predictor = stage_scheduler._execution_time_predictor
        num_stages = self._replicas[replica_id].num_pipeline_stages

        total_time = 0.0
        for stage_id in range(num_stages):
            execution_time = execution_time_predictor.get_execution_time(batch, stage_id)
            total_time += execution_time.total_time

        return total_time

    def _build_dummy_request(
        self, request: Request, is_prefill_complete: bool, num_processed_tokens: int
    ) -> Request:
        dummy_request = Request(
            arrived_at=0,
            num_prefill_tokens=request.num_prefill_tokens,
            num_decode_tokens=request.num_decode_tokens,
            num_processed_tokens=num_processed_tokens,
        )
        dummy_request._is_prefill_complete = is_prefill_complete
        return dummy_request

    def _get_matched_prefix_tokens(self, replica_id: int, request: Request) -> int:
        replica_scheduler = self._replica_schedulers[replica_id]
        if not hasattr(replica_scheduler, "tree_cache"):
            return 0
        if not hasattr(request, "input_token_ids"):
            return 0
        matched_blocks, _ = replica_scheduler.tree_cache.match_prefix(
            request.input_token_ids
        )
        return len(matched_blocks)

    #传入一个request，计算这个request的执行时间，包括prefill和decode两个部分。
    def _estimate_request_execution_time(self, replica_id: int, request: Request) -> float:
        matched_prefill_tokens = self._get_matched_prefix_tokens(replica_id, request)
        effective_prefill_processed = max(
            request.num_processed_prefill_tokens, matched_prefill_tokens
        )
        remaining_prefill_tokens = max(
            request.num_prefill_tokens - effective_prefill_processed, 0
        )
        remaining_decode_tokens = max(
            request.num_decode_tokens - request.num_processed_decode_tokens, 0
        )

        total_time = 0.0

        #估计prefill执行时间的时候就不考虑分chunk多次执行了，假设一次prefill完毕
        if remaining_prefill_tokens > 0:
            prefill_request = self._build_dummy_request(
                request,
                is_prefill_complete=False,
                num_processed_tokens=request.num_processed_tokens,
            )
            prefill_batch = Batch(replica_id, [prefill_request], [remaining_prefill_tokens])
            total_time += self._estimate_batch_execution_time(replica_id, prefill_batch)

        if remaining_decode_tokens > 0:
            decode_processed_tokens = max(
                request.num_processed_tokens, request.num_prefill_tokens
            )
            decode_request = self._build_dummy_request(
                request,
                is_prefill_complete=True,
                num_processed_tokens=decode_processed_tokens,
            )
            decode_batch = Batch(replica_id, [decode_request], [1])
            decode_time_per_token = self._estimate_batch_execution_time(
                replica_id, decode_batch
            )
            #计算deocde一个token的时间再乘以剩余的decode token数量。
            total_time += decode_time_per_token * remaining_decode_tokens

        return total_time

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []

        #replica排队时间目前只计算了等待队列里请求的执行时间，不包含正在运行的batch
        #假设decode全部串行执行，不考虑并发执行。这一项只是用于排序负载的，并不需要做到很精准。
        estimated_queue_time: Dict[int, float] = {}
        for replica_id, replica_scheduler in self._replica_schedulers.items():
            queue_time = 0.0
            for pending_request in replica_scheduler._request_queue:
                queue_time += self._estimate_request_execution_time(
                    replica_id, pending_request
                )
            estimated_queue_time[replica_id] = queue_time

        exec_weight = 1.0
        queue_weight = 0.1

        while self._request_queue:
            request = self._request_queue.pop(0)
            best_replica_id = None
            best_score = float("inf")

            for replica_id in sorted(self._replicas.keys()):
                exec_time = self._estimate_request_execution_time(replica_id, request)
                score = exec_weight * exec_time + queue_weight * estimated_queue_time[
                    replica_id
                ]
                if score < best_score:
                    best_score = score
                    best_replica_id = replica_id

            request_mapping.append((best_replica_id, request))
            estimated_queue_time[best_replica_id] += self._estimate_request_execution_time(
                best_replica_id, request
            )

        return request_mapping


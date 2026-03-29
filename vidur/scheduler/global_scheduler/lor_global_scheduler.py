from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        # 原始实现：按队列请求数量最少的副本进行调度
        # self.sort_requests()
        #
        # request_mapping = []
        # # keep a map of replica_id -> replica_scheduler
        # # this is used to find the replica with the least outstanding requests
        # pending_requests_map = {
        #     replica_scheduler.replica_id: replica_scheduler.num_pending_requests
        #     for replica_scheduler in self._replica_schedulers.values()
        # }
        #
        # # using a very simple implementation here, to keep wiring simple
        # while self._request_queue:
        #     request = self._request_queue.pop(0)
        #     replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
        #     pending_requests_map[replica_id] += 1
        #     request_mapping.append((replica_id, request))
        #
        # return request_mapping

        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # 负载指标 = 队列中请求的 prefill token + decode token 数量之和
        pending_token_load_map = {
            replica_scheduler.replica_id: sum(
                pending_request.num_prefill_tokens
                + pending_request.num_decode_tokens
                for pending_request in replica_scheduler._request_queue
            )
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_token_load_map.items(), key=lambda x: x[1])[0]
            pending_token_load_map[replica_id] += (
                request.num_prefill_tokens + request.num_decode_tokens
            )
            request_mapping.append((replica_id, request))

        return request_mapping

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from vidur.entities import Batch, Request
from vidur.entities.full_request import FullRequest
from vidur.scheduler.replica_scheduler.local_replica_scheduler import LocalReplicaScheduler


@dataclass
class _SJFDecision:
    idx: int
    request: FullRequest
    next_num_tokens: int
    est_time: float


class SLOReplicaScheduler(LocalReplicaScheduler):
    """
    SLO-oriented local replica scheduler.

    Changes vs `LocalReplicaScheduler`:
    - For *new* requests (not yet allocated / not prefill-complete), use SJF:
      rank by remaining prefill tokens (after prefix-cache matching) and pick the shortest one.

    Notes:
    - We deliberately avoid calling `_can_allocate_request()` during ranking,
      because it mutates request state + tree_cache ref counts for new requests.
      Allocation feasibility is checked only when we are about to actually schedule
      the chosen request.
    """

    # -------- estimation helpers (read-only; no scheduler state mutation) --------

    def _get_matched_prefix_tokens(self, request: FullRequest) -> int:
        if not hasattr(self, "tree_cache"):
            return 0
        if not hasattr(request, "input_token_ids"):
            return 0
        matched_blocks, _ = self.tree_cache.match_prefix(request.input_token_ids)
        return len(matched_blocks)

    def _compute_next_prefill_tokens_no_side_effect(
        self, request: FullRequest, num_batch_tokens: int
    ) -> int:
        """
        Mirror of `LocalReplicaScheduler._get_request_next_num_tokens()` for prefill,
        but without relying on `request.num_matched_tokens` being already populated.
        """
        if request.is_prefill_complete:
            return 1

        matched = self._get_matched_prefix_tokens(request)
        remaining = request.num_prefill_tokens - max(request.num_processed_tokens, matched)
        next_num_tokens = min(remaining, self._config.chunk_size - num_batch_tokens)
        return max(0, next_num_tokens)

    def _compute_remaining_prefill_tokens_no_side_effect(self, request: FullRequest) -> int:
        """
        Remaining prefill tokens to be processed (after accounting for prefix-cache hits),
        without mutating scheduler state.
        """
        if request.is_prefill_complete:
            return 0
        matched = self._get_matched_prefix_tokens(request)
        remaining = request.num_prefill_tokens - max(request.num_processed_tokens, matched)
        return max(0, remaining)

    def _build_dummy_request(
        self, request: FullRequest, is_prefill_complete: bool, num_processed_tokens: int
    ) -> Request:
        dummy_request = Request(
            arrived_at=0,
            num_prefill_tokens=request.num_prefill_tokens,
            num_decode_tokens=request.num_decode_tokens,
            num_processed_tokens=num_processed_tokens,
        )
        # predictor code checks this internal flag directly
        dummy_request._is_prefill_complete = is_prefill_complete
        return dummy_request

    def _estimate_batch_execution_time(self, batch: Batch) -> float:
        stage0 = self.get_replica_stage_scheduler(0)
        predictor = stage0._execution_time_predictor
        if predictor is None:
            # fallback: if no predictor wired, use token-count as a proxy
            return float(sum(batch.num_tokens))

        total = 0.0
        for stage_id in range(self._num_stages):
            total += predictor.get_execution_time(batch, stage_id).total_time
        return total

    def _estimate_request_next_prefill_chunk_time(
        self, request: FullRequest, next_num_tokens: int
    ) -> float:
        if next_num_tokens <= 0:
            return 0.0

        matched = self._get_matched_prefix_tokens(request)
        effective_prefill_processed = max(request.num_processed_prefill_tokens, matched)
        remaining_prefill = max(request.num_prefill_tokens - effective_prefill_processed, 0)
        prefill_chunk = min(remaining_prefill, next_num_tokens)
        if prefill_chunk <= 0:
            return 0.0

        # For prediction, reflect matched tokens as "already in kv" (kv_cache_size feature)
        dummy_req = self._build_dummy_request(
            request,
            is_prefill_complete=False,
            num_processed_tokens=effective_prefill_processed,
        )
        dummy_batch = Batch(self._replica_id, [dummy_req], [prefill_chunk])
        return self._estimate_batch_execution_time(dummy_batch)

    # -------- SJF selection --------

    def _rank_new_requests_sjf(
        self, contains_prefill: bool, num_batch_tokens: int
    ) -> List[_SJFDecision]:
        # only consider "new" requests: not allocated yet and not prefill-complete
        decisions: List[_SJFDecision] = []
        for idx, req in enumerate(self._request_queue):
            if req.id in self._allocation_map:
                continue
            if req.is_prefill_complete:
                continue

            next_num_tokens = self._compute_next_prefill_tokens_no_side_effect(
                req, num_batch_tokens
            )
            if next_num_tokens == 0:
                # fully cached or chunk already full; keep FIFO behavior via fallback path
                continue
            
            # try:
            #     est = self._estimate_request_next_prefill_chunk_time(req, next_num_tokens)
            # except Exception:
            #     est = float("inf")
            # SJF key: remaining prefill tokens (not just next chunk),
            # so requests with prefill > chunk_size are not all treated equally.
            est = float(self._compute_remaining_prefill_tokens_no_side_effect(req))

            decisions.append(
                _SJFDecision(
                    idx=idx, request=req, next_num_tokens=next_num_tokens, est_time=est
                )
            )

        decisions.sort(key=lambda d: (d.est_time, d.request.arrived_at))
        return decisions

    def _pop_next_new_request_sjf(
        self, contains_prefill: bool, num_batch_tokens: int
    ) -> Optional[FullRequest]:
        """
        Pick the next request to schedule from `_request_queue`:
        - Try SJF over new prefill requests (no side effects while ranking).
        - For each candidate in increasing predicted time order, call `_can_allocate_request()`.
          If it returns True, we immediately select and remove it from the queue.
        - If none can be allocated, return None.
        """
        ranked = self._rank_new_requests_sjf(contains_prefill, num_batch_tokens)
        if not ranked:
            return None

        for d in ranked:
            req = d.request
            if not self._can_allocate_request(req):
                continue

            # `_can_allocate_request()` has populated prefix/match state for this request.
            # We must schedule it immediately; now remove it from the queue.
            # Note: indices may have shifted since ranking; remove by identity.
            for i, r in enumerate(self._request_queue):
                if r is req:
                    return self._request_queue.pop(i)
            # should not happen
            return req

        return None

    # -------- override scheduling --------

    def _get_next_batch(self) -> Batch:
        # identical to `LocalReplicaScheduler._get_next_batch()` except the "new request" loop
        requests: List[FullRequest] = []
        num_tokens: List[int] = []
        skipped_requests: List[FullRequest] = []
        running_prefills: List[FullRequest] = []
        contains_prefill = False
        num_batch_tokens = 0

        # handle preempted requests first (kept as-is)
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    self._free_request([victim_request])
                    victim_request.restart()
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    self._free_request([request])
                    request.restart()
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # keep original FIFO-ish behavior for skipped preempted requests
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )
        skipped_requests = []

        # -------- NEW REQUEST LOOP (SJF) --------
        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            # chunk full → do not add new prefill requests
            if num_batch_tokens == self._config.chunk_size:
                break

            # Prefer SJF among new prefill requests
            sjf_req = self._pop_next_new_request_sjf(contains_prefill, num_batch_tokens)
            if sjf_req is None:
                # fallback to original FIFO head if SJF has no eligible candidates
                head = self._request_queue[0]
                if not self._can_allocate_request(head):
                    break
                request = self._request_queue.pop(0)
            else:
                request = sjf_req

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            # keep LocalReplicaScheduler's behavior: if chunk is full, stop
            if num_batch_tokens == self._config.chunk_size:
                self._request_queue = [request] + self._request_queue
                break

            self._allocate_request(request)

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)


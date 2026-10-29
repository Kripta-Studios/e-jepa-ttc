"""Opt-in five-millisecond ingestion with unchanged twenty-millisecond queries.

This backend adds no CUDA allocation and preserves the original bulk ingest
method. Calls remain serialized on the one history owner stream; returned query
tensors retain the parent's borrowed-output contract. No ingestion may run
concurrently with a query or with the service consuming that query's output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from operational.train40_runtime.integral import (
    BUCKET_COUNT,
    BUCKET_US,
    HISTORY_US,
    UPDATE_US,
    _validate_raw,
)
from operational.train40_runtime.temporal_prefix import PREFIX_COUNT, TemporalPrefixEventHistory


class SubpacketTemporalPrefixEventHistory(TemporalPrefixEventHistory):
    """Accept 5ms subpackets while enforcing the original event cap per 20ms.

    Four subpackets complete one original packet. Any ingest failure latches an
    invalid history until explicit reset. A rejected fourth subpacket cannot
    undo earlier causal ingestion: no query is admitted from that partial group.
    Parent counters remain lifetime totals; new counters are reset-scoped.
    """

    _subpacket_invalid: bool = False
    _subpacket_calls: int = 0
    _subpacket_completed_updates: int = 0
    _subpacket_pending_events: int = 0
    _subpacket_events: int = 0
    _subpacket_bulk_calls: int = 0
    _subpacket_last_update_events: int = 0

    @property
    def subpacket_phase(self) -> int:
        """Return the number of completed 5ms buckets in the active 20ms group."""
        return self._total_buckets % (UPDATE_US // BUCKET_US)

    @property
    def ready(self) -> bool:
        """Require complete history, a complete 20ms group and no ingest failure."""
        return super().ready and self.subpacket_phase == 0 and not self._subpacket_invalid

    @property
    def stats(self) -> dict[str, int | bool | str | None]:
        """Separate subpacket calls from complete original update packets."""
        result = super().stats
        result.update(
            backend="TEMPORAL_PREFIX_5MS_INGEST_20MS_QUERY",
            ingest_subpacket_us=BUCKET_US,
            query_stride_us=UPDATE_US,
            event_capacity_scope="SUM_OF_FOUR_SUBPACKETS_PER_20MS",
            subpacket_phase=self.subpacket_phase,
            subpacket_invalid=self._subpacket_invalid,
            subpacket_calls_since_reset=self._subpacket_calls,
            subpacket_completed_updates_since_reset=self._subpacket_completed_updates,
            subpacket_events_since_reset=self._subpacket_events,
            subpacket_pending_events=self._subpacket_pending_events,
            bulk_20ms_calls_since_reset=self._subpacket_bulk_calls,
            last_completed_subpacket_update_events=self._subpacket_last_update_events,
            subpacket_counter_scope="SINCE_EXPLICIT_RESET_PARENT_COUNTERS_REMAIN_LIFETIME",
        )
        return result

    def reset(self, start_us: int) -> None:
        """Establish the same parent clock/epoch and clear the failure/group state."""
        super().reset(start_us)
        self._subpacket_invalid = False
        self._subpacket_calls = 0
        self._subpacket_completed_updates = 0
        self._subpacket_pending_events = 0
        self._subpacket_events = 0
        self._subpacket_bulk_calls = 0
        self._subpacket_last_update_events = 0

    def _require_valid(self) -> None:
        if self._subpacket_invalid:
            raise RuntimeError("SUBPACKET_HISTORY_INVALID_RESET_REQUIRED")

    def _require_query_boundary(self) -> None:
        self._require_valid()
        if self.subpacket_phase:
            raise ValueError("query requires a complete group of four 5ms subpackets")

    def query(self, roi: Sequence[float]) -> Tensor:
        """Query only at the original cadence, preserving the borrowed tensor."""
        self._require_query_boundary()
        return super().query(roi)

    def query_many(self, rois: Sequence[Sequence[float]]) -> Tensor:
        """Query all objects only after the fourth subpacket has been ingested."""
        self._require_query_boundary()
        return super().query_many(rois)

    def ingest(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None:
        """Dispatch complete original packets or one explicit causal subpacket."""
        self._require_valid()
        try:
            if end_us - start_us == BUCKET_US:
                self.ingest_subpacket(raw, start_us, end_us)
                return
            if self.subpacket_phase:
                raise ValueError("bulk 20ms ingestion cannot interrupt a partial subpacket group")
            # Includes all original validation, capacity checks, copies and four
            # bucket operations. Do not reconstruct or rebase the original path.
            super().ingest(raw, start_us, end_us)
            self._subpacket_bulk_calls += 1
        except BaseException:
            self._subpacket_invalid = True
            raise

    def ingest_subpacket(self, raw: Mapping[str, np.ndarray], start_us: int, end_us: int) -> None:
        """Append one original bucket; reject aggregate overflow before its copy."""
        self._require_valid()
        try:
            if self.end_us is None or start_us != self.end_us or end_us - start_us != BUCKET_US:
                raise ValueError("subpacket clock rollback/gap: exact contiguous 5000us required")
            values = _validate_raw(raw, start_us, end_us, self.width, self.height)
            count = len(values["t"])
            if self._subpacket_pending_events + count > self.max_packet_events:
                raise ValueError("aggregate 20ms event capacity exceeded; no events truncated")
            if self._copy_event is not None and self._copy_pending and not self._copy_event.query():
                self._copy_event.synchronize()
                self._staging_waits += 1
            for key, value in values.items():
                host = torch.from_numpy(
                    np.where(value > 0, 1, -1).astype(np.int8) if key == "p" else value
                )
                if self.async_transfers:
                    self._pinned[key][:count].copy_(host)
                    host = self._pinned[key][:count]
                self._raw[key][:count].copy_(host, non_blocking=self.async_transfers)
            if self._copy_event is not None:
                self._copy_event.record(torch.cuda.current_stream(self.device))
                self._copy_pending = True
            flat = self._flat[:count]
            torch.mul(self._raw["y"][:count], self.width, out=flat)
            flat.add_(self._raw["x"][:count])
            torch.le(self._raw["p"][:count], 0, out=self._negative[:count])
            torch.mul(self._negative[:count], self.width * self.height, out=self._polarity[:count])
            flat.add_(self._polarity[:count])
            # The following operations match one iteration of the parent loop.
            # The packet is already exactly one half-open 5ms interval.
            c0, c1, c2 = self._count_work
            m0, m1, m2 = self._moment_work
            c0.zero_()
            m0.zero_()
            c0.view(-1).scatter_add_(0, flat, self._ones[:count])
            torch.sub(self._raw["t"][:count], self._epoch_start_us, out=self._relative[:count])
            m0.view(-1).scatter_add_(0, flat, self._relative[:count])
            torch.cumsum(c0, 1, dtype=torch.int32, out=c1)
            torch.cumsum(c1, 2, dtype=torch.int32, out=c2)
            torch.cumsum(m0, 1, out=m1)
            torch.cumsum(m1, 2, out=m2)
            self._cumulative_count[:, 1:, 1:].add_(c2)
            self._cumulative_moment[:, 1:, 1:].add_(m2)
            self._total_buckets += 1
            slot = self._total_buckets % PREFIX_COUNT
            if self._total_buckets % BUCKET_COUNT == 0:
                self._previous_count.copy_(self._cumulative_count)
                self._previous_moment.copy_(self._cumulative_moment)
                self._cumulative_count.zero_()
                self._cumulative_moment.zero_()
                self._epoch_start_us += HISTORY_US
            self._counts[slot].copy_(self._cumulative_count)
            self._moments[slot].copy_(self._cumulative_moment)
            self._filled = min(BUCKET_COUNT, self._filled + 1)
            self.end_us = end_us
            self._subpacket_calls += 1
            self._subpacket_events += count
            self._subpacket_pending_events += count
            if self.subpacket_phase == 0:
                self._ingested_packets += 1
                self._subpacket_completed_updates += 1
                self._subpacket_last_update_events = self._subpacket_pending_events
                self._subpacket_pending_events = 0
        except BaseException:
            self._subpacket_invalid = True
            raise

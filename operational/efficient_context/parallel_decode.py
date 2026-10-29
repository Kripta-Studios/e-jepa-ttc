"""Prefetch immutable native cache decoding without changing the scientific sampler."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .common import Campaign, atomic_json
from .parallel_inputs import ParallelInputCache


def decode_payload(payload: bytes | None, path: Path) -> tuple[bytes, tuple]:
    """Read and decode original NPZ bytes; the worker never writes cache or scientific state."""
    from .compressed_cache import CompressedInputCache

    if payload is None:
        payload = path.read_bytes()
    return payload, CompressedInputCache.unpack(payload)


class DecodingInputCache(ParallelInputCache):
    """Keep four raw workers and add four bounded CPU cache readers/decoders."""

    def __init__(self, c: Campaign, rows: dict) -> None:
        self.decode_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="native-decode")
        self.decoding: dict[str, Future] = {}
        self.decoded_hits = 0
        self.input_wait_seconds = 0.0
        self.input_records = 0
        self.timing_fit: str | None = None
        super().__init__(c, rows)

    def activate(self, fit: dict) -> None:
        """Reset incidental timing counters without altering the restored sampler or any RNG."""
        self.timing_fit = fit["key"]
        self.input_wait_seconds = 0.0
        self.input_records = 0
        super().activate(fit)

    def fill(self) -> None:
        """Look ahead only in the original ordered64-record queue, with no RNG consumption."""
        super().fill()
        for token, raw_future in self.pending:
            if len(self.decoding) >= 64:
                break
            if raw_future is not None or token in self.decoding:
                continue
            payload = self.values.get(token)
            path = self.root / self.token_keys[token]
            if payload is not None or path.exists():
                self.decoding[token] = self.decode_executor.submit(decode_payload, payload, path)

    def refresh_capacity(self) -> None:
        """Keep disposable inputs persistently under the user's separately amended disk cap."""
        from .expanded_disk import authorization
        from .fast_owned_scan import owned_bytes

        value = authorization(self.c)
        noncache = owned_bytes(self.c.out) - owned_bytes(self.root)
        self.limit = max(
            0,
            min(
                value["max_native_cache_bytes"],
                value["max_owned_bytes"] - value["atomic_checkpoint_reservation_bytes"] - noncache,
            ),
        )
        self.evict(0)
        atomic_json(
            self.c.out / "garl/CACHE_BUDGET.json",
            {
                "owned_noncache_bytes": noncache,
                "cache_limit_bytes": self.limit,
                "cached_bytes": self.disk,
                "atomic_checkpoint_reservation_bytes": value["atomic_checkpoint_reservation_bytes"],
                "owned_quota_bytes": value["max_owned_bytes"],
                "lossless_FP32": True,
            },
        )

    def get(self, token: str) -> tuple:
        """Measure main-thread input wait separately from the complete training update."""
        start = time.perf_counter()
        value = self._get(token)
        if getattr(self, "timing_fit", None) is not None:
            self.input_wait_seconds += time.perf_counter() - start
            self.input_records += 1
            if self.input_records % 1024 == 0:
                atomic_json(
                    self.c.out / "garl/INPUT_PREPARATION_TIMING.json",
                    {
                        "fit": self.timing_fit,
                        "session_input_records": self.input_records,
                        "main_thread_input_wait_seconds": self.input_wait_seconds,
                        "includes_read_decode_raw_misses_and_cache_publication": True,
                        "includes_GPU_training": False,
                        "raw_cache_reads": self.reads,
                        "cache_hits": self.hits,
                        "parallel_decoded_hits": self.decoded_hits,
                        "purpose": (
                            "operational diagnosis; not an R0 benchmark or scientific endpoint"
                        ),
                    },
                )
        return value

    def _get(self, token: str) -> tuple:
        """Consume the requested record in order; all cache mutations stay in the main thread."""
        job = self.decoding.pop(token, None)
        if job is None:
            return super().get(token)
        if not self.pending or self.pending[0][0] != token or self.pending[0][1] is not None:
            raise ValueError("cache decoding lookahead differs from the original sampler")
        sequence = self.rows[token]["sequence_id"]
        stat = (self.c.raw / sequence / "events.h5").stat()
        if (stat.st_size, stat.st_mtime_ns) != self.sequence_stats[sequence]:
            raise ValueError("native raw identity changed during CPU cache decoding")
        try:
            payload, value = job.result()
        except FileNotFoundError:
            # Main-thread cache eviction may precede a background reader. The original
            # get handles the retained RAM payload or prepares exactly the same raw input.
            return super().get(token)
        self.pending.popleft()
        key = self.token_keys[token]
        if token in self.values:
            self.values.move_to_end(token)
        elif key in self.files:
            self.files.move_to_end(key)
            self.remember_payload(token, payload, on_disk=True)
        else:
            self.remember_payload(token, payload, on_disk=False)
        self.hits += 1
        self.decoded_hits += 1
        self.fill()
        return value

    def drain(self) -> None:
        """Discard speculative CPU outputs without touching the original complete state."""
        for job in self.decoding.values():
            job.cancel()
        self.decoding.clear()
        super().drain()

    def close(self) -> None:
        """Join CPU decoding before clearing original cache storage and reader workers."""
        self.drain()
        self.decode_executor.shutdown(wait=True, cancel_futures=True)
        super().close()

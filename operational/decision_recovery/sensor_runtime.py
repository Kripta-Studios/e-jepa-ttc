"""Fixed, lossless sensor runtime for the V19 decision-recovery campaign.

The source packs remain the immutable V18 FP32 archives.  This module only
selects the already admitted parallel decoder, bounded decoded-row cache and
single speculative batch prefetch.  It never rewrites or recompresses packs.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from operational.joint_measurement.data import SensorDataset
from operational.joint_measurement.parallel_training_sensor import ParallelTrainingSensors

from . import common

GIB = 1024**3


@dataclass(frozen=True)
class SensorRuntimeConfig:
    """A registered runtime choice; it must not be tuned during a run."""

    workers: int
    cache_bytes: int
    speculative_batches: int
    source_pack_encoding: str = "immutable_v18_deflate_level1_fp32"

    def as_dict(self) -> dict[str, int | str]:
        """Return JSON-safe values for registration and performance receipts."""
        return {
            "workers": self.workers,
            "cache_bytes": self.cache_bytes,
            "speculative_batches": self.speculative_batches,
            "source_pack_encoding": self.source_pack_encoding,
        }


# Eight workers is the V18 admitted ceiling on this 32-logical-core host.  A
# single producer owns decoding, so there is no process replication of the LRU.
TRAINING = SensorRuntimeConfig(workers=8, cache_bytes=4 * GIB, speculative_batches=1)
QA = SensorRuntimeConfig(workers=8, cache_bytes=0, speculative_batches=1)
# Evaluation is a one-pass cohort traversal.  Retaining decoded rows cannot
# yield hits, so zero cache prevents hundreds of MiB of needless RAM retention.
EVALUATION = SensorRuntimeConfig(workers=8, cache_bytes=0, speculative_batches=0)


class RoleParallelSensors(ParallelTrainingSensors):
    """Use the exact parallel V18 decoder for a declared P or H role.

    ``ParallelTrainingSensors`` was intentionally P-only because it served
    random training.  V19 also needs exact full-cohort P/H evaluation.  This
    class changes only construction: all admission, duplicate handling,
    collation, LRU accounting and fingerprint checks remain inherited.
    """

    def __init__(self, role: str, *, config: SensorRuntimeConfig) -> None:
        if role not in {"P", "H"}:
            raise ValueError("Decision recovery admits only P or H sensors")
        if not 1 <= config.workers <= 8:
            raise ValueError("One to eight random-row decoder workers required")
        if config.cache_bytes < 0 or config.speculative_batches not in {0, 1}:
            raise ValueError("Invalid bounded sensor runtime configuration")

        # ParallelTrainingSensors has no role argument.  Initialize its two
        # bases explicitly, reproducing its state without opening a P reader.
        SensorDataset.__init__(self, role)
        self.limit_bytes = config.cache_bytes
        self.rows_cached: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.retained_bytes = 0
        self.fingerprints: dict[int, tuple] = {}
        self.hits = self.misses = self.peak_bytes = 0
        self.executor: ThreadPoolExecutor | None = None
        self.future = None
        self.shadow = None
        self.pinned = False

        self.workers = config.workers
        self.decoder = ThreadPoolExecutor(
            max_workers=config.workers,
            thread_name_prefix=f"v19-{role.lower()}-decode",
        )
        self.tokens = self.rows.sample_token.astype(str).tolist()
        self.current_rows: dict[int, dict[str, np.ndarray]] | None = None
        self.duplicate_draws = 0
        self.decoder_closed = False
        self.load_lock = threading.Lock()
        self.runtime_config = config

    def counters(self) -> dict:
        """Add the immutable role and fixed runtime contract to V18 counters."""
        return {
            **super().counters(),
            "role": self.role,
            "runtime_config": self.runtime_config.as_dict(),
        }


def make_training_sensors(*, qa: bool = False) -> RoleParallelSensors:
    """Build the fixed P reader used by QA or scientific training."""
    config = QA if qa else TRAINING
    sensors = RoleParallelSensors("P", config=config)
    return sensors


def make_evaluation_sensors(role: str) -> RoleParallelSensors:
    """Build a no-retention, exact full-cohort P/H evaluation reader."""
    return RoleParallelSensors(role, config=EVALUATION)


def runtime_contract() -> dict[str, dict[str, int | str] | str]:
    """Describe choices that must be hashed before CUDA QA or training."""
    return {
        "training": TRAINING.as_dict(),
        "qa": QA.as_dict(),
        "evaluation": EVALUATION.as_dict(),
        "sampler": "authoritative torch.Generator; shadow state only for one-batch prefetch",
        "ordering": "input indices and duplicate draws preserved exactly",
        "compression": "reuse immutable V18 bytes; no transcode or cache rewrite",
        "process_model": "one process, one prefetch thread, bounded decoder thread pool",
    }


def _exact_batch(left: object, right: object) -> bool:
    for field in fields(left):  # type: ignore[arg-type]
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def benchmark_cpu() -> dict:
    """Measure the fixed worker ceiling on four admitted B32 batches, without CUDA.

    The first batch warms each application reader and is excluded.  The next
    three batches are compared field-for-field with the serial native reader.
    This is a technical transport benchmark, never a scientific model metric.
    """
    starts = (0, 32, 64, 96)

    def consume(reader: object, *, keep: bool) -> tuple[list[float], list[object]]:
        elapsed, batches = [], []
        for position, start in enumerate(starts):
            tick = time.perf_counter()
            batch = reader.batch(list(range(start, start + 32)))  # type: ignore[attr-defined]
            duration = time.perf_counter() - tick
            if position:
                elapsed.append(duration)
                if keep:
                    batches.append(batch)
        return elapsed, batches

    serial = SensorDataset("P")
    try:
        serial_times, expected = consume(serial, keep=True)
    finally:
        serial.close()
    rows = []
    for workers in (1, 2, 4, 8):
        reader = RoleParallelSensors(
            "P", config=SensorRuntimeConfig(workers, cache_bytes=0, speculative_batches=0)
        )
        try:
            elapsed, observed = consume(reader, keep=True)
        finally:
            reader.shutdown()
        if len(observed) != len(expected) or not all(
            _exact_batch(left, right) for left, right in zip(expected, observed, strict=True)
        ):
            raise ValueError("Parallel sensor benchmark changed native batch bytes")
        rows.append(
            {
                "workers": workers,
                "seconds": elapsed,
                "mean_seconds": sum(elapsed) / len(elapsed),
                "speedup_over_serial": sum(serial_times) / sum(elapsed),
            }
        )
    pack = Path(serial.folder) / "pack_00000.zip"
    value = {
        "schema": "v19_sensor_cpu_benchmark_v1",
        "status": "PASS",
        "scope": "P pack0; application-warm B32; three timed batches; CPU only",
        "os_cache_state": "uncontrolled and likely warm; not a cold-storage benchmark",
        "timing_limitations": (
            "One source pack and one host observation; thread speedup is decoder-only, "
            "not end-to-end training throughput"
        ),
        "scientific_updates": 0,
        "serial_seconds": serial_times,
        "serial_mean_seconds": sum(serial_times) / len(serial_times),
        "parallel": rows,
        "selected_workers": TRAINING.workers,
        "selection_reason": "fixed V18 ceiling and fastest measured exact route",
        "end_to_end_training_speedup_claim": False,
        "source_pack_sha256": common.sha(pack),
        "source_receipt_sha256": common.sha(pack.with_suffix(".json")),
        "sensor_runtime_sha256": common.sha(Path(__file__)),
        "contract": runtime_contract(),
    }
    common.write_json(common.RUN / "performance/SENSOR_CPU_BENCHMARK.json", value)
    return value

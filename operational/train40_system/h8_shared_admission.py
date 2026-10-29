"""Verify real TRAIN raw preparation through the fixed shared-memory arenas."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.durable_io import atomic_json


def run(output: Path, raw_root: Path) -> None:
    """Compare each shared B16 buffer with the original canonical raw tensor byte for byte."""
    from operational.train40_system import history_resources8
    from operational.train40_system.h8_fast_admission import _job
    from operational.train40_system.h8_shared_pool import h8_shared_pool

    history_resources8.worker_init()
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        index = {
            key: stored[key] for key in ("windows_us", "square_xyxy", "sequences", "delta_t_s")
        }
    changes = np.flatnonzero(index["sequences"][1:] != index["sequences"][:-1]) + 1
    rows = [0, 31, 1000, *[int(row) for row in changes[:3]]]
    jobs = [_job(raw_root, row, index)[0] for row in rows]
    expected = [history_resources8.prepare(job) for job in jobs]
    observations = []
    started = time.perf_counter()
    with h8_shared_pool(
        max_workers=8, initializer=history_resources8.worker_init, profile=True
    ) as pool:
        pending = [pool.submit(history_resources8.prepare, job) for job in jobs]
        for row, reference, future in zip(rows, expected, pending, strict=True):
            actual = future.result()
            if (
                reference.shape != actual.shape
                or reference.dtype != actual.dtype
                or not np.array_equal(reference, actual)
                or reference.tobytes() != actual.tobytes()
            ):
                raise ValueError(f"Shared-memory H8 raw parity failed at TRAIN row {row}")
            observations.append(
                {
                    "row": row,
                    "raw_B16_bytes_exact": True,
                    "sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                    "worker": future.profile,
                }
            )
            del actual
        counters = pool.snapshot()
    if history_resources8._reader_pool is not None:
        history_resources8._reader_pool.close()
    torch.set_num_threads(1)
    atomic_json(
        output / "h8_fast_admission/SHARED_RAW_ADMISSION.json",
        {
            "status": "PASSED",
            "source_sha256": digest(Path(__file__)),
            "pool_source_sha256": digest(Path(__file__).with_name("h8_shared_pool.py")),
            "observations": observations,
            "elapsed_seconds": time.perf_counter() - started,
            "pool": counters,
            "optimizer_updates": 0,
            "GPU_used": False,
            "targets_read": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve())

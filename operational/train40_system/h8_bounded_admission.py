"""Real TRAIN CPU parity for dense-burst bounded union preparation."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.durable_io import atomic_json


@torch.inference_mode()
def main() -> None:
    """Compare identical fixed TRAIN windows without reading labels or writing features."""
    from operational.train40_system import history_resources8 as canonical
    from operational.train40_system.h8_bounded_union import encode_union
    from operational.train40_system.h8_fast_admission import _job

    output = ROOT / "artifacts/train40_system_20261005"
    destination = output / "h8_transport_admission/H8_BOUNDED_RAW_ADMISSION.json"
    rows = [0, 56075, 56090, 56120]
    report: dict[str, Any] = {
        "status": "RUNNING",
        "optimizer_updates": 0,
        "targets_read": False,
        "public_TRAIN40_only": True,
        "rows": rows,
        "started_utc": datetime.now(UTC).isoformat(),
        "source_sha256": digest(Path(__file__)),
        "candidate_source_sha256": digest(Path(__file__).with_name("h8_bounded_union.py")),
        "canonical_source_sha256": digest(Path(canonical.__file__)),
        "observations": [],
        "retained_bytes_max": 256 * 1024**2,
        "production_interrupted": False,
    }
    atomic_json(destination, report)
    canonical.worker_init()
    rng = torch.get_rng_state().clone()
    begun = time.perf_counter()
    try:
        with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
            index = {key: stored[key] for key in ("windows_us", "square_xyxy", "sequences")}
        for position, row in enumerate(rows):
            job, valid = _job(Path("E:/eAP_dataset"), row, index)
            report["current_row"] = row
            atomic_json(destination, report)
            values: dict[str, np.ndarray] = {}
            timings = {}
            profile: dict[str, Any] = {}
            order = ("canonical", "bounded") if position % 2 == 0 else ("bounded", "canonical")
            for candidate in order:
                start = time.perf_counter()
                if candidate == "canonical":
                    value = canonical.prepare(job)
                else:
                    assert canonical._reader_pool is not None
                    value = encode_union(
                        canonical._reader_pool.get(job["path"]),
                        job["windows"],
                        np.arange(15, -1, -1, dtype=np.int64) * 50_000,
                        np.concatenate((np.zeros(8, dtype=bool), valid)),
                        tuple(job["square"]),
                        sequence_id=job["sequence"],
                        roi_size=128,
                        event_pixel_diff=5,
                        retained_bytes_max=256 * 1024**2,
                        profile=profile,
                    ).numpy()
                timings[candidate] = time.perf_counter() - start
                values[candidate] = value
            left, right = values["canonical"], values["bounded"]
            if (
                left.dtype != right.dtype
                or left.shape != right.shape
                or not np.array_equal(left, right)
            ):
                raise ValueError(f"Bounded raw tensor differs at TRAIN row {row}")
            hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in (left, right)]
            if hashes[0] != hashes[1]:
                raise ValueError(f"Bounded raw tensor bytes differ at TRAIN row {row}")
            report["observations"].append(
                {
                    "TRAIN_ordinal": row,
                    "sequence": job["sequence"],
                    "timing_order": order,
                    "seconds": timings,
                    "profile": profile,
                    "exact_bytes": True,
                    "shape": list(right.shape),
                    "dtype": str(right.dtype),
                    "sha256": hashes[0],
                }
            )
            atomic_json(destination, report)
            del left, right, value, values
        if not torch.equal(rng, torch.get_rng_state()):
            raise ValueError("Raw preparation changed torch RNG")
        split_cases = sum(item["profile"]["split_count"] > 0 for item in report["observations"])
        if split_cases < 2:
            raise ValueError("Admission must exercise at least two genuinely oversized raw unions")
        report["oversized_union_cases"] = split_cases
        report.update(status="PASSED", all_tensors_exact_bytes=True, rng_unchanged=True)
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if canonical._reader_pool is not None:
            canonical._reader_pool.close()
        report["elapsed_seconds"] = time.perf_counter() - begun
        report["finished_utc"] = datetime.now(UTC).isoformat()
        atomic_json(destination, report)


if __name__ == "__main__":
    main()

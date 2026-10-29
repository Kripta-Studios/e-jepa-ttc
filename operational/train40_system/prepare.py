"""Lossless, bounded and fragment-resumable TRAIN40 producer input materialization."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from operational.efficient_context.common import ROOT, atomic_json, digest
from operational.train40_system.data_audit import OUTPUT

if TYPE_CHECKING:
    from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheConfig

_readers: dict = {}


def configuration() -> GarlTTCLHRCacheConfig:
    """Preserve the historical producer's full FP32 3×12×128×128 input recipe."""
    from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheConfig

    return GarlTTCLHRCacheConfig(
        store_full_frame_events=False,
        store_garl_event_roi=False,
        store_jepa_event_roi=False,
        store_event_v4_common_roi=True,
        materialize_splits=("train",),
        workers=4,
        event_v4_storage_dtype="float32",
    )


def worker_init() -> None:
    """Use four bounded processes with persistent per-process HDF5 readers."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def write_shard(task: dict) -> dict:
    """Publish a checked lossless shard and then its durable completion receipt."""
    from e_jepa_ttc.data.garlttc_calibration import CalibrationResolver
    from e_jepa_ttc.data.garlttc_lhr_cache import _materialize_row
    from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader

    begun = time.perf_counter()
    records = []
    config = configuration()
    for row in task["rows"]:
        path = Path(task["raw_root"]) / str(row["events_path"])
        if path not in _readers:
            # At most two sequence handles per worker, each with bounded chunk caches.
            while len(_readers) >= 2:
                _readers.pop(next(iter(_readers))).close()
            _readers[path] = CachedEventReader(path)
        record = _materialize_row(
            row,
            eap_root=Path(task["raw_root"]),
            config=config,
            event_readers=_readers,
            rgb_reader=None,
            mask_reader=None,
            first_track_timestamp_us=None,
            calibration=CalibrationResolver(),
        )
        records.append(record)
    events = np.stack([r["event_v4_common_roi"] for r in records])
    if events.dtype != np.float32 or events.shape[1:] != (3, 12, 128, 128):
        raise ValueError("Producer input shape or FP32 contract drifted")
    if not np.isfinite(events).all():
        raise ValueError("Nonfinite producer input")
    arrays = {
        "ordinals": np.asarray(task["ordinals"], dtype=np.int64),
        "events": events,
        "tokens": np.asarray([r["sample_token"] for r in records]),
        "target_ttc": np.asarray([r["ttc_s"] for r in records], dtype=np.float32),
        "visible_heights": np.stack([r["garl_visible_heights_px"] for r in records]),
        "boxes": np.stack([r["event_v4_boxes_xyxy"] for r in records]),
        "square": np.stack([r["event_v4_common_square_xyxy"] for r in records]),
        "motion": np.stack([r["observable_motion"] for r in records]),
        "delta": np.asarray([r["garl_delta_t_s"] for r in records], dtype=np.float32),
    }
    path = Path(task["destination"])
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as stored:
        for name, expected in arrays.items():
            if not np.array_equal(stored[name], expected):
                raise ValueError(f"Lossless shard roundtrip failed: {name}")
    os.replace(temporary, path)
    receipt = {
        "status": "VERIFIED",
        "ordinals": task["ordinals"],
        "rows": len(records),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - begun,
        "freeze_sha256": task["freeze_sha256"],
        "raw_sha256": task["raw_sha256"],
        "exact_compression_parity": True,
        "optimizer_updates": 0,
    }
    atomic_json(path.with_suffix(".json"), receipt)
    return receipt


def run(
    output: Path,
    raw_root: Path,
    *,
    maximum_shards: int | None = None,
) -> None:
    """Advance all available verified TRAIN inputs; never wait on a missing sequence."""
    import pandas as pd
    import psutil

    audit = json.loads((output / "DATA_AUDIT.json").read_text(encoding="utf-8"))
    if (
        audit["status"] != "PASSED"
        or digest(output / "TRAIN40_ROWS.parquet") != audit["rows_sha256"]
    ):
        raise ValueError("Complete pinned TRAIN40 annotation audit required")
    freeze_path = output / "PREPARE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["source_sha256"] != digest(Path(__file__)):
        raise ValueError("Preparation source differs from its QA freeze")
    if freeze["config"] != json.loads(json.dumps(asdict(configuration()))):
        raise ValueError("Preparation recipe differs from its QA freeze")
    for entry in freeze["files"]:
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen preparation dependency changed: {entry['path']}")
    raw_plan = json.loads((output / "RAW_PLAN.json").read_text(encoding="utf-8"))
    pins = {entry["sequence_id"]: entry for entry in raw_plan["files"]}
    rows = pd.read_parquet(output / "TRAIN40_ROWS.parquet").to_dict(orient="records")
    directory = Path(freeze.get("cache_root", str(output / "producer_inputs"))).resolve()
    if raw_root.resolve() == directory or raw_root.resolve() in directory.parents:
        raise ValueError("New cache must be separate from downloaded raw datasets")
    directory.mkdir(parents=True, exist_ok=True)
    verified = {}
    for sequence in audit["sequences"]:
        path = raw_root / "data/train" / sequence / "events.h5"
        new_receipt = output / "raw_receipts" / f"{sequence}.json"
        old_receipt = ROOT / (
            f"artifacts/efficient_context_20261004/data_recovery/files/{sequence}.json"
        )
        receipt_path = new_receipt if new_receipt.is_file() else old_receipt
        if not path.is_file() or not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] != "VERIFIED" or receipt["sha256"] != pins[sequence]["sha256"]:
            raise ValueError(f"Raw restoration contract differs for {sequence}")
        stat = path.stat()
        if stat.st_size != receipt["bytes"] or stat.st_mtime_ns != receipt["mtime_ns"]:
            raise ValueError(f"Raw restoration receipt is stale for {sequence}")
        verified[sequence] = receipt["sha256"]
    tasks = []
    completed_rows = 0
    completed_bytes = 0
    for start in range(0, len(rows), 32):
        group = rows[start : start + 32]
        destination = directory / f"shard_{start // 32:05d}.npz"
        receipt_path = destination.with_suffix(".json")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt["freeze_sha256"] != digest(freeze_path)
                or not destination.is_file()
                or digest(destination) != receipt["sha256"]
            ):
                raise ValueError(f"Existing preparation fragment changed: {destination}")
            completed_rows += receipt["rows"]
            completed_bytes += receipt["bytes"]
            continue
        sequences = {str(r["sequence_id"]) for r in group}
        if not sequences.issubset(verified):
            continue
        tasks.append(
            {
                "rows": group,
                "ordinals": list(range(start, start + len(group))),
                "raw_root": str(raw_root),
                "destination": str(destination),
                "freeze_sha256": digest(freeze_path),
                "raw_sha256": {s: verified[s] for s in sorted(sequences)},
            }
        )
    if maximum_shards is not None:
        tasks = tasks[:maximum_shards]
    pending = {}
    iterator = iter(tasks)
    deadline = json.loads((output / "AUTHORIZATION.json").read_text(encoding="utf-8"))[
        "deadline_utc"
    ]
    from datetime import datetime

    with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as pool:
        while True:
            can_submit = (
                psutil.virtual_memory().available >= 4 * 1024**3
                and completed_bytes < freeze["new_input_disk_cap_bytes"]
                and psutil.disk_usage(str(directory)).free
                >= freeze.get("cache_disk_free_reserve_bytes", 20_000_000_000)
                and psutil.disk_usage(str(output)).free >= 20_000_000_000
                and datetime.now().astimezone() < datetime.fromisoformat(deadline)
            )
            while len(pending) < 4 and can_submit:
                task = next(iterator, None)
                if task is None:
                    break
                pending[pool.submit(write_shard, task)] = task
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                try:
                    receipt = future.result()
                except Exception as exc:
                    atomic_json(
                        Path(task["destination"]).with_suffix(".failure.json"),
                        {
                            "error": str(exc),
                            "ordinals": task["ordinals"],
                            "optimizer_updates": 0,
                            "scientific_negative": False,
                        },
                    )
                    continue
                completed_rows += receipt["rows"]
                completed_bytes += receipt["bytes"]
            atomic_json(
                output / "INPUT_PREPARATION_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_rows": completed_rows,
                    "total_rows": len(rows),
                    "compressed_bytes": completed_bytes,
                    "verified_raw_sequences": len(verified),
                    "workers": 4,
                    "optimizer_updates": 0,
                },
            )
    atomic_json(
        output / "INPUT_PREPARATION_PROGRESS.json",
        {
            "status": "COMPLETE" if completed_rows == len(rows) else "PARTIAL_PRESERVED",
            "completed_rows": completed_rows,
            "total_rows": len(rows),
            "compressed_bytes": completed_bytes,
            "verified_raw_sequences": len(verified),
            "missing_raw_receipts": sorted(set(audit["sequences"]) - set(verified)),
            "optimizer_updates": 0,
            "resume_command": (
                "../e-jepa-ttc/.venv/Scripts/python.exe -m operational.train40_system.prepare "
                "--raw-root E:/eAP_dataset"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--maximum-shards", type=int)
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve(), maximum_shards=args.maximum_shards)

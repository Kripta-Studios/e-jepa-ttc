"""Bounded process parallelism around the frozen native H8 input transform."""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_union
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.efficient_context.common import digest
from operational.sota_evidence.prepare_test import freeze_inputs
from operational.sota_evidence.test_inputs import supported_job
from operational.train40_system.durable_io import atomic_json

_pool: ReaderPool | None = None


def worker_init() -> None:
    """Own one HDF5 reader per process; match the tested two-thread transform."""
    global _pool
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    _pool = ReaderPool()


def materialize(task: dict[str, Any]) -> dict[str, Any]:
    """Publish one lossless input fragment without accessing model weights."""
    if _pool is None:
        raise RuntimeError("Worker must own an initialized reader pool")
    original = task["job"]
    destination = Path(task["destination"])
    begin = time.monotonic()
    reader = _pool.get(original["path"])
    timestamps = reader.datasets["events/t"]
    job = supported_job(original, int(timestamps[0]), int(timestamps[-1]))
    events = encode_union(
        reader,
        job["windows"],
        np.arange(750000, -1, -50000, dtype=np.int64),
        np.concatenate((np.zeros(8, bool), job["valid"])),
        tuple(job["square"]),
        sequence_id=job["sequence"],
        roi_size=128,
        event_pixel_diff=5,
    ).numpy()[-8:]
    if events.shape != (8, 3, 12, 128, 128) or not np.isfinite(events).all():
        raise ValueError("Invalid prepared H8 sensor tensor")
    temporary = destination.with_suffix(".pending.npz")
    np.savez_compressed(temporary, events=events, valid=job["valid"])
    os.replace(temporary, destination)
    receipt = {
        "input_freeze_sha256": task["binding"],
        "sample_token": job["sample_token"],
        "sha256": digest(destination),
        "raw_sha256": task["raw_sha256"],
        "seconds": time.monotonic() - begin,
        "optimizer_updates": 0,
        "execution_sha256": task["execution_sha256"],
    }
    atomic_json(destination.with_suffix(".json"), receipt)
    return receipt


def run(
    inputs: Path,
    media: Path,
    raw_root: Path,
    output: Path,
    cache: Path,
    plan: Path,
    *,
    workers: int,
    wait_seconds: int,
) -> None:
    """Reuse verified fragments, process ready sequences, and wait only within a deadline."""
    if not 1 <= workers <= 4:
        raise ValueError("At most four CPU workers are admitted")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    frozen = freeze_inputs(inputs, media, raw_root, output)
    binding = digest(output / "INPUT_FREEZE.json")
    execution = {
        "source_sha256": digest(Path(__file__)),
        "workers": workers,
        "threads_per_worker": 2,
        "transform": "unchanged encode_union",
        "input_freeze_sha256": binding,
    }
    execution_path = output / "PARALLEL_PREPARATION.json"
    if (
        execution_path.exists()
        and json.loads(execution_path.read_text(encoding="utf-8")) != execution
    ):
        raise ValueError("Parallel preparation execution identity changed")
    atomic_json(execution_path, execution)
    execution_sha = digest(execution_path)
    pins = {
        item["path"]: item for item in json.loads(plan.read_text(encoding="utf-8-sig"))["files"]
    }
    rows = frozen["rows"]
    completed = set()
    for number, job in enumerate(rows):
        path = cache / f"query_{number:05d}.npz"
        if path.with_suffix(".json").exists():
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (
                receipt["input_freeze_sha256"] != binding
                or receipt["sample_token"] != job["sample_token"]
                or digest(path) != receipt["sha256"]
            ):
                raise ValueError("Prepared input fragment changed")
            completed.add(number)
    started = time.monotonic()
    deadline = started + wait_seconds
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        while len(completed) < len(rows):
            advanced = False
            for sequence in sorted({r["sequence"] for r in rows}):
                pending_rows = [
                    (i, r)
                    for i, r in enumerate(rows)
                    if i not in completed and r["sequence"] == sequence
                ]
                if not pending_rows:
                    continue
                raw = Path(pending_rows[0][1]["path"])
                if not raw.exists():
                    continue
                pin = pins[f"data/test/{sequence}/events.h5"]
                if raw.stat().st_size != pin["size"]:
                    raise ValueError("Raw input size differs from pinned release")
                expected = pin["lfs"]["oid"]
                identity = {
                    "bytes": raw.stat().st_size,
                    "mtime_ns": raw.stat().st_mtime_ns,
                    "sha256": expected,
                    "path": str(raw),
                }
                raw_receipt = output / "raw" / f"{sequence}.json"
                if raw_receipt.exists():
                    if json.loads(raw_receipt.read_text(encoding="utf-8")) != identity:
                        raise ValueError("Verified raw input changed")
                else:
                    atomic_json(
                        output / "PREPARATION_PROGRESS.json",
                        {
                            "status": "VERIFYING_RAW_SHA256",
                            "sequence": sequence,
                            "completed": len(completed),
                            "total": len(rows),
                            "utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    if digest(raw) != expected:
                        raise ValueError("Raw input SHA256 differs from pinned release")
                    atomic_json(raw_receipt, identity)
                # Only one pending tensor per worker, no unbounded executor queue.
                for start in range(0, len(pending_rows), workers):
                    if shutil.disk_usage(cache).free < 30 * 1024**3:
                        raise RuntimeError("Less than 30 GiB free cache space")
                    chunk = pending_rows[start : start + workers]
                    futures = [
                        pool.submit(
                            materialize,
                            {
                                "job": job,
                                "destination": str(cache / f"query_{number:05d}.npz"),
                                "binding": binding,
                                "raw_sha256": expected,
                                "execution_sha256": execution_sha,
                            },
                        )
                        for number, job in chunk
                    ]
                    for (number, _), future in zip(chunk, futures, strict=True):
                        future.result()
                        completed.add(number)
                    advanced = True
                    atomic_json(
                        output / "PREPARATION_PROGRESS.json",
                        {
                            "status": "PREPARING",
                            "completed": len(completed),
                            "total": len(rows),
                            "sequence": sequence,
                            "elapsed_s": time.monotonic() - started,
                            "utc": datetime.now(UTC).isoformat(),
                        },
                    )
            if len(completed) == len(rows) or time.monotonic() >= deadline:
                break
            if not advanced:
                atomic_json(
                    output / "PREPARATION_PROGRESS.json",
                    {
                        "status": "WAITING_FOR_DOWNLOAD",
                        "completed": len(completed),
                        "total": len(rows),
                        "utc": datetime.now(UTC).isoformat(),
                    },
                )
                time.sleep(30)
    atomic_json(
        output / "PREPARATION_RESULT.json",
        {
            "status": "COMPLETE" if len(completed) == len(rows) else "WAITING_FOR_DOWNLOAD",
            "completed": len(completed),
            "total": len(rows),
            "input_freeze_sha256": binding,
            "cache": str(cache.resolve()),
            "elapsed_s": time.monotonic() - started,
            "optimizer_updates": 0,
            "test_labels_read": False,
            "execution_sha256": execution_sha,
        },
    )


def main() -> None:
    """Prepare with a maximum of four CPU workers and a bounded download wait."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "media", "raw-root", "output", "cache", "plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-seconds", type=int, default=21600)
    args = parser.parse_args()
    run(
        args.inputs,
        args.media,
        args.raw_root,
        args.output,
        args.cache,
        args.plan,
        workers=args.workers,
        wait_seconds=args.wait_seconds,
    )


if __name__ == "__main__":
    main()

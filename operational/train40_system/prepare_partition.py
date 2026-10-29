"""Resume selected disjoint TRAIN40 inputs with retryable Windows metadata writes."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.prepare import write_shard


def extra_groups(rows: list[dict], original: set[str]) -> list[tuple[int, list[dict]]]:
    """Partitions are disjoint: the first pass accepts only all-original groups."""
    return [
        (start, rows[start : start + 32])
        for start in range(0, len(rows), 32)
        if not {str(row["sequence_id"]) for row in rows[start : start + 32]} <= original
    ]


def worker_init() -> None:
    """Keep the frozen CPU kernel; change only progress publication."""
    from operational.train40_system import prepare

    prepare.atomic_json = atomic_json
    prepare.worker_init()


def run(output: Path, raw_root: Path, partition: str, workers: int) -> None:
    """Reuse the QA-frozen bit-exact encoder with disjoint output ownership."""
    import pandas as pd
    import psutil

    freeze_path = output / "PREPARE_FREEZE.json"
    freeze = read(freeze_path)
    verify_sources(freeze)
    verify_sources(read(output / "PARTITION_PREPARE_FREEZE.json"))
    extra_freeze = read(output / "EXTRA_PREPARE_FREEZE.json")
    verify_sources(extra_freeze)
    original = set(extra_freeze["original_pass_sequences"])
    audit = read(output / "DATA_AUDIT.json")
    if len(original) != 31 or not original < set(audit["sequences"]):
        raise ValueError("Exact active 31-sequence task partition required")
    if digest(output / "TRAIN40_ROWS.parquet") != audit["rows_sha256"]:
        raise ValueError("Full TRAIN40 metadata changed")
    directory = Path(freeze["cache_root"])
    rows = pd.read_parquet(output / "TRAIN40_ROWS.parquet").to_dict(orient="records")
    tasks, complete = [], 0
    groups = [(start, rows[start : start + 32]) for start in range(0, len(rows), 32)]
    if partition == "original":
        groups = [
            (start, group)
            for start, group in groups
            if {str(row["sequence_id"]) for row in group} <= original
        ]
    for start, group in groups:
        path = directory / f"shard_{start // 32:05d}.npz"
        if path.with_suffix(".json").exists():
            receipt = read(path.with_suffix(".json"))
            if digest(path) != receipt["sha256"] or receipt["freeze_sha256"] != digest(freeze_path):
                raise ValueError("Completed extra input shard changed")
            complete += len(group)
            continue
        sequences = {str(row["sequence_id"]) for row in group}
        pins = {}
        for sequence in sequences:
            receipt_path = output / "raw_receipts" / f"{sequence}.json"
            if not receipt_path.exists():
                receipt_path = (
                    ROOT
                    / f"artifacts/efficient_context_20261004/data_recovery/files/{sequence}.json"
                )
            receipt = read(receipt_path)
            raw = raw_root / "data/train" / sequence / "events.h5"
            stat = raw.stat()
            if receipt["status"] != "VERIFIED" or (stat.st_size, stat.st_mtime_ns) != (
                receipt["bytes"],
                receipt["mtime_ns"],
            ):
                raise ValueError("Extra preparation raw sequence not verified")
            pins[sequence] = receipt["sha256"]
        tasks.append(
            {
                "rows": group,
                "ordinals": list(range(start, start + len(group))),
                "raw_root": str(raw_root),
                "destination": str(path),
                "freeze_sha256": digest(freeze_path),
                "raw_sha256": pins,
            }
        )
    iterator, pending = iter(tasks), {}
    deadline = datetime.fromisoformat(read(output / "AUTHORIZATION.json")["deadline_utc"])
    size, size_at = 0, float("-inf")
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        while True:
            if time.monotonic() - size_at > 30:
                size = sum(path.stat().st_size for path in directory.glob("shard_*.npz"))
                size_at = time.monotonic()
            allowed = (
                psutil.virtual_memory().available >= 4 * 1024**3
                and size + 500_000_000 < freeze["new_input_disk_cap_bytes"]
                and psutil.disk_usage(str(directory)).free >= 100_000_000_000
                and psutil.disk_usage(str(output)).free >= 20_000_000_000
                and datetime.now(UTC) < deadline
            )
            while len(pending) < workers and allowed:
                task = next(iterator, None)
                if task is None:
                    break
                pending[pool.submit(write_shard, task)] = task
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    receipt = future.result()
                    complete += receipt["rows"]
                except Exception as exc:
                    atomic_json(
                        Path(task["destination"]).with_suffix(".failure.json"),
                        {"error": str(exc), "scientific_negative": False, "optimizer_updates": 0},
                    )
            atomic_json(
                output / f"INPUT_{partition.upper()}_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_extra_rows": complete,
                    "workers": workers,
                    "max_combined_raw_workers": workers + 4 if partition == "original" else workers,
                    "optimizer_updates": 0,
                    "disjoint_from_active_original_pass": True,
                },
            )
    atomic_json(
        output / f"INPUT_{partition.upper()}_PROGRESS.json",
        {
            "status": "PASS_FINISHED_PRESERVED",
            "completed_extra_rows": complete,
            "workers": workers,
            "optimizer_updates": 0,
            "disjoint_from_active_original_pass": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("original", "all"), default="original")
    parser.add_argument("--workers", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve(), args.partition, args.workers)

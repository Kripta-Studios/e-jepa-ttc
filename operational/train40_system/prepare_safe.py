"""Preserve the raw encoding kernel while excluding transient pending files from disk accounting."""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.prepare import write_shard
from operational.train40_system.prepare_partition import worker_init


def committed_bytes(directory: Path) -> int:
    """Pending files can disappear during rename; reserve them separately from committed bytes."""
    total = 0
    for path in directory.glob("shard_*.npz"):
        if re.fullmatch(r"shard_[0-9]{5}\.npz", path.name):
            total += path.stat().st_size
    return total


def selected_groups(
    rows: list[dict], original: set[str], partition: str
) -> list[tuple[int, list[dict]]]:
    """Keep the original all-old31 versus any-new9 partition exactly disjoint."""
    groups = []
    for start in range(0, len(rows), 32):
        group = rows[start : start + 32]
        old = {str(row["sequence_id"]) for row in group} <= original
        if (
            partition == "all"
            or (partition == "original" and old)
            or (partition == "extra" and not old)
        ):
            groups.append((start, group))
    return groups


def run(output: Path, raw_root: Path, partition: str, workers: int) -> None:
    """Resume valid fragments with unchanged tensor generation and safe accounting metadata."""
    import pandas as pd
    import psutil

    verify_sources(read(output / "SAFE_PREPARATION_FREEZE.json"))
    freeze_path = output / "PREPARE_FREEZE.json"
    freeze = read(freeze_path)
    verify_sources(freeze)
    original = set(read(output / "EXTRA_PREPARE_FREEZE.json")["original_pass_sequences"])
    audit = read(output / "DATA_AUDIT.json")
    if len(original) != 31 or not original < set(audit["sequences"]):
        raise ValueError("Frozen exact disjoint task partition required")
    if digest(output / "TRAIN40_ROWS.parquet") != audit["rows_sha256"]:
        raise ValueError("Full TRAIN40 metadata changed")
    directory = Path(freeze["cache_root"])
    rows = pd.read_parquet(output / "TRAIN40_ROWS.parquet").to_dict(orient="records")
    tasks, completed = [], 0
    for start, group in selected_groups(rows, original, partition):
        path = directory / f"shard_{start // 32:05d}.npz"
        if path.with_suffix(".json").exists():
            receipt = read(path.with_suffix(".json"))
            if digest(path) != receipt["sha256"] or receipt["freeze_sha256"] != digest(freeze_path):
                raise ValueError("Preserve changed existing input fragment")
            completed += len(group)
            continue
        pins = {}
        for sequence in {str(row["sequence_id"]) for row in group}:
            receipt_path = output / "raw_receipts" / f"{sequence}.json"
            if not receipt_path.exists():
                receipt_path = ROOT / (
                    f"artifacts/efficient_context_20261004/data_recovery/files/{sequence}.json"
                )
            receipt = read(receipt_path)
            stat = (raw_root / "data/train" / sequence / "events.h5").stat()
            if receipt["status"] != "VERIFIED" or (stat.st_size, stat.st_mtime_ns) != (
                receipt["bytes"],
                receipt["mtime_ns"],
            ):
                raise ValueError("Verified raw TRAIN source required")
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
    deadline = datetime.fromisoformat(read(output / "AUTHORIZATION.json")["deadline_utc"])
    iterator, pending, size, sampled = iter(tasks), {}, 0, float("-inf")
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        while True:
            if time.monotonic() - sampled >= 30:
                size, sampled = committed_bytes(directory), time.monotonic()
            allowed = (
                psutil.virtual_memory().available >= 4 * 1024**3
                and size + 1_000_000_000 < freeze["new_input_disk_cap_bytes"]
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
                    completed += receipt["rows"]
                except Exception as exc:
                    atomic_json(
                        Path(task["destination"]).with_suffix(".failure.json"),
                        {
                            "error": str(exc),
                            "scientific_negative": False,
                            "optimizer_updates": 0,
                        },
                    )
            atomic_json(
                output / f"INPUT_SAFE_{partition.upper()}_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_rows": completed,
                    "workers": workers,
                    "partition": partition,
                    "optimizer_updates": 0,
                    "kernel_changed": False,
                },
            )
    atomic_json(
        output / f"INPUT_SAFE_{partition.upper()}_PROGRESS.json",
        {
            "status": "PASS_FINISHED_PRESERVED",
            "completed_rows": completed,
            "partition": partition,
            "optimizer_updates": 0,
            "kernel_changed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("original", "extra", "all"), required=True)
    parser.add_argument("--workers", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve(), args.partition, args.workers)

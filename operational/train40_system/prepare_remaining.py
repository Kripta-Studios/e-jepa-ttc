"""Use four additional CPU workers only for rows excluded from the active 31-sequence pass."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from operational.efficient_context.common import ROOT, atomic_json, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.prepare import worker_init, write_shard


def extra_groups(rows: list[dict], original: set[str]) -> list[tuple[int, list[dict]]]:
    """Partitions are disjoint: the first pass accepts only all-original groups."""
    return [
        (start, rows[start : start + 32])
        for start in range(0, len(rows), 32)
        if not {str(row["sequence_id"]) for row in rows[start : start + 32]} <= original
    ]


def run(output: Path, raw_root: Path) -> None:
    """Reuse the QA-frozen bit-exact encoder with disjoint output ownership."""
    import pandas as pd
    import psutil

    freeze_path = output / "PREPARE_FREEZE.json"
    freeze = read(freeze_path)
    verify_sources(freeze)
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
    for start, group in extra_groups(rows, original):
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
    with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as pool:
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
            while len(pending) < 4 and allowed:
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
                output / "INPUT_EXTRA_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_extra_rows": complete,
                    "workers": 4,
                    "max_combined_raw_workers": 8,
                    "optimizer_updates": 0,
                    "disjoint_from_active_original_pass": True,
                },
            )
    atomic_json(
        output / "INPUT_EXTRA_PROGRESS.json",
        {
            "status": "PASS_FINISHED_PRESERVED",
            "completed_extra_rows": complete,
            "workers": 4,
            "optimizer_updates": 0,
            "disjoint_from_active_original_pass": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve())

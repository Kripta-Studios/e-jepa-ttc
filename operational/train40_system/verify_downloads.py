"""Verify pinned TRAIN media downloads incrementally without rereading historical raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from operational.efficient_context.common import atomic_json, digest
from operational.train40_system.data_audit import OUTPUT, safe_media_path


def verify(task: tuple[Path, Path, dict]) -> bool:
    """Hash a completed file once and reject concurrent changes or wrong bytes."""
    path, destination, entry = task
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    before = path.stat()
    if destination.is_file():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if (
            saved["sha256"] == entry["sha256"]
            and saved["bytes"] == before.st_size
            and saved["mtime_ns"] == before.st_mtime_ns
            and saved["status"] == "VERIFIED"
        ):
            return True
        raise ValueError(f"Previously verified data changed: {path}")
    checksum = digest(path)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        return False
    if checksum != entry["sha256"]:
        atomic_json(
            destination.with_suffix(".failure.json"),
            {
                "path": str(path),
                "expected_sha256": entry["sha256"],
                "observed_sha256": checksum,
                "optimizer_updates": 0,
            },
        )
        raise ValueError(f"Downloaded TRAIN data fails SHA-256: {path}")
    atomic_json(
        destination,
        {
            "status": "VERIFIED",
            "path": str(path),
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": checksum,
            "optimizer_updates": 0,
        },
    )
    return True


def run(output: Path, raw_root: Path, *, watch: bool = False) -> None:
    """Advance raw and RGB receipts; partial availability is a dependency, not a result."""
    import pandas as pd
    from huggingface_hub import HfApi, RepoFile

    raw_plan = json.loads((output / "RAW_PLAN.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "DATA_AUDIT.json").read_text(encoding="utf-8"))
    table_path = output / "TRAIN40_ROWS.parquet"
    if digest(table_path) != audit["rows_sha256"]:
        raise ValueError("Pinned TRAIN metadata changed")
    table = pd.read_parquet(table_path, columns=["sequence_id", "rgb_shard_paths"])
    paths = sorted({str(p) for array in table["rgb_shard_paths"] for p in array})
    plan_path = output / "RGB_PLAN.json"
    if plan_path.is_file():
        rgb_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        info = HfApi().get_paths_info(
            raw_plan["repo"], paths, repo_type="dataset", revision=raw_plan["revision"]
        )
        entries = []
        for entry in info:
            if not isinstance(entry, RepoFile):
                raise ValueError("TRAIN media endpoint resolved to a directory")
            if entry.lfs is None:
                raise ValueError("Pinned RGB TAR requires authoritative full SHA-256")
            entries.append(
                {"filename": entry.path, "bytes": entry.size, "sha256": entry.lfs.sha256}
            )
        if set(e["filename"] for e in entries) != set(paths):
            raise ValueError("The Hub does not expose all referenced TRAIN RGB shards")
        rgb_plan = {
            "repo": raw_plan["repo"],
            "revision": raw_plan["revision"],
            "files": sorted(entries, key=lambda x: x["filename"]),
        }
        atomic_json(plan_path, rgb_plan)
    (output / "raw_receipts").mkdir(exist_ok=True)
    (output / "rgb_receipts").mkdir(exist_ok=True)
    tasks = []
    for entry in raw_plan["files"]:
        sequence = entry["sequence_id"]
        historical = output.parent / (
            f"efficient_context_20261004/data_recovery/files/{sequence}.json"
        )
        destination = (
            historical if historical.is_file() else (output / "raw_receipts" / f"{sequence}.json")
        )
        tasks.append((safe_media_path(entry["filename"], sequence, raw_root), destination, entry))
    for entry in rgb_plan["files"]:
        relative = entry["filename"]
        destination = (
            output / "rgb_receipts" / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
        )
        tasks.append(
            (safe_media_path(relative, relative.split("/")[2], raw_root), destination, entry)
        )
    while True:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(verify, tasks))
        complete = sum(outcomes)
        atomic_json(
            output / "MEDIA_VERIFICATION_PROGRESS.json",
            {
                "status": "COMPLETE" if complete == len(tasks) else "PARTIAL_PRESERVED",
                "verified_raw": sum(outcomes[:40]),
                "raw_total": 40,
                "verified_rgb": sum(outcomes[40:]),
                "rgb_total": len(tasks) - 40,
                "optimizer_updates": 0,
                "read_only_payload_verification": True,
            },
        )
        if not watch or complete == len(tasks):
            return
        time.sleep(1800)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve(), watch=args.watch)

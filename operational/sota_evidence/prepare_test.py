"""Resumable CPU preparation of all public test12 queries, without TTC labels."""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.efficient_context.common import digest
from operational.sota_evidence.test_inputs import (
    EXPOSURE_COLUMNS,
    INPUT_COLUMNS,
    describe_input,
    supported_job,
)
from operational.train40_system.durable_io import atomic_json


def freeze_inputs(inputs: Path, media: Path, raw_root: Path, output: Path) -> dict[str, Any]:
    """Freeze a complete label-free population before preparing any test tensor."""
    rows = pd.read_parquet(inputs, columns=list(INPUT_COLUMNS))
    if rows.sample_token.duplicated().any() or len(rows) != 6762:
        raise ValueError("Expected exactly 6762 unique official sample tokens")
    exposure_rows = pd.read_parquet(media, columns=list(EXPOSURE_COLUMNS)).to_dict("records")
    exposures = {(r["sequence_id"], r["rgb_member_path"]): r for r in exposure_rows}
    if len(exposures) != len(exposure_rows):
        raise ValueError("Ambiguous public exposure metadata")
    jobs = [
        describe_input(row, exposures, raw_root, split="test") for row in rows.to_dict("records")
    ]
    jobs.sort(key=lambda r: (r["sequence"], r["anchor"], r["sample_token"]))
    value = {
        "schema": "garlttc_native_h8_test_inputs_v1",
        "rows": jobs,
        "input_sha256": digest(inputs),
        "media_metadata_sha256": digest(media),
        "sources": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("test_inputs.py"),
                Path("src/e_jepa_ttc/efficient_context/mapped_union.py"),
                Path("src/e_jepa_ttc/simplex_t/context_raw_union.py"),
                Path("src/e_jepa_ttc/simplex_t/cached_event_reader.py"),
                Path("src/e_jepa_ttc/data/event_v4_geometry.py"),
                Path("src/e_jepa_ttc/data/eap_representation.py"),
            )
        },
        "candidate": "TRAIN40_H8_median_of_seeds_7_13_23",
        "test_labels_read": False,
        "optimizer_updates": 0,
        "history_lags_us": list(range(350000, -1, -50000)),
        "precision": "float32_no_tf32",
        "producer_batch": 16,
        "roi_policy": "supplied_current_common_ROI_at_public_exposure_end",
        "selection": "fixed retained H8 reference; no test-driven selection",
    }
    path = output / "INPUT_FREEZE.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Frozen inputs or preprocessing changed")
    else:
        atomic_json(path, value)
    return value


def prepare(
    inputs: Path,
    media: Path,
    raw_root: Path,
    output: Path,
    cache: Path,
    plan: Path,
    *,
    wait_seconds: int = 0,
) -> None:
    """Process available verified sequences with bounded RAM and persistent receipts."""
    from e_jepa_ttc.efficient_context.mapped_union import encode_union

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    frozen = freeze_inputs(inputs, media, raw_root, output)
    binding = digest(output / "INPUT_FREEZE.json")
    downloads = json.loads(plan.read_text(encoding="utf-8-sig"))
    pins = {item["path"]: item for item in downloads["files"]}
    deadline = time.monotonic() + wait_seconds
    completed: set[int] = set()
    rows = frozen["rows"]
    pool = ReaderPool()
    started = time.monotonic()
    try:
        while len(completed) < len(rows):
            for number, original in enumerate(rows):
                if number in completed:
                    continue
                destination = cache / f"query_{number:05d}.npz"
                receipt_path = destination.with_suffix(".json")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if (
                        receipt["input_freeze_sha256"] != binding
                        or receipt["sample_token"] != original["sample_token"]
                        or digest(destination) != receipt["sha256"]
                    ):
                        raise ValueError("Prepared input changed or belongs to another query")
                    completed.add(number)
                    continue
                raw = Path(original["path"])
                if not raw.exists():
                    continue
                key = f"data/test/{original['sequence']}/events.h5"
                pin = pins[key]
                expected = pin["lfs"]["oid"]
                if raw.stat().st_size != pin["size"]:
                    raise ValueError("Raw input size differs from pinned HF release")
                raw_receipt = output / "raw" / f"{original['sequence']}.json"
                identity = {
                    "bytes": raw.stat().st_size,
                    "mtime_ns": raw.stat().st_mtime_ns,
                    "sha256": expected,
                    "path": str(raw),
                }
                if raw_receipt.exists():
                    if json.loads(raw_receipt.read_text(encoding="utf-8")) != identity:
                        raise ValueError("Verified raw input changed")
                else:
                    atomic_json(
                        output / "PREPARATION_PROGRESS.json",
                        {
                            "status": "VERIFYING_RAW_SHA256",
                            "sequence": original["sequence"],
                            "completed": len(completed),
                            "total": len(rows),
                            "utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    if digest(raw) != expected:
                        raise ValueError("Raw input SHA256 differs from pinned HF release")
                    atomic_json(raw_receipt, identity)
                if shutil.disk_usage(cache).free < 30 * 1024**3:
                    raise RuntimeError("Preparation paused: less than 30 GiB free cache space")
                begin = time.monotonic()
                reader = pool.get(raw)
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
                atomic_json(
                    receipt_path,
                    {
                        "input_freeze_sha256": binding,
                        "sample_token": job["sample_token"],
                        "sha256": digest(destination),
                        "raw_sha256": expected,
                        "seconds": time.monotonic() - begin,
                        "optimizer_updates": 0,
                    },
                )
                completed.add(number)
                atomic_json(
                    output / "PREPARATION_PROGRESS.json",
                    {
                        "status": "PREPARING",
                        "completed": len(completed),
                        "total": len(rows),
                        "sequence": original["sequence"],
                        "elapsed_s": time.monotonic() - started,
                        "utc": datetime.now(UTC).isoformat(),
                    },
                )
            if len(completed) == len(rows) or time.monotonic() >= deadline:
                break
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
    finally:
        pool.close()
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
        },
    )


def main() -> None:
    """Prepare available sequences; optionally wait a bounded time for downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "media", "raw-root", "output", "cache", "plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    prepare(
        args.inputs,
        args.media,
        args.raw_root,
        args.output,
        args.cache,
        args.plan,
        wait_seconds=args.wait_seconds,
    )


if __name__ == "__main__":
    main()

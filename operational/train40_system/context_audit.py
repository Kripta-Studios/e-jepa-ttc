"""Audit full TRAIN40 causal H8 support and supplied-ROI exposure timing before fitting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pyarrow.dataset as ds
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def run(output: Path, raw_root: Path) -> None:
    """Read only sensor bounds and public TRAIN frame metadata; keep all TTC labels closed."""
    from e_jepa_ttc.data.eap import _require_hdf5plugin

    audit = read(output / "DATA_AUDIT.json")
    if digest(output / "TRAIN40_INDEX.npz") != audit["index_sha256"]:
        raise ValueError("Canonical TRAIN metadata changed")
    media_path = raw_root / "data/train.parquet"
    revision = read(output / "RAW_PLAN.json")["revision"]
    files = HfApi().get_paths_info(
        "NAIL-HNU/eAP-dataset", ["data/train.parquet"], repo_type="dataset", revision=revision
    )
    if len(files) != 1 or not isinstance(files[0], RepoFile) or files[0].lfs is None:
        raise ValueError("Exact HF training frame metadata pin required")
    media_sha = digest(media_path)
    if media_sha != files[0].lfs.sha256:
        raise ValueError("Downloaded TRAIN frame metadata checksum differs")
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        index = {key: stored[key] for key in ("tokens", "sequences", "windows_us", "rgb_members")}
    frames = (
        ds.dataset(media_path)
        .to_table(
            columns=[
                "sequence_id",
                "rgb_member_path",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
            filter=ds.field("sequence_id").isin(audit["sequences"]),
        )
        .to_pylist()
    )
    lookup = {(frame["sequence_id"], frame["rgb_member_path"]): frame for frame in frames}
    if len(lookup) != len(frames):
        raise ValueError("Ambiguous TRAIN frame identity")
    _require_hdf5plugin()
    bounds = {}
    for sequence in audit["sequences"]:
        raw = raw_root / "data/train" / sequence / "events.h5"
        receipt_path = output / "raw_receipts" / (sequence + ".json")
        if not receipt_path.exists():
            receipt_path = (
                ROOT / f"artifacts/efficient_context_20261004/data_recovery/files/{sequence}.json"
            )
        receipt = read(receipt_path)
        stat = raw.stat()
        if receipt["status"] != "VERIFIED" or (stat.st_size, stat.st_mtime_ns) != (
            receipt["bytes"],
            receipt["mtime_ns"],
        ):
            raise ValueError("Verified raw TRAIN identity changed")
        with h5py.File(raw, "r") as handle:
            timestamps = handle["events/t"]
            if not isinstance(timestamps, h5py.Dataset):
                raise ValueError("Raw event timestamps must be a dataset")
            bounds[sequence] = (int(timestamps[0]), int(timestamps[-1]))
    available = np.empty(88744, np.int64)
    valid = np.empty((88744, 8), bool)
    missing_current = []
    for row in range(88744):
        sequence, windows = str(index["sequences"][row]), index["windows_us"][row]
        endpoints = [lookup[(sequence, str(member))] for member in index["rgb_members"][row]]
        starts = [int(frame["rgb_exposure_start_timestamp_us"]) for frame in endpoints]
        ends = [int(frame["rgb_exposure_end_timestamp_us"]) for frame in endpoints]
        if starts != windows[1:, 1].tolist() or any(
            end < start for start, end in zip(starts, ends, strict=True)
        ):
            raise ValueError("Event/RGB exposure clock mapping differs")
        available[row] = max(ends)
        lower, upper = bounds[sequence]
        valid[row] = (windows.min() - np.arange(7, -1, -1) * 50000 >= lower) & (
            windows.max() <= upper
        )
        if not valid[row, -1]:
            missing_current.append(
                {
                    "ordinal": row,
                    "sequence_id": sequence,
                    "sample_token": str(index["tokens"][row]),
                    "windows_us": windows.tolist(),
                    "raw_bounds_us": [lower, upper],
                }
            )
    destination = output / "TRAIN40_CONTEXT.npz"
    temporary = destination.with_suffix(".pending.npz")
    np.savez_compressed(
        temporary,
        valid=valid,
        available_us=available,
        anchor_us=index["windows_us"][:, -1, 1],
        tokens=index["tokens"],
    )
    os.replace(temporary, destination)
    atomic_json(
        output / "CONTEXT_AUDIT.json",
        {
            "status": "PASSED" if not missing_current else "DEPENDENCY_UNRESOLVED",
            "population": 88744,
            "sequences": 40,
            "source_sha256": digest(Path(__file__)),
            "index_sha256": audit["index_sha256"],
            "frame_metadata_sha256": media_sha,
            "context_sha256": digest(destination),
            "source_bounds_us": bounds,
            "missing_current_support": missing_current,
            "valid_H8_slot_count_histogram": {
                str(n): int((valid.sum(1) == n).sum()) for n in range(9)
            },
            "exposure_age_us_min": int((available - index["windows_us"][:, -1, 1]).min()),
            "exposure_age_us_max": int((available - index["windows_us"][:, -1, 1]).max()),
            "targets_read": False,
            "optimizer_updates": 0,
            "does_not_block_independent_encoder_fits": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve())

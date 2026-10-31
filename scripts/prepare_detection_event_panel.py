"""Build a resumable, label-free causal event cache for fixed development panels."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from e_jepa_ttc.detection.eap_rgb_data import safe_dataset_path, sha256_file
from e_jepa_ttc.detection.event_data import CausalEventConfig, CausalEventVoxelizer


def write_json(path: Path, value: object) -> None:
    """Atomically publish a small receipt."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def verify_cached(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    """Accept an existing frame only when its identity and stored bytes still match."""
    receipt_path = path.with_suffix(".json")
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["identity"] != expected or sha256_file(path) != receipt["file_sha256"]:
        raise ValueError(f"Existing event cache differs: {path}")
    return receipt


def build_sequence(task: dict[str, Any]) -> dict[str, Any]:
    """Hash one source and encode its ordered windows with one bounded HDF5 reader."""
    root, output = Path(str(task["root"])), Path(str(task["output"]))
    rows = task["rows"]
    assert isinstance(rows, list) and rows
    sequence = str(rows[0]["sequence_id"])
    directory = output / sequence
    directory.mkdir(exist_ok=True)
    relative = str(rows[0]["events_path"])
    if any(row["events_path"] != relative for row in rows):
        raise ValueError("One sequence has inconsistent event paths")
    source = safe_dataset_path(root, relative)
    expected = task["public_member"]
    assert isinstance(expected, dict)
    stat = source.stat()
    if stat.st_size != expected["size_bytes"]:
        raise ValueError(f"HDF5 size differs: {sequence}")
    source_identity = {
        "path": relative,
        "sha256": expected["sha256"],
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    source_receipt = directory / "SOURCE.json"
    if source_receipt.exists():
        if json.loads(source_receipt.read_text()) != source_identity:
            raise ValueError(f"HDF5 source changed since admission: {sequence}")
    else:
        if sha256_file(source) != expected["sha256"]:
            raise ValueError(f"HDF5 public hash differs: {sequence}")
        write_json(source_receipt, source_identity)
    config = CausalEventConfig(width=160, height=96, max_open_handles=1)
    records = []
    started = time.monotonic()
    with CausalEventVoxelizer(root, config) as voxelizer:
        for row in rows:
            token = str(row["sample_token"])
            path = directory / (hashlib.sha256(token.encode()).hexdigest() + ".npz")
            identity = {**row, "contract_sha256": task["contract_sha256"]}
            receipt = verify_cached(path, identity)
            if receipt is None:
                began = time.monotonic()
                result = voxelizer.encode(
                    relative,
                    sequence_id=sequence,
                    rgb_exposure_end_us=int(row["rgb_exposure_end_timestamp_us"]),
                )
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, voxel=result.voxel)
                os.replace(temporary, path)
                # Compression is lossless; verify the exact tensor before publication.
                with np.load(path, allow_pickle=False) as restored:
                    if not np.array_equal(restored["voxel"], result.voxel):
                        raise ValueError("Compressed voxel roundtrip changed values")
                receipt = {
                    "identity": identity,
                    "file_sha256": sha256_file(path),
                    "source_sha256": expected["sha256"],
                    "representation": result.receipt,
                    "seconds": time.monotonic() - began,
                }
                write_json(path.with_suffix(".json"), receipt)
            records.append(
                {
                    "sample_token": token,
                    "role": row["role"],
                    "path": path.relative_to(output).as_posix(),
                    "sha256": receipt["file_sha256"],
                }
            )
            if len(records) % 16 == 0 or len(records) == len(rows):
                write_json(
                    directory / "PROGRESS.json",
                    {
                        "completed": len(records),
                        "total": len(rows),
                        "seconds": time.monotonic() - started,
                        "observed_at_unix": time.time(),
                    },
                )
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("Event source changed during cache preparation")
    write_json(directory / "INDEX.json", records)
    return {"sequence": sequence, "frames": len(records), "records": records}


def main() -> None:
    """Prepare only TRAIN/selection sensor inputs; never read object annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    public = json.loads(args.public_manifest.read_text())["files"]
    metadata = args.root / "data/train.parquet"
    if sha256_file(metadata) != public["data/train.parquet"]["sha256"]:
        raise ValueError("Public metadata hash differs")
    columns = [
        "sample_token",
        "sequence_id",
        "events_path",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
    ]
    frame = pd.read_parquet(metadata, columns=columns)
    if frame.sample_token.duplicated().any():
        raise ValueError("Metadata sample tokens are not unique")
    populations = {
        role: manifest["roles"][role]["sample_tokens"] for role in ("train", "selection")
    }
    if set(populations["train"]) & set(populations["selection"]):
        raise ValueError("Event panel roles overlap")
    ownership = {token: role for role, tokens in populations.items() for token in tokens}
    frame = frame[frame.sample_token.isin(ownership)].copy()
    if set(frame.sample_token) != set(ownership):
        raise ValueError("Requested event tokens absent from public metadata")
    frame["role"] = frame.sample_token.map(ownership)
    if frame.groupby("sequence_id").role.nunique().max() != 1:
        raise ValueError("A sequence crosses roles")
    contract = {
        "schema": "eap_causal_event_panel_v1",
        "scope": "development inputs, no labels",
        "manifest_sha256": sha256_file(args.manifest),
        "public_manifest_sha256": sha256_file(args.public_manifest),
        "metadata_sha256": sha256_file(metadata),
        "config": asdict(CausalEventConfig(width=160, height=96, max_open_handles=1)),
        "populations": populations,
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in [
                Path(__file__),
                ROOT / "src/e_jepa_ttc/detection/event_data.py",
                ROOT / "src/e_jepa_ttc/data/eap.py",
                ROOT / "src/e_jepa_ttc/representations/voxel_grid.py",
            ]
        },
    }
    contract_path = args.output / "CONTRACT.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing event cache contract differs")
    write_json(contract_path, contract)
    tasks = [
        {
            "root": str(args.root.resolve()),
            "output": str(args.output.resolve()),
            "rows": cast(pd.DataFrame, part)
            .sort_values("rgb_exposure_end_timestamp_us")
            .to_dict("records"),
            "public_member": public[str(part.iloc[0].events_path)],
            "contract_sha256": sha256_file(contract_path),
        }
        for _, part in frame.groupby("sequence_id", sort=True)
    ]
    write_json(args.output / "STATUS.json", {"status": "RUNNING", "frames": len(frame)})
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build_sequence, tasks):
            records.extend(result.pop("records"))
            print(json.dumps(result), flush=True)
    write_json(args.output / "INDEX.json", records)
    write_json(
        args.output / "STATUS.json",
        {
            "status": "COMPLETE",
            "frames": len(records),
            "contract_sha256": sha256_file(contract_path),
            "index_sha256": sha256_file(args.output / "INDEX.json"),
        },
    )


if __name__ == "__main__":
    main()

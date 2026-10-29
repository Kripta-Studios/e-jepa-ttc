"""Persist a hash-bound sensor-time RGB history audit for frozen eAP roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from e_jepa_ttc.detection import temporal_data as temporal_module
from e_jepa_ttc.detection.temporal_data import TemporalRGBIndex, audit_temporal_support


def sha256_file(path: Path) -> str:
    """Hash one source or metadata artifact in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_split(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "eap_detection_sequence_split_v1":
        raise ValueError("Unexpected detection split schema")
    if value.get("source_split") != "train":
        raise ValueError("Temporal development roles must derive from public train")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("dataset_revision", ""))):
        raise ValueError("Temporal split needs a pinned dataset revision")
    assignments = value.get("assignments")
    if not isinstance(assignments, dict) or set(assignments) != {
        "train",
        "selection",
        "calibration",
    }:
        raise ValueError("Split must preserve train/selection/calibration roles")
    owners: set[str] = set()
    for role, sequences in assignments.items():
        if not isinstance(sequences, list) or len(sequences) != len(set(sequences)):
            raise ValueError(f"Invalid sequence list for {role}")
        if owners & set(sequences):
            raise ValueError("Temporal split roles overlap")
        owners.update(sequences)
    return value


def build_audit(*, data_root: Path, split_path: Path) -> dict[str, Any]:
    """Build the temporal audit without opening images, labels, events or test."""
    split = _load_split(split_path)
    media_path = data_root / "data/train.parquet"
    media_hash = sha256_file(media_path)
    if media_hash != split.get("media_sha256"):
        raise ValueError("Canonical train metadata differs from the frozen split")
    assignments = {role: list(split["assignments"][role]) for role in ("train", "selection")}
    sequences = assignments["train"] + assignments["selection"]
    rows = pq.read_table(
        media_path,
        columns=[
            "sample_token",
            "split",
            "sequence_id",
            "rgb_exposure_start_timestamp_us",
            "rgb_exposure_end_timestamp_us",
        ],
        filters=[("sequence_id", "in", sequences)],
    ).to_pylist()
    if any(row["split"] != "train" or row["sequence_id"] not in sequences for row in rows):
        raise ValueError("Temporal audit rows leave their frozen public-train roles")
    index = TemporalRGBIndex(rows)
    support = audit_temporal_support(index, assignments)
    script_path = Path(__file__).resolve()
    module_path = Path(temporal_module.__file__).resolve()
    return {
        "schema": "eap_detection_temporal_audit_receipt_v1",
        "dataset_revision": split["dataset_revision"],
        "source_split": "train",
        "media": {
            "path": "data/train.parquet",
            "sha256": media_hash,
            "columns_read": [
                "sample_token",
                "split",
                "sequence_id",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
        },
        "split": {
            "path": split_path.as_posix(),
            "sha256": sha256_file(split_path),
            "train_sequence_ids": assignments["train"],
            "selection_sequence_ids": assignments["selection"],
            "calibration_sequence_ids_excluded": list(split["assignments"]["calibration"]),
        },
        "implementation": {
            "module": str(module_path),
            "module_sha256": sha256_file(module_path),
            "entrypoint": str(script_path),
            "entrypoint_sha256": sha256_file(script_path),
        },
        "support": support,
        "scope": {
            "causality": "sensor exposure timestamps only",
            "arrival_timestamp": "unavailable",
            "rgb_opened": False,
            "event_data_opened": False,
            "labels_opened": False,
            "test_opened": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report = build_audit(data_root=args.data_root, split_path=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "roles": report["support"]["roles"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

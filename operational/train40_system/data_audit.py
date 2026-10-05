"""Validate every TRAIN40 annotation and bind the unchanged producer input recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from operational.efficient_context.common import ROOT, atomic_json, digest

SPLIT_SHA = "ab2a68609d535379b1d6ed3617b3487cf61dce38dcb13c53246d5ac74c0b3251"
OUTPUT = ROOT / "artifacts/train40_system_20261005"
SPLIT = ROOT / (
    "artifacts/efficient_context_20261004/garl/public_checkpoint_audit/dataset/splits/train.txt"
)


def published_sequences(path: Path) -> list[str]:
    """Accept precisely the previously verified public TRAIN40 split."""
    if digest(path) != SPLIT_SHA:
        raise ValueError("Published TRAIN40 split SHA-256 changed")
    sequences = path.read_text(encoding="utf-8").split()
    if len(sequences) != 40 or len(set(sequences)) != 40:
        raise ValueError("Published split must contain forty unique sequences")
    return sequences


def safe_media_path(relative: str, sequence: str, root: Path) -> Path:
    """Confine media access to this sequence's explicitly authorized TRAIN directory."""
    parts = relative.replace("\\", "/").split("/")
    if len(parts) < 4 or parts[:3] != ["data", "train", sequence]:
        raise ValueError(f"Media is outside the selected TRAIN sequence: {relative}")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Unsafe media path")
    resolved = (root / relative).resolve()
    parent = (root / "data/train" / sequence).resolve()
    if parent not in resolved.parents:
        raise ValueError("Media path escapes TRAIN root")
    return resolved


def describe_row(row: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    """Resolve all training geometry and signed targets without reading event payloads."""
    from e_jepa_ttc.data.event_v4_geometry import (
        box_in_common_roi,
        common_square_from_boxes,
        shifted_precontext_window,
    )
    from e_jepa_ttc.data.garlttc_eap import normalize_boxes_xyxy, normalize_event_windows_us
    from e_jepa_ttc.data.garlttc_lhr_cache import (
        _box_features,
        _official_ttc_at_endpoint,
        _visible_heights,
        select_temporal_indices,
    )

    sequence = str(row["sequence_id"])
    raw = safe_media_path(str(row["events_path"]), sequence, raw_root)
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    stamps = [int(x) for x in row["frame_timestamps_us"]]
    rgb_shards = [str(x) for x in row["rgb_shard_paths"]]
    rgb_members = [str(x) for x in row["rgb_member_paths"]]
    if len({len(boxes), len(windows), len(stamps), len(rgb_shards), len(rgb_members)}) != 1:
        raise ValueError("TRAIN metadata lengths must be aligned")
    first, second, context = select_temporal_indices(
        stamps,
        anchor_timestamp_us=int(row["timestamp_us"]),
        target_delta_t_s=0.1,
        tolerance_s=0.025,
        context_delta_t_s=0.1,
        context_tolerance_s=0.05,
    )
    t0_window = (
        windows[context]
        if context is not None
        else shifted_precontext_window(windows[first], shift_s=0.1)
    )
    t0_box = boxes[context] if context is not None else boxes[first]
    square = common_square_from_boxes(boxes, (first, second), margin_fraction=0.25)
    target = _official_ttc_at_endpoint(row, second)
    if not np.isfinite(target) or not (target < 0 or target > 0.1):
        raise ValueError("Official signed TTC lies outside the fixed phase domain")
    heights = _visible_heights(row, (first, second), boxes, 128)
    if not np.isfinite(heights).all() or not (heights > 0).all():
        raise ValueError("Visible-height training supervision is invalid")
    delta = (stamps[second] - stamps[first]) * 1e-6
    motion, _ = _box_features(boxes[first], boxes[second], delta)
    for endpoint in (first, second):
        safe_media_path(rgb_shards[endpoint], sequence, raw_root)
        if not rgb_members[endpoint].startswith("rgb/") or ".." in rgb_members[endpoint]:
            raise ValueError("RGB tar member escapes its declared namespace")
    return {
        "raw": str(raw),
        "sequence": sequence,
        "token": str(row["sample_token"]),
        "windows": (t0_window, windows[first], windows[second]),
        "square": square,
        "boxes": np.stack(
            [
                box_in_common_roi(b, square, roi_size=128)
                for b in (t0_box, boxes[first], boxes[second])
            ]
        ),
        "target": target,
        "phase": -np.log1p(-0.1 / target),
        "heights": heights,
        "motion": motion,
        "delta": delta,
        "indices": (first, second),
        "proxy": context is None,
        "rgb_shards": (rgb_shards[first], rgb_shards[second]),
        "rgb_members": (rgb_members[first], rgb_members[second]),
    }


def run(garl_root: Path, raw_root: Path, output: Path) -> None:
    """Audit all forty sequences, retaining every row and the exact canonical join."""
    from e_jepa_ttc.data.garlttc_eap import load_garlttc_train_index

    output.mkdir(parents=True, exist_ok=True)
    sequences = published_sequences(SPLIT)
    authorization = json.loads((output / "AUTHORIZATION.json").read_text(encoding="utf-8"))
    if authorization["public_TRAIN40_split_sha256"] != SPLIT_SHA:
        raise ValueError("Authorization differs from public TRAIN40")
    index = load_garlttc_train_index(garl_root, sequences)
    if index.source_merged_row_count != 88744 or index.selected_row_count != 88744:
        raise ValueError("All 88,744 public TRAIN rows must be retained")
    if set(index.sequence_ids) != set(sequences):
        raise ValueError("TRAIN rows and public forty-sequence list differ")
    records = []
    for ordinal, row in enumerate(index.merged.to_dict(orient="records")):
        try:
            records.append(describe_row(row, raw_root))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            atomic_json(
                output / "DATA_AUDIT_FAILURE.json",
                {
                    "ordinal": ordinal,
                    "sequence_id": str(row["sequence_id"]),
                    "sample_token": str(row["sample_token"]),
                    "error": str(exc),
                    "optimizer_updates": 0,
                    "scientific_negative": False,
                },
            )
            raise
        if (ordinal + 1) % 4096 == 0:
            atomic_json(
                output / "DATA_AUDIT_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "audited_rows": ordinal + 1,
                    "total_rows": 88744,
                    "optimizer_updates": 0,
                },
            )
    arrays = {
        "tokens": np.asarray([x["token"] for x in records]),
        "sequences": np.asarray([x["sequence"] for x in records]),
        "windows_us": np.asarray([x["windows"] for x in records], dtype=np.int64),
        "square_xyxy": np.asarray([x["square"] for x in records], dtype=np.float64),
        "boxes_xyxy": np.asarray([x["boxes"] for x in records], dtype=np.float32),
        "visible_heights": np.asarray([x["heights"] for x in records], dtype=np.float32),
        "observable_motion": np.asarray([x["motion"] for x in records], dtype=np.float32),
        "delta_t_s": np.asarray([x["delta"] for x in records], dtype=np.float32),
        "ttc_s": np.asarray([x["target"] for x in records], dtype=np.float64),
        "phase": np.asarray([x["phase"] for x in records], dtype=np.float64),
        "endpoint_indices": np.asarray([x["indices"] for x in records], dtype=np.int64),
        "t0_box_proxy": np.asarray([x["proxy"] for x in records], dtype=np.bool_),
        "rgb_shards": np.asarray([x["rgb_shards"] for x in records]),
        "rgb_members": np.asarray([x["rgb_members"] for x in records]),
    }
    temp = output / "TRAIN40_INDEX.pending.npz"
    np.savez_compressed(temp, **arrays)
    os.replace(temp, output / "TRAIN40_INDEX.npz")
    data_path = output / "TRAIN40_ROWS.parquet"
    temp_rows = data_path.with_suffix(".pending.parquet")
    index.merged.to_parquet(temp_rows, index=False)
    os.replace(temp_rows, data_path)
    target = arrays["ttc_s"]
    atomic_json(
        output / "DATA_AUDIT.json",
        {
            "status": "PASSED",
            "rows": len(records),
            "sequences": sorted(sequences),
            "sequences_count": len(sequences),
            "dropped_rows": 0,
            "data_sha256": index.data_sha256,
            "annotations_sha256": index.annotations_sha256,
            "join_keys_sha256": index.join_keys_sha256,
            "split_sha256": SPLIT_SHA,
            "index_sha256": digest(output / "TRAIN40_INDEX.npz"),
            "rows_sha256": digest(data_path),
            "source_sha256": digest(Path(__file__)),
            "negative_targets": int((target < 0).sum()),
            "positive_targets": int((target > 0.1).sum()),
            "raw_files_currently_present": sum(
                (raw_root / "data/train" / s / "events.h5").is_file() for s in sequences
            ),
            "row_counts": dict(sorted(Counter(x["sequence"] for x in records).items())),
            "training_only": True,
            "optimizer_updates": 0,
            "historical_validation_sequence_ids_now_train_not_evaluation": [
                "DGqicHUGWb",
                "pBqGOb2vYq",
                "qoohcdtLDH",
            ],
            "sealed_evaluation_accessed": False,
            "canonical_metadata_fingerprint": hashlib.sha256(
                arrays["windows_us"].tobytes() + arrays["square_xyxy"].tobytes()
            ).hexdigest(),
        },
    )
    atomic_json(
        output / "DATA_AUDIT_PROGRESS.json",
        {
            "status": "COMPLETE",
            "audited_rows": len(records),
            "total_rows": 88744,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garl-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.garl_root.resolve(), args.raw_root.resolve(), args.output.resolve())

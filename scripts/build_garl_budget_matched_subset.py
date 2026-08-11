#!/usr/bin/env python
"""Build a Garl public 8192/2048 subset from the exact E-JEPA S1 cache tokens."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheDataset
from scripts.build_garl_matched_screen_subset import _select_exact_public_rows, _sha256, _write_role

JOIN_KEYS = ("sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us")


def _rows(manifest: Path, split: str) -> pd.DataFrame:
    ds = GarlTTCLHRCacheDataset(manifest, splits=(split,))
    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        r = ds[i]
        rows.append({
            "sequence_id": str(r["sequence_id"]),
            "sample_token": str(r["sample_token"]),
            "track_id": str(r["track_id"]),
            "public_track_id": str(r["public_track_id"]),
            "timestamp_us": int(r["timestamp_us"]),
            "cache_ttc_s": float(r["ttc_s"]),
        })
    out = pd.DataFrame(rows)
    if out["sample_token"].duplicated().any():
        raise ValueError(f"duplicate {split} sample_token")
    return out


def build(train_manifest: Path, validation_manifest: Path, public_data: Path, public_labels: Path, output_dir: Path, expected_train: int, expected_validation: int) -> dict[str, Any]:
    train = _rows(train_manifest, "train")
    validation = _rows(validation_manifest, "validation")
    if len(train) != expected_train or len(validation) != expected_validation:
        raise ValueError(f"cache counts mismatch: train={len(train)}, validation={len(validation)}")
    if set(train["sequence_id"]) & set(validation["sequence_id"]):
        raise ValueError("train and validation sequences overlap")
    data = pd.read_parquet(public_data)
    labels = pd.read_parquet(public_labels)
    selected = {
        "train": _select_exact_public_rows(train, data, labels, split="train"),
        "validation": _select_exact_public_rows(validation, data, labels, split="validation"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = {split: _write_role(output_dir, split, d, l) for split, (d, l) in selected.items()}
    result: dict[str, Any] = {
        # Keep the existing artifact type so the already-audited official cache builder
        # can consume this larger budget-matched subset without a second code path.
        "artifact_type": "garl_event_only_matched_screen_subset_v1",
        "protocol_variant": "budget_matched_s1_8192_train_2048_validation_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_train_validation_only",
        "roles": roles,
        "checks": {
            "exact_ejepa_cache_tokens": True,
            "exact_join_keys": True,
            "target_equality": True,
            "train_validation_sequence_disjoint": True,
            "bbox_used_by_official_preprocessing_only": True,
            "bbox_is_not_direct_model_input": True,
            "private_test_opened": False,
        },
        "sources": {
            "train_cache_manifest": {"path": str(train_manifest.resolve()), "sha256": _sha256(train_manifest)},
            "validation_cache_manifest": {"path": str(validation_manifest.resolve()), "sha256": _sha256(validation_manifest)},
            "public_data": {"path": str(public_data.resolve()), "sha256": _sha256(public_data)},
            "public_labels": {"path": str(public_labels.resolve()), "sha256": _sha256(public_labels)},
        },
    }
    sign_artifact(result)
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache-manifest", type=Path, required=True)
    p.add_argument("--validation-cache-manifest", type=Path, required=True)
    p.add_argument("--public-data-parquet", type=Path, required=True)
    p.add_argument("--public-labels-parquet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-train-rows", type=int, default=8192)
    p.add_argument("--expected-validation-rows", type=int, default=2048)
    args = p.parse_args()
    try:
        result = build(args.train_cache_manifest.resolve(), args.validation_cache_manifest.resolve(), args.public_data_parquet.resolve(), args.public_labels_parquet.resolve(), args.output_dir.resolve(), args.expected_train_rows, args.expected_validation_rows)
    except Exception as exc:
        print(f"budget-matched subset failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

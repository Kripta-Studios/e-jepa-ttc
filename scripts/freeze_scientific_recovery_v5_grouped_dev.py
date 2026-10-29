#!/usr/bin/env python
"""Freeze the target-blind train-only grouped-development protocol for V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402
    GarlTTCLHRCacheDataset,
)
from e_jepa_ttc.data.scientific_recovery_v5 import (  # noqa: E402
    IDENTITY_COLUMNS,
    build_train_only_grouped_dev,
    validate_cache_identities,
)

FORBIDDEN_SELECTION_COLUMNS = (
    "ttc",
    "target",
    "label",
    "prediction",
    "metric",
    "score",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _cache_identities(manifest_path: Path) -> pd.DataFrame:
    dataset = GarlTTCLHRCacheDataset(manifest_path, splits=("train",))
    rows = []
    for index in range(len(dataset)):
        record = dataset[index]
        rows.append({column: str(record[column]) for column in IDENTITY_COLUMNS})
    return pd.DataFrame(rows)


def freeze_protocol(
    *,
    train_metadata_path: Path,
    cache_manifest_path: Path,
    public_split_path: Path,
    metric_implementation_path: Path,
    output_path: Path,
    folds: int,
    seed: int,
    expected_rows: int,
    expected_sequences: int,
) -> dict[str, Any]:
    metadata = pd.read_parquet(train_metadata_path)
    if len(metadata) != expected_rows:
        raise ValueError(f"Expected {expected_rows} train rows, found {len(metadata)}")
    suspicious = sorted(
        column
        for column in metadata.columns
        if any(fragment in str(column).lower() for fragment in FORBIDDEN_SELECTION_COLUMNS)
    )
    if suspicious:
        raise ValueError(f"Train fold metadata contains forbidden selection columns: {suspicious}")

    public_split = _read_json(public_split_path)
    assignments = public_split.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Public split artifact lacks assignments")
    expected_train = {str(value) for value in assignments.get("train", [])}
    public_validation = {str(value) for value in assignments.get("validation", [])}
    observed_train = set(metadata["sequence_id"].astype(str))
    if observed_train != expected_train:
        raise ValueError("Train metadata sequence universe differs from frozen public train split")
    if observed_train & public_validation:
        raise ValueError("Public validation sequence leaked into train-only grouped development")

    cache_manifest = _read_json(cache_manifest_path)
    cache_identity_check = validate_cache_identities(
        metadata,
        _cache_identities(cache_manifest_path),
    )
    protocol = build_train_only_grouped_dev(
        metadata,
        folds=folds,
        seed=seed,
        expected_sequence_count=expected_sequences,
    )
    tracked_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v5_train_only_grouped_dev_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_a8_results",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tracked_dirty": tracked_dirty,
        **protocol,
        "cache_identity_check": cache_identity_check,
        "sources": {
            "train_metadata": {
                "path": str(train_metadata_path.resolve()),
                "sha256": _sha256(train_metadata_path),
                "columns_used_for_split": list(IDENTITY_COLUMNS),
            },
            "cache_manifest": {
                "path": str(cache_manifest_path.resolve()),
                "sha256": _sha256(cache_manifest_path),
                "artifact_sha256": cache_manifest.get("artifact_sha256"),
            },
            "public_split": {
                "path": str(public_split_path.resolve()),
                "sha256": _sha256(public_split_path),
            },
            "metric_implementation": {
                "path": str(metric_implementation_path.resolve()),
                "sha256": _sha256(metric_implementation_path),
            },
        },
        "source_universe_contract": {
            "fixed_before_v5": True,
            "historical_8192_selection_used_official_ttc_bucket": True,
            "historical_selection_limitation_acknowledged": True,
            "fold_assignment_reads_target_or_performance": False,
            "fold_assignment_columns": list(IDENTITY_COLUMNS),
        },
        "checks": {
            "train_only_grouped_dev": True,
            "sequence_disjoint_folds": True,
            "every_train_sequence_dev_exactly_once": True,
            "sample_token_unique": True,
            "same_cache_universe": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        },
    }
    sign_artifact(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--public-split", type=Path, required=True)
    parser.add_argument(
        "--metric-implementation",
        type=Path,
        default=ROOT / "src/e_jepa_ttc/evaluation/garl_ttc_protocol.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--expected-rows", type=int, default=8192)
    parser.add_argument("--expected-sequences", type=int, default=9)
    args = parser.parse_args()
    try:
        result = freeze_protocol(
            train_metadata_path=args.train_metadata.resolve(),
            cache_manifest_path=args.cache_manifest.resolve(),
            public_split_path=args.public_split.resolve(),
            metric_implementation_path=args.metric_implementation.resolve(),
            output_path=args.output.resolve(),
            folds=args.folds,
            seed=args.seed,
            expected_rows=args.expected_rows,
            expected_sequences=args.expected_sequences,
        )
    except Exception as exc:
        print(f"V5 grouped-dev freeze failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

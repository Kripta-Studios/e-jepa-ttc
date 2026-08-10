#!/usr/bin/env python
"""Build an exact public Garl validation subset from frozen prediction tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS  # noqa: E402

JOIN_KEYS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
)


class _AtomicOutput(AbstractContextManager[Path]):
    def __init__(self, target: Path) -> None:
        self.target = target
        self.staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")

    def __enter__(self) -> Path:
        if self.target.exists():
            raise FileExistsError(f"output directory already exists: {self.target}")
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if exc_type is not None:
            if self.staging.exists():
                if (
                    self.staging.parent != self.target.parent
                    or not self.staging.name.startswith(f".{self.target.name}.staging-")
                ):
                    raise PermissionError("refusing to clean an unverified staging path")
                shutil.rmtree(self.staging)
            return None
        os.replace(self.staging, self.target)
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_list_hash(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _validate_unique_tokens(frame: pd.DataFrame, source: Path) -> pd.Series:
    tokens = cast(pd.Series, frame["sample_token"]).astype(str)
    if bool(tokens.isna().to_numpy().any()) or bool(
        (tokens.str.len() == 0).to_numpy().any()
    ):
        raise ValueError(f"{source} contains empty sample_token values")
    duplicated = cast(pd.Series, tokens[tokens.duplicated(keep=False)])
    if not duplicated.empty:
        raise ValueError(
            f"{source} contains duplicate sample_token values: "
            f"{sorted(duplicated.unique().tolist())[:5]}"
        )
    return tokens


def _ordered_subset(
    source: pd.DataFrame,
    desired_tokens: list[str],
    source_path: Path,
) -> pd.DataFrame:
    tokens = _validate_unique_tokens(source, source_path)
    indexed = source.assign(sample_token=tokens).set_index("sample_token", drop=False)
    missing = sorted(set(desired_tokens) - set(indexed.index))
    if missing:
        raise ValueError(
            f"{source_path} is missing {len(missing)} requested tokens: {missing[:5]}"
        )
    subset = indexed.loc[desired_tokens].reset_index(drop=True)
    if subset["sample_token"].astype(str).tolist() != desired_tokens:
        raise RuntimeError("subset token order differs from frozen prediction order")
    return subset


def _bucket_counts(values: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, lower, upper in BUCKETS:
        selected = (values > lower) & (values <= upper)
        counts[name] = int(selected.sum())
    return counts


def build_subset(
    *,
    predictions_path: Path,
    data_parquet: Path,
    labels_parquet: Path,
    output_dir: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Filter public train parquets to an exact frozen validation token list."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    for source in (predictions_path, data_parquet, labels_parquet):
        if not source.is_file():
            raise FileNotFoundError(source)
    predictions = pd.read_csv(predictions_path)
    _require_columns(
        predictions,
        {"sample_token", "sequence_id", "target_ttc_s"},
        predictions_path,
    )
    desired_tokens = _validate_unique_tokens(predictions, predictions_path).tolist()
    if len(desired_tokens) != expected_count:
        raise ValueError(
            f"prediction token count {len(desired_tokens)} differs from {expected_count}"
        )
    prediction_sequences = predictions["sequence_id"].astype(str)
    if prediction_sequences.str.len().eq(0).any():
        raise ValueError("predictions contain an empty sequence_id")

    data = pd.read_parquet(data_parquet)
    labels = pd.read_parquet(labels_parquet)
    _require_columns(data, set(JOIN_KEYS), data_parquet)
    _require_columns(labels, set(JOIN_KEYS) | {"ttc"}, labels_parquet)
    data_subset = _ordered_subset(data, desired_tokens, data_parquet)
    labels_subset = _ordered_subset(labels, desired_tokens, labels_parquet)
    for key in JOIN_KEYS:
        left = data_subset[key].astype(str).to_numpy()
        right = labels_subset[key].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"filtered public data/labels disagree on join key {key!r}")
    data_sequences = data_subset["sequence_id"].astype(str).to_numpy()
    if not np.array_equal(data_sequences, prediction_sequences.to_numpy()):
        raise ValueError("prediction sequence IDs disagree with public data rows")
    official_ttc = np.asarray(
        pd.to_numeric(cast(pd.Series, labels_subset["ttc"]), errors="coerce"),
        dtype=np.float64,
    )
    prediction_ttc = np.asarray(
        pd.to_numeric(
            cast(pd.Series, predictions["target_ttc_s"]), errors="coerce"
        ),
        dtype=np.float64,
    )
    if not np.isfinite(official_ttc).all() or not np.isfinite(prediction_ttc).all():
        raise ValueError("TTC targets must be finite")
    if not np.allclose(official_ttc, prediction_ttc, rtol=0.0, atol=1.0e-6):
        maximum = float(np.max(np.abs(official_ttc - prediction_ttc)))
        raise ValueError(f"prediction/public TTC targets differ; maximum error={maximum}")

    sequences = list(dict.fromkeys(prediction_sequences.tolist()))
    if not sequences:
        raise ValueError("validation subset contains no sequences")
    with _AtomicOutput(output_dir) as staging:
        output_data = staging / "data.parquet"
        output_labels = staging / "labels.parquet"
        output_assets = staging / "assets.txt"
        data_subset.to_parquet(output_data, index=False)
        labels_subset.to_parquet(output_labels, index=False)
        output_assets.write_text("\n".join(sequences) + "\n", encoding="utf-8")
        roundtrip_data = pd.read_parquet(output_data)
        roundtrip_labels = pd.read_parquet(output_labels)
        if roundtrip_data["sample_token"].astype(str).tolist() != desired_tokens:
            raise RuntimeError("data parquet roundtrip changed frozen token order")
        if roundtrip_labels["sample_token"].astype(str).tolist() != desired_tokens:
            raise RuntimeError("labels parquet roundtrip changed frozen token order")
        payload: dict[str, Any] = {
            "artifact_type": "garl_public_validation_exact_subset_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "source_scope": "public_train_parquets_filtered_by_validation_predictions",
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": _sha256(predictions_path),
            },
            "sources": {
                "data": {"path": str(data_parquet.resolve()), "sha256": _sha256(data_parquet)},
                "labels": {
                    "path": str(labels_parquet.resolve()),
                    "sha256": _sha256(labels_parquet),
                },
            },
            "outputs": {
                "data": {"path": "data.parquet", "sha256": _sha256(output_data)},
                "labels": {"path": "labels.parquet", "sha256": _sha256(output_labels)},
                "assets": {"path": "assets.txt", "sha256": _sha256(output_assets)},
            },
            "sample_count": len(desired_tokens),
            "sequence_count": len(sequences),
            "sequence_ids": sequences,
            "sample_tokens_sha256": _canonical_list_hash(desired_tokens),
            "bucket_counts": _bucket_counts(official_ttc),
            "join_keys": list(JOIN_KEYS),
            "one_to_one_join_verified": True,
            "target_equality_atol": 1.0e-6,
        }
        sign_artifact(payload)
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--data-parquet",
        type=Path,
        default=Path(r"E:\GarlTTC_dataset\data\train.parquet"),
    )
    parser.add_argument(
        "--labels-parquet",
        type=Path,
        default=Path(r"E:\GarlTTC_dataset\annotations\train.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=2048)
    args = parser.parse_args()
    try:
        payload = build_subset(
            predictions_path=cast(Path, args.predictions).resolve(),
            data_parquet=cast(Path, args.data_parquet).resolve(),
            labels_parquet=cast(Path, args.labels_parquet).resolve(),
            output_dir=cast(Path, args.output_dir).resolve(),
            expected_count=int(args.expected_count),
        )
    except Exception as error:
        parser.exit(2, f"Garl validation subset build failed: {type(error).__name__}: {error}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

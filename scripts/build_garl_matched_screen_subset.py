"""Materialize the exact public 2048/2048 screen rows for matched Garl training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402

JOIN_KEYS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _scalar_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)  # type: ignore[arg-type]


def _scalar_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)  # type: ignore[arg-type]


def _cache_rows(cache_manifest: Path) -> dict[str, pd.DataFrame]:
    manifest = _read_json(cache_manifest)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Cache manifest contains no shards.")
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    root = cache_manifest.parent
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise TypeError("Cache shard entry must be a mapping.")
        split = str(shard.get("split"))
        if split not in rows:
            raise ValueError(f"Unexpected cache split: {split!r}")
        path = root / str(shard.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"Cache shard not found: {path}")
        expected_hash = str(shard.get("sha256"))
        if _sha256(path) != expected_hash:
            raise ValueError(f"Cache shard hash mismatch: {path}")
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list) or len(records) != int(shard.get("count", -1)):
            raise ValueError(f"Cache shard row count mismatch: {path}")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(f"Cache record is not a mapping: {path}")
            rows[split].append(
                {
                    "sequence_id": str(record["sequence_id"]),
                    "sample_token": str(record["sample_token"]),
                    "track_id": str(record["track_id"]),
                    "public_track_id": str(record["public_track_id"]),
                    "timestamp_us": _scalar_int(record["timestamp_us"]),
                    "cache_ttc_s": _scalar_float(record["ttc_s"]),
                }
            )
        del records
    result = {split: pd.DataFrame(values) for split, values in rows.items()}
    expected_counts = manifest.get("split_counts", {})
    for split, frame in result.items():
        if len(frame) != int(expected_counts.get(split, -1)):
            raise ValueError(f"Cache manifest split count mismatch for {split}.")
        if frame["sample_token"].duplicated().any():
            raise ValueError(f"Cache {split} contains duplicate sample tokens.")
    if set(result["train"]["sequence_id"]) & set(result["validation"]["sequence_id"]):
        raise ValueError("Cache train and validation sequence groups overlap.")
    return result


def _select_exact_public_rows(
    cache_rows: pd.DataFrame,
    public_data: pd.DataFrame,
    public_labels: pd.DataFrame,
    *,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for label, frame, required in (
        ("data", public_data, set(JOIN_KEYS)),
        ("labels", public_labels, set(JOIN_KEYS) | {"ttc"}),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Public {label} parquet is missing columns: {missing}")
        if frame["sample_token"].astype(str).duplicated().any():
            raise ValueError(f"Public {label} parquet has duplicate sample tokens.")
    tokens = cache_rows["sample_token"].astype(str).tolist()
    data = public_data[public_data["sample_token"].astype(str).isin(tokens)].copy()
    labels = public_labels[public_labels["sample_token"].astype(str).isin(tokens)].copy()
    if len(data) != len(cache_rows) or len(labels) != len(cache_rows):
        raise ValueError(
            f"Public {split} selection is incomplete: cache={len(cache_rows)}, "
            f"data={len(data)}, labels={len(labels)}"
        )
    selected_label_columns = list(JOIN_KEYS) + ["ttc"]
    audit_labels = labels.loc[:, selected_label_columns].copy()
    audit = cache_rows.merge(
        audit_labels,
        on=list(JOIN_KEYS),
        validate="one_to_one",
        how="inner",
    )
    if len(audit) != len(cache_rows):
        raise ValueError(f"Public {split} join keys disagree with cache records.")
    if not np.allclose(
        audit["cache_ttc_s"].to_numpy(dtype=np.float64),
        audit["ttc"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(f"Public {split} TTC targets disagree with cache records.")
    order = {token: index for index, token in enumerate(tokens)}
    data["_matched_order"] = [order[str(value)] for value in data["sample_token"]]
    labels["_matched_order"] = [order[str(value)] for value in labels["sample_token"]]
    data_order = np.argsort(np.asarray(data["_matched_order"], dtype=np.int64))
    label_order = np.argsort(np.asarray(labels["_matched_order"], dtype=np.int64))
    data = (
        data.iloc[data_order]
        .drop(columns="_matched_order")
        .reset_index(drop=True)
    )
    labels = (
        labels.iloc[label_order]
        .drop(columns="_matched_order")
        .reset_index(drop=True)
    )
    return data, labels


def _write_role(
    destination: Path,
    split: str,
    data: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    data_path = destination / f"{split}_data.parquet"
    labels_path = destination / f"{split}_labels.parquet"
    assets_path = destination / f"{split}_assets.txt"
    data.to_parquet(data_path, index=False)
    labels.to_parquet(labels_path, index=False)
    sequences = sorted(data["sequence_id"].astype(str).unique().tolist())
    assets_path.write_text("\n".join(sequences) + "\n", encoding="utf-8")
    roundtrip_data = pd.read_parquet(data_path)
    roundtrip_labels = pd.read_parquet(labels_path)
    if not data.equals(roundtrip_data) or not labels.equals(roundtrip_labels):
        raise RuntimeError(f"Parquet roundtrip changed the matched {split} rows.")
    return {
        "rows": len(data),
        "sequences": sequences,
        "sequence_count": len(sequences),
        "data": {"path": data_path.name, "sha256": _sha256(data_path)},
        "labels": {"path": labels_path.name, "sha256": _sha256(labels_path)},
        "assets": {"path": assets_path.name, "sha256": _sha256(assets_path)},
    }


def build_subset(
    *,
    cache_manifest: Path,
    public_data_parquet: Path,
    public_labels_parquet: Path,
    validation_subset_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build and sign exact train/validation parquets for the matched baseline."""

    for path in (
        cache_manifest,
        public_data_parquet,
        public_labels_parquet,
        validation_subset_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required matched-subset input not found: {path}")
    cache = _cache_rows(cache_manifest)
    public_data = pd.read_parquet(public_data_parquet)
    public_labels = pd.read_parquet(public_labels_parquet)
    selected = {
        split: _select_exact_public_rows(rows, public_data, public_labels, split=split)
        for split, rows in cache.items()
    }
    validation_manifest = _read_json(validation_subset_manifest)
    expected_validation_tokens = set(cache["validation"]["sample_token"].astype(str))
    validation_data_path = validation_subset_manifest.parent / str(
        validation_manifest["outputs"]["data"]["path"]
    )
    prior_validation = pd.read_parquet(validation_data_path, columns=["sample_token"])
    if set(prior_validation["sample_token"].astype(str)) != expected_validation_tokens:
        raise ValueError("Existing exact validation subset does not match validation cache tokens.")

    output_dir.mkdir(parents=True, exist_ok=True)
    roles = {
        split: _write_role(output_dir, split, data, labels)
        for split, (data, labels) in selected.items()
    }
    report: dict[str, Any] = {
        "artifact_type": "garl_event_only_matched_screen_subset_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_train_validation_only",
        "roles": roles,
        "checks": {
            "exact_cache_tokens": True,
            "exact_join_keys": True,
            "target_equality": True,
            "parquet_roundtrip": True,
            "train_validation_sequence_disjoint": True,
            "existing_validation_subset_token_equality": True,
            "bbox_used_by_official_preprocessing_only": True,
            "bbox_is_not_direct_model_input": True,
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
        },
        "sources": {
            "cache_manifest": {
                "path": str(cache_manifest.resolve()),
                "sha256": _sha256(cache_manifest),
                "artifact_sha256": _read_json(cache_manifest).get("artifact_sha256"),
            },
            "public_data": {
                "path": str(public_data_parquet.resolve()),
                "sha256": _sha256(public_data_parquet),
            },
            "public_labels": {
                "path": str(public_labels_parquet.resolve()),
                "sha256": _sha256(public_labels_parquet),
            },
            "validation_subset_manifest": {
                "path": str(validation_subset_manifest.resolve()),
                "sha256": _sha256(validation_subset_manifest),
                "artifact_sha256": validation_manifest.get("artifact_sha256"),
            },
        },
    }
    sign_artifact(report)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--public-data-parquet", type=Path, required=True)
    parser.add_argument("--public-labels-parquet", type=Path, required=True)
    parser.add_argument("--validation-subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_subset(
            cache_manifest=args.cache_manifest,
            public_data_parquet=args.public_data_parquet,
            public_labels_parquet=args.public_labels_parquet,
            validation_subset_manifest=args.validation_subset_manifest,
            output_dir=args.output_dir,
        )
    except Exception as error:
        print(f"matched subset build failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

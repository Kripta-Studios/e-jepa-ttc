"""Evaluate the immutable official Garl-TTC model on a materialized validation split.

The split is supplied explicitly as data parquet, labels parquet, and an asset
list.  No benchmark test input is discovered or read by this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)

JOIN_KEYS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
)


class _OfficialDataset(Protocol):
    """Structural subset of the release dataset used by this script."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any] | None: ...

    def get_collate_fn(self) -> Callable[[list[Any]], Mapping[str, Any] | None]: ...


@dataclass(frozen=True)
class OfficialAPI:
    """Late-bound interfaces imported from the immutable official release."""

    load_config: Callable[..., dict[str, Any]]
    dataset_type: Callable[..., _OfficialDataset]
    network_type: Callable[..., torch.nn.Module]
    load_checkpoint: Callable[..., None]


@dataclass(frozen=True)
class ValidationIndex:
    """Validated one-to-one validation rows selected by the asset list."""

    frame: pd.DataFrame
    assets: tuple[str, ...]


InferenceRunner = Callable[..., pd.DataFrame]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _read_assets(path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assets = tuple(line.strip() for line in lines if line.strip())
    if not assets:
        raise ValueError("Validation asset list is empty.")
    if len(set(assets)) != len(assets):
        raise ValueError("Validation asset list contains duplicate sequence IDs.")
    return assets


def _require_columns(frame: pd.DataFrame, required: Set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_validation_index(
    data_parquet: Path,
    labels_parquet: Path,
    asset_list: Path,
) -> ValidationIndex:
    """Load and validate a complete one-to-one, asset-filtered validation index."""

    for source in (data_parquet, labels_parquet, asset_list):
        if not source.is_file():
            raise FileNotFoundError(f"Required validation input not found: {source}")
    assets = _read_assets(asset_list)
    data = pd.read_parquet(data_parquet)
    labels = pd.read_parquet(labels_parquet)
    required_keys = set(JOIN_KEYS)
    _require_columns(data, required_keys, data_parquet)
    _require_columns(labels, required_keys | {"ttc"}, labels_parquet)
    for name, frame in (("data", data), ("labels", labels)):
        if frame[list(JOIN_KEYS)].isna().to_numpy().any():
            raise ValueError(f"Validation {name} parquet contains null join keys.")
        if frame.duplicated(list(JOIN_KEYS), keep=False).any():
            raise ValueError(f"Validation {name} parquet contains duplicate join keys.")

    data_selected = data[data["sequence_id"].astype(str).isin(assets)].copy()
    labels_selected = labels[labels["sequence_id"].astype(str).isin(assets)].copy()
    present = set(data_selected["sequence_id"].astype(str))
    missing_assets = sorted(set(assets) - present)
    if missing_assets:
        raise ValueError(f"Asset list sequences absent from data parquet: {missing_assets}")
    merged = data_selected.merge(
        labels_selected,
        on=list(JOIN_KEYS),
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_label"),
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        counts = merged.loc[unmatched, "_merge"].value_counts().to_dict()
        raise ValueError(f"Validation data/labels join is incomplete: {counts}")
    merged = merged.drop(columns="_merge")
    if merged.empty:
        raise ValueError("Validation split contains no selected rows.")
    if merged["sample_token"].astype(str).duplicated().any():
        raise ValueError("Validation sample_token values are not unique.")
    target = np.asarray(pd.to_numeric(merged["ttc"], errors="coerce"), dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Validation TTC labels must all be finite.")
    merged["sample_token"] = merged["sample_token"].astype(str)
    merged["sequence_id"] = merged["sequence_id"].astype(str)
    return ValidationIndex(frame=merged, assets=assets)


def _load_official_api(release_root: Path) -> OfficialAPI:
    if not release_root.is_dir():
        raise FileNotFoundError(f"Official Garl-TTC release root not found: {release_root}")
    resolved = str(release_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    config_module = importlib.import_module("garl_ttc.config")
    dataset_module = importlib.import_module("garl_ttc.datasets")
    model_module = importlib.import_module("garl_ttc.models")
    runtime_module = importlib.import_module("garl_ttc.engine.runtime")
    return OfficialAPI(
        load_config=cast(Callable[..., dict[str, Any]], config_module.load_config),
        dataset_type=cast(Callable[..., _OfficialDataset], dataset_module.TTCEstimationDataset),
        network_type=cast(Callable[..., torch.nn.Module], model_module.TTCNetwork),
        load_checkpoint=cast(Callable[..., None], runtime_module.load_checkpoint),
    )


def _flatten_strings(value: object, *, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence):
        raise TypeError(f"Official batch {field} must be a string sequence.")
    flattened: list[str] = []
    for item in value:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, Sequence) and len(item) == 1 and isinstance(item[0], str):
            flattened.append(item[0])
        else:
            raise TypeError(f"Official batch {field} has an unsupported nested value: {item!r}")
    return flattened


def _prediction_from_raw(raw: torch.Tensor, mode: str, delta_t_s: float) -> np.ndarray:
    values = raw.detach().to(dtype=torch.float64, device="cpu").numpy().reshape(len(raw), -1)
    if mode == "height_ratio":
        if values.shape[1] != 2:
            raise ValueError("Official height_ratio model must return exactly two heights.")
        ratio = values[:, 0] / values[:, 1]
        return delta_t_s / (1.0 - ratio)
    if mode == "height_ratio_direct":
        return delta_t_s / (1.0 - values[:, 0])
    if mode == "baseline":
        return values[:, 0]
    raise ValueError(f"Unsupported official prediction mode: {mode!r}")


def _run_official_inference(
    *,
    release_root: Path,
    config_path: Path,
    checkpoint: Path,
    dataset_root: Path,
    data_parquet: Path,
    labels_parquet: Path,
    asset_list: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    api = _load_official_api(release_root)
    config = api.load_config(config_path)
    dataset_config = config.setdefault("dataset", {})
    dataset_config.update({"root": str(dataset_root.resolve()), "annotation_format": "parquet"})
    dataset_config.setdefault("test", {}).update(
        {
            "asset_path": str(asset_list.resolve()),
            "data_parquet": str(data_parquet.resolve()),
            "labels_parquet": str(labels_parquet.resolve()),
        }
    )
    config.setdefault("testing_settings", {}).update(
        {"batch_size": batch_size, "num_threads": num_workers, "shuffle": False}
    )
    selected_device = (
        ("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        if device == "auto"
        else device
    )
    torch_device = torch.device(selected_device)
    model = api.network_type(config, is_train=False).to(torch_device)
    api.load_checkpoint(model, checkpoint, torch_device, strict=True)
    model.eval()
    dataset = api.dataset_type(config, "test", seed=0, db_mode="window")
    loader = DataLoader(
        cast(Dataset[Any], dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=False,
    )
    mode = str(config["model"]["mode"])
    delta_t_s = float(cast(Any, model).dT)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            data = cast(torch.Tensor, batch["data"]).to(torch_device, non_blocking=True)
            target = cast(torch.Tensor, batch["target"]).detach().cpu().numpy().reshape(-1)
            raw_output = model(data)
            raw = raw_output[0] if isinstance(raw_output, tuple) else raw_output
            if not isinstance(raw, torch.Tensor):
                raise TypeError("Official TTCNetwork returned a non-tensor prediction.")
            prediction = _prediction_from_raw(raw, mode, delta_t_s)
            tokens = _flatten_strings(batch.get("sample_token"), field="sample_token")
            if not (len(tokens) == len(target) == len(prediction)):
                raise ValueError("Official batch token, target, and prediction counts differ.")
            for index, token in enumerate(tokens):
                rows.append(
                    {
                        "sample_token": token,
                        "target_from_loader_ttc_s": float(target[index]),
                        "predicted_ttc_s": float(prediction[index]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(
    *,
    release_root: Path,
    config_path: Path,
    checkpoint: Path,
    dataset_root: Path,
    data_parquet: Path,
    labels_parquet: Path,
    asset_list: Path,
    output_dir: Path,
    device: str = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    inference_runner: InferenceRunner | None = None,
) -> dict[str, Any]:
    """Run strict validation-only inference and write predictions plus signed metrics."""

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative.")
    for source in (config_path, checkpoint):
        if not source.is_file():
            raise FileNotFoundError(f"Official release artifact not found: {source}")
    index = load_validation_index(data_parquet, labels_parquet, asset_list)
    tracked_release_files = {
        "config": config_path,
        "checkpoint": checkpoint,
        "ttc_network_source": release_root / "garl_ttc" / "models" / "ttc_network.py",
        "dataset_source": release_root / "garl_ttc" / "datasets" / "ttc_dataset.py",
    }
    for source in tracked_release_files.values():
        if not source.is_file():
            raise FileNotFoundError(f"Official release artifact not found: {source}")
    release_hashes_before = {
        name: _sha256_file(path) for name, path in tracked_release_files.items()
    }
    runner = inference_runner or _run_official_inference
    predictions = runner(
        release_root=release_root,
        config_path=config_path,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        data_parquet=data_parquet,
        labels_parquet=labels_parquet,
        asset_list=asset_list,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    _require_columns(
        predictions,
        {"sample_token", "target_from_loader_ttc_s", "predicted_ttc_s"},
        Path("<inference output>"),
    )
    predictions = predictions.copy()
    predictions["sample_token"] = predictions["sample_token"].astype(str)
    if predictions["sample_token"].duplicated().any():
        raise ValueError("Official inference produced duplicate sample_token values.")
    expected_tokens = set(index.frame["sample_token"])
    actual_tokens = set(predictions["sample_token"])
    if expected_tokens != actual_tokens:
        raise ValueError(
            "Official inference token set mismatch: "
            f"missing={len(expected_tokens - actual_tokens)}, "
            f"extra={len(actual_tokens - expected_tokens)}"
        )
    reference = index.frame[["sample_token", "sequence_id"]].copy()
    reference["target_ttc_s"] = np.asarray(index.frame["ttc"], dtype=np.float64)
    output = reference.merge(predictions, on="sample_token", validate="one_to_one")
    loader_target = output["target_from_loader_ttc_s"].to_numpy(dtype=np.float64)
    target = output["target_ttc_s"].to_numpy(dtype=np.float64)
    if not np.allclose(loader_target, target, rtol=1e-5, atol=1e-6):
        raise ValueError("Targets returned by the official loader disagree with labels parquet.")
    prediction = output["predicted_ttc_s"].to_numpy(dtype=np.float64)
    signed = signed_garl_metrics(target, prediction)
    macro = sequence_macro_signed_metrics(target, prediction, output["sequence_id"].astype(str))

    release_hashes_after = {
        name: _sha256_file(path) for name, path in tracked_release_files.items()
    }
    if release_hashes_after != release_hashes_before:
        raise RuntimeError("Official Garl-TTC release artifacts changed during evaluation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    output.to_parquet(predictions_path, index=False)
    provenance: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "validation_only": True,
        "test_data_used": False,
        "release_modified": False,
        "release_root": str(release_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "inputs": {
            "data_parquet": {
                "path": str(data_parquet.resolve()),
                "sha256": _sha256_file(data_parquet),
            },
            "labels_parquet": {
                "path": str(labels_parquet.resolve()),
                "sha256": _sha256_file(labels_parquet),
            },
            "asset_list": {"path": str(asset_list.resolve()), "sha256": _sha256_file(asset_list)},
        },
        "official_release_artifacts": {
            name: {"path": str(tracked_release_files[name].resolve()), "sha256": digest}
            for name, digest in release_hashes_before.items()
        },
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
            **_git_state(ROOT),
        },
        "official_release_git": _git_state(release_root),
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "asset_count": len(index.assets),
        "sample_count": len(output),
    }
    report: dict[str, Any] = {
        "artifact_type": "official_garl_validation_evaluation_v1",
        "status": "completed",
        "signed_garl_metrics": signed,
        "sequence_macro_signed_metrics": macro,
        "predictions": {
            "path": str(predictions_path.resolve()),
            "sha256": _sha256_file(predictions_path),
            "rows": len(output),
        },
        "provenance": provenance,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the official validation evaluator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-parquet", type=Path, required=True)
    parser.add_argument("--labels-parquet", type=Path, required=True)
    parser.add_argument("--asset-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning non-zero for every validation or runtime failure."""

    args = parse_args(argv)
    release_root = cast(Path, args.release_root)
    config_path = cast(Path | None, args.config) or (
        release_root / "configs" / "garl_ttc_eventdecoder.yaml"
    )
    checkpoint = cast(Path | None, args.checkpoint) or (
        release_root / "checkpoints" / "paper_ours_full.pth"
    )
    try:
        report = evaluate(
            release_root=release_root,
            config_path=config_path,
            checkpoint=checkpoint,
            dataset_root=cast(Path, args.dataset_root),
            data_parquet=cast(Path, args.data_parquet),
            labels_parquet=cast(Path, args.labels_parquet),
            asset_list=cast(Path, args.asset_list),
            output_dir=cast(Path, args.output_dir),
            device=str(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
    except Exception as error:  # CLI boundary: preserve a deterministic non-zero status.
        message = f"official Garl validation evaluation failed: {type(error).__name__}: {error}"
        print(message, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

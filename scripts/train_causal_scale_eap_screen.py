#!/usr/bin/env python
"""Train the event causal-scale arm on a bounded public eAP/Garl validation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.reproducibility import environment_snapshot, resolve_device  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import (  # noqa: E402
    CausalScaleEAPTrainingConfig,
    checkpoint_payload,
    train_real_causal_scale,
)
from scripts.evaluate_causal_scale_v5_operator import (  # noqa: E402
    _classify_worktree_status,
)

DEFAULT_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def _resolve(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("path references must be strings")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _model_config(path: Path) -> CausalScaleTTCConfig:
    raw = _read_yaml(path)
    if raw.pop("model", None) != "e_jepa_causal_scale_event_v8":
        raise ValueError("real screen requires the causal-scale event v8 model")
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return CausalScaleTTCConfig(**raw)


def _finite_json(value: object) -> object:
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {str(key): _finite_json(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_finite_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reset_peak_memory_stats(device: torch.device) -> None:
    """Reset CUDA peak accounting without passing a device to the Windows API."""

    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()


def run(
    config_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    """Execute a train/validation-only run; no benchmark test labels are opened."""

    raw = _read_yaml(config_path)
    experiment = raw.get("experiment")
    data = raw.get("data")
    if not isinstance(experiment, dict) or not isinstance(data, dict):
        raise ValueError("experiment and data sections are required")
    if data.get("opened_splits") != ["train", "validation"]:
        raise ValueError("this screen may open train and validation only")
    forbidden = ("official_test_opened", "codabench_opened", "evttc_test_opened")
    if any(data.get(key) is not False for key in forbidden):
        raise ValueError("private/CodaBench/EvTTC test access must remain false")
    manifest_path = _resolve(data["cache_manifest"])
    expected_manifest_hash = str(data["cache_manifest_sha256"])
    actual_manifest_hash = _sha256(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("cache manifest hash differs from the frozen protocol")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != data.get("cache_artifact_sha256"):
        raise ValueError("cache artifact identity differs from the frozen protocol")
    if manifest.get("split_counts") != {"train": 2048, "validation": 2048}:
        raise ValueError("frozen screen requires exactly 2048 train and validation rows")
    train_sequences = {str(value) for value in data.get("train_sequence_ids", [])}
    validation_sequences = {
        str(value) for value in data.get("validation_sequence_ids", [])
    }
    if len(train_sequences) != 9 or len(validation_sequences) != 3:
        raise ValueError("frozen protocol requires 9 train and 3 validation sequences")
    if train_sequences & validation_sequences:
        raise ValueError("train and validation sequence IDs overlap")

    status_lines = _git(
        "-c", "core.quotepath=false", "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    worktree = _classify_worktree_status(status_lines)
    code_dirty = bool(worktree["tracked_dirty"] or worktree["untracked_code_paths"])
    if code_dirty:
        raise RuntimeError("representative real screen requires clean tracked/code state")

    model_path = _resolve(raw["model_config"])
    model_config = _model_config(model_path)
    training_raw = raw.get("training")
    loss_raw = raw.get("loss")
    if not isinstance(training_raw, dict) or not isinstance(loss_raw, dict):
        raise ValueError("training and loss mappings are required")
    training_config = CausalScaleEAPTrainingConfig(**training_raw)
    loss_config = CausalScaleTTCLossConfig(**loss_raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    train_dataset = GarlTTCObjectEventV4Dataset(
        str(manifest_path), splits=("train",)
    )
    validation_dataset = GarlTTCObjectEventV4Dataset(
        str(manifest_path), splits=("validation",)
    )
    device = resolve_device(device_name)
    _reset_peak_memory_stats(device)
    result = train_real_causal_scale(
        model_config,
        training_config,
        loss_config,
        train_dataset,
        validation_dataset,
        device,
        checkpoint_dir=state_dir,
        resume=resume,
    )
    checkpoint_path = output_dir / "model_best.pt"
    temporary_checkpoint = output_dir / ".model_best.pt.tmp"
    torch.save(
        checkpoint_payload(result, training_config, loss_config), temporary_checkpoint
    )
    os.replace(temporary_checkpoint, checkpoint_path)
    validation = result.best_validation
    predictions = pd.DataFrame(
        {
            "sample_token": validation["sample_tokens"],
            "sequence_id": validation["sequence_ids"],
            "target_ttc_s": validation["target_ttc_s"],
            "prediction_ttc_s": validation["prediction_ttc_s"],
        }
    )
    predictions_path = output_dir / "validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics = {
        key: value
        for key, value in validation.items()
        if key not in {"sample_tokens", "sequence_ids", "target_ttc_s", "prediction_ttc_s"}
    }
    payload: dict[str, Any] = {
        "artifact_type": "causal_scale_eap_public_validation_screen_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "selectable": False,
        "sota_claim_authorized": False,
        "official_test_opened": False,
        "garl_comparison_pending": True,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": code_dirty,
        "worktree": worktree,
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "model_config": {
            "path": model_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(model_path),
        },
        "cache": {
            "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": actual_manifest_hash,
            "artifact_sha256": manifest["artifact_sha256"],
            "split_counts": manifest["split_counts"],
            "train_sequence_ids": sorted(train_sequences),
            "validation_sequence_ids": sorted(validation_sequences),
        },
        "model_input_contract": {
            "forward_inputs": ["event_v4_common_roi", "garl_delta_t_s"],
            "weak_bbox_supervision_only": True,
            "bbox_is_not_segmentation_ground_truth": True,
            "t0_proxy_box_excluded": training_config.mask_t0_as_proxy,
        },
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "selection": {
            "split": "validation",
            "best_epoch": result.best_epoch,
            **result.best_selection,
        },
        "validation_metrics": metrics,
        "history": result.history,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else None
        ),
        "checkpoint": {"path": checkpoint_path.name, "sha256": _sha256(checkpoint_path)},
        "predictions": {"path": predictions_path.name, "sha256": _sha256(predictions_path)},
        "environment": environment_snapshot(),
        "device": str(device),
        "decision_contract": raw["decision_contract"],
    }
    payload = cast(dict[str, Any], _finite_json(payload))
    sign_artifact(payload)
    _atomic_json(output_dir / "summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(
            args.config.resolve(),
            args.output_dir.resolve(),
            device_name=args.device,
            resume=args.resume,
        )
    except Exception as error:
        parser.exit(2, f"causal-scale eAP screen failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_vram_mb": payload["peak_vram_mb"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

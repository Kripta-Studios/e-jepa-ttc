"""Production V8 temporal trainer backed by the audited A5 training loop.

The cache is V8-native; only the *training-only* adapter translates its raster and
geometry fields to the established causal-scale batch contract.  This deliberately
does not route V8 rows through the historical V4 data loader.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.dinov3_relational_teacher_cache import DINOv3RelationalTeacherDataset
from e_jepa_ttc.data.scientific_recovery_v8_adapter import V8ToObjectEventV4Dataset
from e_jepa_ttc.data.scientific_recovery_v8_cache import (
    ScientificRecoveryV8CacheDataset,
    collate_scientific_recovery_v8,
)
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig, causal_scale_ttc_loss
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
from e_jepa_ttc.reproducibility import environment_snapshot, resolve_device
from e_jepa_ttc.training.causal_scale_eap import (
    CausalScaleEAPTrainingConfig,
    checkpoint_payload,
    train_real_causal_scale,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mapping_dataclass(cls: type[Any], raw: Mapping[str, Any]) -> Any:  # noqa: ANN401
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


def _model(path: Path) -> CausalScaleTTCConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model config must be a mapping")
    raw.pop("model", None)
    raw["risk_thresholds_s"] = tuple(raw["risk_thresholds_s"])
    return CausalScaleTTCConfig(**raw)


def _closed_cache(
    path: Path, fixture: bool
) -> tuple[ScientificRecoveryV8CacheDataset, Mapping[str, Any]]:
    cache = ScientificRecoveryV8CacheDataset(path)
    manifest = cache.manifest
    if manifest.get("sealed_splits_opened") is not False or manifest.get("train_only") is not True:
        raise ValueError("V8 training accepts only a signed closed train cache")
    if bool(manifest.get("fixture_only")) != fixture:
        raise ValueError("fixture cache and --fixture-smoke must agree exactly")
    return cache, manifest


def run_v8_temporal_training(
    *,
    config_path: Path,
    output_dir: Path,
    device_name: str,
    resume: bool = False,
    fixture_smoke: bool = False,
) -> dict[str, Any]:
    """Run one frozen fold using 18-epoch A5 optimization and signed V8 evidence."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V8 fold config must be a mapping")
    exp, data, training, loss = (
        config.get(key) for key in ("experiment", "data", "training", "loss")
    )
    if not all(isinstance(value, dict) for value in (exp, data, training, loss)):
        raise ValueError("V8 fold config lacks experiment/data/training/loss mappings")
    assert (
        isinstance(exp, dict)
        and isinstance(data, dict)
        and isinstance(training, dict)
        and isinstance(loss, dict)
    )
    if fixture_smoke:
        training = dict(training)
        training.update(
            {
                "epochs": 1,
                "minimum_epochs": 1,
                "foreground_warmup_epochs": 0,
                "early_stopping_patience": 1,
                "batch_size": 2,
                "precision": "fp32",
                "maximum_runtime_hours": 0.1,
            }
        )
    elif int(training.get("epochs", 0)) < 18 or int(training.get("minimum_epochs", 0)) < 8:
        raise ValueError("production V8 requires the frozen 18-epoch/minimum-8 contract")
    cache_path = Path(str(data["cache_manifest"]))
    if not cache_path.is_absolute():
        cache_path = (config_path.parents[3] / cache_path).resolve()
    cache, cache_manifest = _closed_cache(cache_path, fixture_smoke)
    fold = int(data["outer_fold"])
    train = V8ToObjectEventV4Dataset(cache, outer_fold=fold, split="train")
    dev = V8ToObjectEventV4Dataset(cache, outer_fold=fold, split="dev")
    dev_sample_weights = {
        str(dev[index]["sample_token"]): float(dev[index]["sample_weight"])
        for index in range(len(dev))
    }
    train_cfg = _mapping_dataclass(CausalScaleEAPTrainingConfig, training)
    # DINO is mandatory when the frozen configuration requests it, and is wrapped
    # around the train view only.  No dev row can be loaded by this teacher cache.
    dino = data.get("dinov3_relational_teacher")
    if train_cfg.representation_supervision != "none":
        if not isinstance(dino, dict):
            raise ValueError("V8 DINO supervision requires signed data.dinov3_relational_teacher")
        manifest_path = Path(str(dino["manifest"]))
        if not manifest_path.is_absolute():
            manifest_path = (config_path.parents[3] / manifest_path).resolve()
        train = DINOv3RelationalTeacherDataset(
            train,
            manifest_path=manifest_path,
            expected_artifact_sha256=str(dino["artifact_sha256"]),
            expected_manifest_sha256=str(dino["manifest_sha256"]),
            allowed_sample_tokens={
                str(train[index]["sample_token"]) for index in range(len(train))
            },
        )
        if train_cfg.representation_teacher_cache_artifact_sha256 != str(dino["artifact_sha256"]):
            raise ValueError("DINO teacher hash differs between training and data contracts")
    model_ref = Path(str(config["model_config"]))
    model_path = (
        model_ref if model_ref.is_absolute() else (config_path.parents[3] / model_ref).resolve()
    )
    model_cfg = _model(model_path)
    if model_cfg.in_channels != int(cache.shape[1]):
        raise ValueError("frozen model input channels differ from signed V8 cache")
    loss_cfg = _mapping_dataclass(CausalScaleTTCLossConfig, loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_real_causal_scale(
        model_cfg,
        train_cfg,
        loss_cfg,
        train,
        dev,
        resolve_device(device_name),
        checkpoint_dir=output_dir / "state",
        resume=resume,
    )
    checkpoint = output_dir / "model_best.pt"
    torch.save(
        checkpoint_payload(
            result, train_cfg, loss_cfg, artifact_type="causal_scale_eap_grouped_dev_checkpoint_v1"
        ),
        checkpoint,
    )
    validation = result.best_validation
    predictions_path = output_dir / "dev_predictions.csv"
    frame = pd.DataFrame(
        {
            "sample_token": validation["sample_tokens"],
            "sequence_id": validation["sequence_ids"],
            "track_id": validation["track_ids"],
            "target_ttc_s": validation["target_ttc_s"],
            "prediction_ttc_s": validation["prediction_ttc_s"],
            "fold": fold,
            "seed": train_cfg.seed,
            "sample_weight": [
                dev_sample_weights[str(token)] for token in validation["sample_tokens"]
            ],
        }
    )
    frame.to_csv(predictions_path, index=False, lineterminator="\n")
    metrics_path = output_dir / "train_metrics.json"
    _atomic_json(
        metrics_path,
        {
            "history": result.history,
            "best_selection": result.best_selection,
            "best_epoch": result.best_epoch,
            "environment": environment_snapshot(),
        },
    )
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_fold_result_v1",
        "status": "fixture_smoke_completed"
        if fixture_smoke
        else "completed_train_only_grouped_dev",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": exp["name"],
        "arm": exp["arm"],
        "fold": fold,
        "seed": train_cfg.seed,
        "fixture_smoke": fixture_smoke,
        "config": {"path": config_path.as_posix(), "sha256": _sha(config_path)},
        "model_config": {"path": model_path.as_posix(), "sha256": _sha(model_path)},
        "protocol": cache_manifest.get("protocol"),
        "frozen_manifest": cache_manifest.get("frozen_manifest"),
        "cache": {
            "path": cache_path.as_posix(),
            "sha256": _sha(cache_path),
            "artifact_sha256": cache_manifest.get("artifact_sha256"),
        },
        "checkpoint": {"path": checkpoint.name, "sha256": _sha(checkpoint)},
        "predictions": {
            "path": predictions_path.name,
            "sha256": _sha(predictions_path),
            "rows": len(frame),
        },
        "metrics": {
            "path": metrics_path.name,
            "sha256": _sha(metrics_path),
            "selection": result.best_selection,
        },
        "closed_evaluation": {
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }
    sign_artifact(payload)
    _atomic_json(output_dir / "summary.json", payload)
    return payload


__all__ = ["run_v8_temporal_training"]


def run_v8_cache_smoke(
    *, cache_manifest: Path, output_dir: Path, outer_fold: int, allow_fixture: bool = False
) -> dict[str, Any]:
    """Legacy fixture-only one-update proof, intentionally not a fold result.

    Kept for the narrow cache contract test.  Production callers must use
    :func:`run_v8_temporal_training`, and aggregators reject this artifact type.
    """
    if not allow_fixture:
        raise ValueError("fixture V8 smoke requires explicit allow_fixture")
    cache, manifest = _closed_cache(cache_manifest, True)
    from torch.utils.data import DataLoader, Subset

    train_indices = [i for i in range(len(cache)) if int(cache[i]["outer_fold"]) != outer_fold]
    dev_indices = [i for i in range(len(cache)) if int(cache[i]["outer_fold"]) == outer_fold]
    if not train_indices or not dev_indices:
        raise ValueError("fixture needs train and dev rows")
    model = __import__(
        "e_jepa_ttc.models.causal_scale_ttc", fromlist=["CausalScaleTTC"]
    ).CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=int(cache.shape[1]),
            hidden_dim=8,
            geometry_dim=16,
            residual_depth=1,
            foreground_temporal_smoothing_mode="causal_left",
        )
    )
    batch = next(
        iter(
            DataLoader(
                Subset(cache, train_indices),
                batch_size=2,
                collate_fn=collate_scientific_recovery_v8,
            )
        )
    )
    delta = (batch.endpoint_us[:, 1:] - batch.endpoint_us[:, :-1]).float() / 1_000_000.0
    output = model(batch.representations, delta)
    loss = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=batch.target_ttc,
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
    ).total
    if not torch.isfinite(loss):
        raise RuntimeError("fixture V8 loss is non-finite")
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "state").mkdir(exist_ok=True)
    checkpoint = output_dir / "state" / "last.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    rows = [
        {
            "sample_token": cache[i]["sample_token"],
            "sequence_id": cache[i]["sequence_id"],
            "track_id": cache[i]["track_id"],
            "target_ttc_s": float(cache[i]["target_ttc"]),
            "prediction_ttc_s": float("nan"),
            "fold": outer_fold,
        }
        for i in dev_indices
    ]
    prediction = output_dir / "dev_predictions.csv"
    pd.DataFrame(rows).to_csv(prediction, index=False)
    payload = {
        "artifact_type": "scientific_recovery_v8_cache_training_smoke_v1",
        "status": "completed_train_only_grouped_dev",
        "outer_fold": outer_fold,
        "fixture_smoke": True,
        "cache_manifest": {"path": str(cache_manifest), "sha256": _sha(cache_manifest)},
        "cache_artifact_sha256": manifest.get("artifact_sha256"),
        "checkpoint": {"path": "state/last.pt", "sha256": _sha(checkpoint)},
        "predictions": {"path": prediction.name, "sha256": _sha(prediction)},
        "one_update_loss": float(loss.detach()),
    }
    sign_artifact(payload)
    _atomic_json(output_dir / "summary.json", payload)
    return payload


__all__.append("run_v8_cache_smoke")


def export_router_expert_predictions(
    *,
    predictions: pd.DataFrame,
    output_path: Path,
    expert: str,
    role: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Sign router evidence without conflating it with a B1/B2 fold result.

    Callers provide actual A5/C2F evaluation rows.  The strict schema keeps the
    inner-OOF and outer-dev provenance needed by the nested router.
    """
    if expert not in {"a5", "c2f"} or role not in {"inner_oof", "outer_dev"}:
        raise ValueError("router expert must be a5/c2f with inner_oof/outer_dev role")
    required = {
        "sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s",
        "outer_fold", "shared_event_count_log1p", "shared_event_rate_log1p",
        "a5_flow_magnitude", "c2f_flow_magnitude", "a5_margin", "c2f_margin",
        "a5_log_variance", "c2f_log_variance",
    }
    if role == "inner_oof":
        required.add("inner_fold")
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"router expert prediction rows lack required fields: {missing}")
    if predictions["sample_token"].astype(str).duplicated().any():
        raise ValueError("router expert prediction sample tokens must be unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, lineterminator="\n")
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_router_expert_prediction_v1",
        "expert": expert,
        "role": role,
        "rows": len(predictions),
        "predictions": {"path": output_path.name, "sha256": _sha(output_path)},
        "protocol": dict(protocol),
    }
    sign_artifact(payload)
    _atomic_json(output_path.with_suffix(".summary.json"), payload)
    return payload


__all__.append("export_router_expert_predictions")

"""Production V8 temporal trainer backed by the audited A5 training loop.

The cache is V8-native; only the *training-only* adapter translates its raster and
geometry fields to the established causal-scale batch contract.  This deliberately
does not route V8 rows through the historical V4 data loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.data.dinov3_relational_teacher_cache import DINOv3RelationalTeacherDataset
from e_jepa_ttc.data.scientific_recovery_v8_adapter import V8ToObjectEventV4Dataset
from e_jepa_ttc.data.scientific_recovery_v8_cache import (
    ScientificRecoveryV8CacheDataset,
    collate_scientific_recovery_v8,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8 import validate_oof_frame
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


def assert_v8_dino_teacher_matches_source_rows(
    dino: Mapping[str, Any],
    *,
    expected_source_train_rows: int,
    manifest_path: Path,
) -> None:
    """Refuse a DINO universe that cannot cover the frozen V8 train rows."""

    if _sha(manifest_path) != str(dino.get("manifest_sha256")):
        raise ValueError("DINO teacher manifest file hash differs from protocol")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("DINO teacher manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("DINO teacher manifest must be a mapping")
    row_count = int(manifest.get("scope", {}).get("row_count", -1))
    expected = int(expected_source_train_rows)
    if row_count != expected:
        raise ValueError(
            "DINO teacher row_count differs from expected_source_train_rows "
            f"({row_count} != {expected})"
        )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _artifact_hash_from_ref(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("artifact_sha256", "sha256"):
            raw = value.get(key)
            if isinstance(raw, str) and len(raw) == 64:
                return raw
    if isinstance(value, str) and len(value) == 64:
        return value
    return None


def _read_signed_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"signed artifact is invalid JSON: {path}") from error
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"signed artifact hash is invalid: {path}")
    return payload


def export_v8_point_predictions(
    predictions: Sequence[object],
    log_variances: Sequence[object],
    known_mask: Sequence[object] | None,
) -> tuple[list[float], list[float], list[bool], list[str]]:
    """Record A5 unknown-support NaNs instead of aborting a selectable checkpoint.

    Causal-scale eval writes NaN when ``known_mask`` is false and counts those
    rows in ``failure_rate_pct``. The OOF CSV contract already requires
    ``failure_reason`` for non-finite rows; the TTC candidate gate then decides
    whether ``finite_fraction == 1``.
    """

    if len(log_variances) != len(predictions):
        raise ValueError("ttc_log_variance length differs from predictions")
    known_list = None if known_mask is None else list(known_mask)
    if known_list is not None and len(known_list) != len(predictions):
        raise ValueError("known_mask length differs from predictions")
    exported_pred: list[float] = []
    exported_var: list[float] = []
    finite_flags: list[bool] = []
    reasons: list[str] = []
    for index, raw in enumerate(predictions):
        value = float(raw)
        is_finite = math.isfinite(value)
        known = True if known_list is None else bool(known_list[index])
        if is_finite:
            variance = float(log_variances[index])
            if not math.isfinite(variance):
                raise ValueError("finite point TTC lacks finite log-variance")
            exported_pred.append(value)
            exported_var.append(variance)
            finite_flags.append(True)
            reasons.append("")
            continue
        exported_pred.append(float("nan"))
        exported_var.append(float("nan"))
        finite_flags.append(False)
        reasons.append("no_known_causal_support" if not known else "non_finite_point_ttc")
    return exported_pred, exported_var, finite_flags, reasons


def _binding_record(path: Path, *, file_sha256: str, artifact_sha256: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "sha256": file_sha256,
        "artifact_sha256": artifact_sha256,
    }


def resolve_v8_cache_protocol_binding(
    cache_manifest: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    """Bind fold results to the signed protocol via the cache file-hash fields.

    Production caches store ``protocol_path`` and ``protocol_sha256`` (file hash).
    They do not store a nested ``protocol.artifact_sha256`` object. Fold summaries
    still need the signed protocol artifact hash after the file bytes are verified.
    """

    nested_hash = _artifact_hash_from_ref(cache_manifest.get("protocol"))
    path_raw = cache_manifest.get("protocol_path")
    expected_file = cache_manifest.get("protocol_sha256")
    if isinstance(path_raw, str) and path_raw:
        path = Path(path_raw)
        if not path.is_file():
            raise ValueError(f"signed V8 cache protocol path is missing: {path}")
        if not isinstance(expected_file, str) or len(expected_file) != 64:
            raise ValueError("signed V8 cache lacks protocol file SHA-256")
        file_sha256 = _sha(path)
        if file_sha256 != expected_file:
            raise ValueError("protocol file hash differs from signed V8 cache binding")
        payload = _read_signed_mapping(path)
        artifact = payload.get("artifact_sha256")
        if not isinstance(artifact, str) or len(artifact) != 64:
            raise ValueError("signed V8 protocol lacks artifact_sha256")
        if nested_hash is not None and nested_hash != artifact:
            raise ValueError("cache protocol artifact hash disagrees with protocol file")
        return artifact, _binding_record(path, file_sha256=file_sha256, artifact_sha256=artifact)
    if nested_hash is not None:
        nested = cache_manifest.get("protocol")
        path_value = ""
        file_value = nested_hash
        if isinstance(nested, Mapping):
            path_value = str(nested.get("path", ""))
            raw_file = nested.get("sha256")
            if isinstance(raw_file, str) and len(raw_file) == 64:
                file_value = raw_file
        return nested_hash, {
            "path": path_value,
            "sha256": file_value,
            "artifact_sha256": nested_hash,
        }
    raise ValueError("signed V8 cache lacks protocol artifact bindings")


def _frozen_manifest_lists_fold_config(
    payload: Mapping[str, Any], *, config_path: Path, config_sha256: str
) -> bool:
    listed = payload.get("enabled_seed7_configs")
    if not isinstance(listed, Mapping):
        return False
    name = config_path.name
    for item in listed.values():
        if not isinstance(item, Mapping):
            continue
        if item.get("sha256") != config_sha256:
            continue
        raw = item.get("path")
        if isinstance(raw, str) and Path(raw).name == name:
            return True
    return False


def resolve_v8_frozen_manifest_binding(
    cache_manifest: Mapping[str, Any], *, config_path: Path
) -> tuple[str, dict[str, str]]:
    """Bind fold results to the freeze file that lists this yaml.

    Temporal caches are protocol-identity-bound and never wrote a nested
    ``frozen_manifest`` object. The training-time freeze lives beside the fold yaml.
    """

    path = config_path.parent / "frozen_manifest.json"
    nested_hash = _artifact_hash_from_ref(cache_manifest.get("frozen_manifest"))
    if not path.is_file():
        if nested_hash is not None:
            nested = cache_manifest.get("frozen_manifest")
            path_value = ""
            file_value = nested_hash
            if isinstance(nested, Mapping):
                path_value = str(nested.get("path", ""))
                raw_file = nested.get("sha256")
                if isinstance(raw_file, str) and len(raw_file) == 64:
                    file_value = raw_file
            return nested_hash, {
                "path": path_value,
                "sha256": file_value,
                "artifact_sha256": nested_hash,
            }
        raise ValueError("signed V8 cache lacks frozen-manifest artifact bindings")
    payload = _read_signed_mapping(path)
    artifact = payload.get("artifact_sha256")
    if not isinstance(artifact, str) or len(artifact) != 64:
        raise ValueError("signed V8 frozen manifest lacks artifact_sha256")
    if not _frozen_manifest_lists_fold_config(
        payload, config_path=config_path, config_sha256=_sha(config_path)
    ):
        raise ValueError("frozen manifest does not list this V8 fold config")
    if nested_hash is not None and nested_hash != artifact:
        raise ValueError("cache frozen-manifest artifact hash disagrees with freeze file")
    return artifact, _binding_record(path, file_sha256=_sha(path), artifact_sha256=artifact)


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
    steps = int(data.get("steps", cache.shape[0] if len(cache.shape) > 0 else 3))
    train = V8ToObjectEventV4Dataset(cache, outer_fold=fold, split="train", steps=steps)
    dev = V8ToObjectEventV4Dataset(cache, outer_fold=fold, split="dev", steps=steps)
    dev_sample_weights = {
        str(dev[index]["sample_token"]): float(dev[index]["sample_weight"])
        for index in range(len(dev))
    }
    dev_temporal_diagnostics: dict[str, tuple[float, float, float]] = {}
    for cache_index in dev.indices:
        row = cache[cache_index]
        token = str(row["sample_token"])
        diagnostics = row.get("endpoint_diagnostics")
        if not isinstance(diagnostics, list) or not diagnostics:
            dev_temporal_diagnostics[token] = (0.0, 0.0, 0.0)
            continue
        latest = diagnostics[-1]
        event_count = float(latest.get("event_count", latest.get("state_event_count", 0.0)))
        start_us = float(latest.get("support_start_us", row["endpoint_us"][-2]))
        end_us = float(latest.get("support_end_us", row["endpoint_us"][-1]))
        support_ms = max(0.0, (end_us - start_us) / 1000.0)
        event_rate = event_count / max(support_ms / 1000.0, 1.0e-9)
        dev_temporal_diagnostics[token] = (event_count, event_rate, support_ms)
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
        assert_v8_dino_teacher_matches_source_rows(
            dino,
            expected_source_train_rows=int(data["expected_source_train_rows"]),
            manifest_path=manifest_path,
        )
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
    overrides = config.get("model_overrides", {})
    if overrides:
        if not isinstance(overrides, Mapping):
            raise ValueError("model_overrides must be a mapping")
        allowed_overrides = {
            "temporal_channel_gate_enabled",
            "temporal_channel_gate_patch_grid",
            "temporal_channel_gate_hidden_dim",
        }
        unknown = sorted(set(overrides) - allowed_overrides)
        if unknown:
            raise ValueError(f"unsupported V8 model overrides: {unknown}")
        model_cfg = CausalScaleTTCConfig(**{**asdict(model_cfg), **dict(overrides)})
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
    checkpoint_sha256 = _sha(checkpoint)
    config_sha256 = _sha(config_path)
    if fixture_smoke:
        protocol_sha256 = _artifact_hash_from_ref(cache_manifest.get("protocol"))
        frozen_manifest_sha256 = _artifact_hash_from_ref(cache_manifest.get("frozen_manifest"))
        protocol_ref = cache_manifest.get("protocol")
        frozen_ref = cache_manifest.get("frozen_manifest")
    else:
        protocol_sha256, protocol_ref = resolve_v8_cache_protocol_binding(cache_manifest)
        frozen_manifest_sha256, frozen_ref = resolve_v8_frozen_manifest_binding(
            cache_manifest, config_path=config_path
        )
    predictions_path = output_dir / "dev_predictions.csv"
    tokens = [str(value) for value in validation["sample_tokens"]]
    ttc_log_variance = validation.get("ttc_log_variance", [0.0] * len(tokens))
    known_mask = validation.get("known_mask")
    if not isinstance(known_mask, list):
        known_mask = None
    prediction_ttc, prediction_log_variance, finite_flags, failure_reasons = (
        export_v8_point_predictions(
            validation["prediction_ttc_s"],
            ttc_log_variance,
            known_mask,
        )
    )
    temporal_diag = [dev_temporal_diagnostics[token] for token in tokens]
    frame = pd.DataFrame(
        {
            "token_id": tokens,
            "sequence_id": validation["sequence_ids"],
            "track_id": validation["track_ids"],
            "outer_fold": fold,
            "seed": train_cfg.seed,
            "target_ttc": validation["target_ttc_s"],
            "sample_weight": [dev_sample_weights[token] for token in tokens],
            "prediction_ttc": prediction_ttc,
            "prediction_log_variance": prediction_log_variance,
            "finite": finite_flags,
            "failure_reason": failure_reasons,
            "event_count": [value[0] for value in temporal_diag],
            "event_rate": [value[1] for value in temporal_diag],
            "support_ms": [value[2] for value in temporal_diag],
            "model_name": str(exp["arm"]),
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    validate_oof_frame(frame, label="V8 fold dev predictions")
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
        "status": "fixture_smoke_completed" if fixture_smoke else "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_name": exp["name"],
        "arm": exp["arm"],
        "outer_fold": fold,
        "fold": fold,
        "seed": train_cfg.seed,
        "fixture": bool(fixture_smoke),
        "fixture_smoke": bool(fixture_smoke),
        "base_git_commit": cache_manifest.get("git_base_commit"),
        "implementation_git_commit": _git_commit(),
        "git_commit": _git_commit(),
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "config": {"path": config_path.as_posix(), "sha256": config_sha256},
        "model_config": {"path": model_path.as_posix(), "sha256": _sha(model_path)},
        "protocol": protocol_ref,
        "frozen_manifest": frozen_ref,
        "cache": {
            "path": cache_path.as_posix(),
            "sha256": _sha(cache_path),
            "artifact_sha256": cache_manifest.get("artifact_sha256"),
        },
        "checkpoint": {"path": checkpoint.name, "sha256": checkpoint_sha256},
        "dev_predictions": {
            "path": predictions_path.name,
            "sha256": _sha(predictions_path),
            "rows": len(frame),
        },
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


__all__ = [
    "assert_v8_dino_teacher_matches_source_rows",
    "export_v8_point_predictions",
    "resolve_v8_cache_protocol_binding",
    "resolve_v8_frozen_manifest_binding",
    "run_v8_temporal_training",
]


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
        "sample_token",
        "sequence_id",
        "track_id",
        "target_ttc_s",
        "prediction_ttc_s",
        "outer_fold",
        "shared_event_count_log1p",
        "shared_event_rate_log1p",
        "a5_flow_magnitude",
        "c2f_flow_magnitude",
        "a5_margin",
        "c2f_margin",
        "a5_log_variance",
        "c2f_log_variance",
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

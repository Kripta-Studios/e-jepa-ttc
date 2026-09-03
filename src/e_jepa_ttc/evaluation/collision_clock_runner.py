"""Frozen outer-train/outer-dev execution pipeline for E-Clock X0."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.data.collision_clock_cache import (
    CacheMode,
    CollisionClockOuterDevBatch,
    CollisionClockOuterTrainSequence,
    CollisionClockTrain8192Cache,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    ROW_LEVEL_OOF_COLUMNS,
    load_signed_json,
    module_topology_sha256,
    require_reference_family,
    tensor_state_sha256,
    validate_protocol_reference_binding,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (
    load_causal_scale_replay_checkpoint,
)
from e_jepa_ttc.losses.collision_clock import uniform_benchmark_phase_loss
from e_jepa_ttc.models.collision_clock_features import (
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
)
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase
from e_jepa_ttc.models.collision_clock_ttc import (
    CollisionClockConfig,
    CollisionClockTTCOutput,
    X0A5Replay,
    X0HeightBypassDirectPhase,
    X0PairDirectPhase,
)
from e_jepa_ttc.training.collision_clock_eap import (
    CollisionClockScientificIdentity,
    CollisionClockTrainingConfig,
    require_frozen_checkpoint,
    train_collision_clock_updates,
)

_FROZEN_CAPABILITY = object()


@dataclass(frozen=True)
class FrozenCollisionClockCheckpoint:
    """Evaluation capability obtainable only after checkpoint verification."""

    model: nn.Module
    checkpoint_path: Path
    checkpoint_file_sha256: str
    checkpoint_manifest_sha256: str
    external_official_a5: bool
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _FROZEN_CAPABILITY:
            raise ValueError("frozen checkpoint capability cannot be self-declared")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_signed_state(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    signed = sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(signed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return signed


def _load_signed_state(path: Path, *, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not verify_artifact_hash(payload)
        or payload.get("artifact_type") != artifact_type
    ):
        raise ValueError(f"state signature/type mismatch: {path}")
    return payload


def _direct_model(config: dict[str, Any]) -> X0HeightBypassDirectPhase:
    values = dict(config["model"])
    values["feature_source"] = config["feature_source"]
    values["motion_feature_mode"] = config["motion_feature_mode"]
    clock = CollisionClockConfig(**values)
    encoder = HeightBypassEndpointEncoder(
        HeightBypassEncoderConfig(
            in_channels=clock.in_channels,
            hidden_dim=clock.encoder_hidden_dim,
            token_dim=clock.encoder_token_dim,
            residual_depth=clock.residual_depth,
            dropout=clock.dropout,
        )
    )
    return X0HeightBypassDirectPhase(encoder, clock)


def _official_checkpoint(
    *, fold: int, reference: dict[str, Any], source_root: Path
) -> tuple[Path, str]:
    family = require_reference_family(reference, "official_a5_oof")
    records = family.get("official_fold_checkpoints")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("official A5 requires three signed fold checkpoints")
    matches = [record for record in records if int(record["outer_fold"]) == fold]
    if len(matches) != 1:
        raise ValueError("official A5 checkpoint fold identity mismatch")
    record = matches[0]
    path = source_root / Path(str(record["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or compute_file_hash(str(path)) != record["file_sha256"]
    ):
        raise ValueError("official A5 checkpoint physical identity mismatch")
    return path, str(record["file_sha256"])


def dry_run_dag(
    *,
    config: dict[str, Any],
    config_path: Path,
    protocol: dict[str, Any],
    reference: dict[str, Any],
    cache_root: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Describe every fold/path without opening shards or creating outputs."""

    arm_id = str(config["arm_id"])
    folds = []
    for fold in (0, 1, 2):
        fold_root = output_root / f"fold-{fold}"
        node: dict[str, Any] = {
            "outer_fold": fold,
            "steps": [
                "verify_canonical_cache",
                "construct_outer_train_fold_not_equal_k",
                "construct_outer_dev_fold_equal_k",
                "initialize_exact_arm",
                "train_outer_train_only_fixed_updates",
                "save_last_update_fixed_budget",
                "freeze_checkpoint",
                "evaluate_outer_dev_once",
                "export_raw_phase_inverse_ttc",
                "write_signed_fold_summary",
            ],
            "checkpoint": str(fold_root / "last_update.pt"),
            "oof_predictions": str(fold_root / "oof_predictions.csv"),
            "fold_summary": str(fold_root / "fold_summary.json"),
        }
        if arm_id in {"X0-A5-REPLAY", "X0-PAIR-U"}:
            family = require_reference_family(reference, "official_a5_oof")
            record = family["official_fold_checkpoints"][fold]
            node["official_a5_checkpoint"] = str(source_root / Path(record["path"]))
            node["official_a5_checkpoint_sha256"] = record["file_sha256"]
        if arm_id == "X0-A5-REPLAY":
            node["steps"] = [
                "verify_canonical_cache",
                "construct_outer_dev_fold_equal_k",
                "verify_official_a5_fold_checkpoint",
                "evaluate_outer_dev_once",
                "export_raw_phase_inverse_ttc",
                "write_signed_fold_summary",
            ]
        folds.append(node)
    return sign_artifact(
        {
            "artifact_type": "eclock_x0_dry_run_dag_v2",
            "evidence_class": "dry_run",
            "scientific_result": False,
            "arm_id": arm_id,
            "config_path": str(config_path),
            "config_sha256": compute_file_hash(str(config_path)),
            "protocol_sha256": protocol["artifact_sha256"],
            "reference_sha256": reference["artifact_sha256"],
            "cache_root_observed": str(cache_root.absolute()),
            "cache_manifest_path": str(cache_root / "manifest.json"),
            "opens_cache_shards": False,
            "creates_scientific_results": False,
            "checkpoint_policy": "last_update_fixed_budget",
            "outer_dev_during_training": False,
            "folds": folds,
        }
    )


def _freeze_local(model: nn.Module, checkpoint_path: Path) -> FrozenCollisionClockCheckpoint:
    manifest = require_frozen_checkpoint(checkpoint_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenCollisionClockCheckpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=str(manifest["checkpoint_file_sha256"]),
        checkpoint_manifest_sha256=str(manifest["artifact_sha256"]),
        external_official_a5=False,
        _capability=_FROZEN_CAPABILITY,
    )


def _freeze_official_a5(
    model: nn.Module, checkpoint_path: Path, checkpoint_sha256: str
) -> FrozenCollisionClockCheckpoint:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    binding = sign_artifact(
        {
            "artifact_type": "eclock_x0_official_a5_checkpoint_binding_v2",
            "reference_family": "official_a5_oof",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_file_sha256": checkpoint_sha256,
            "frozen": True,
        }
    )
    return FrozenCollisionClockCheckpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=checkpoint_sha256,
        checkpoint_manifest_sha256=binding["artifact_sha256"],
        external_official_a5=True,
        _capability=_FROZEN_CAPABILITY,
    )


def _prediction_coordinates(
    output: object,
    *,
    delta_t_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(output, CollisionClockTTCOutput):
        phase = output.benchmark_phase_mean.detach().cpu().numpy().astype(np.float64)
        inverse = output.inverse_ttc_mean.detach().cpu().numpy().astype(np.float64)
        raw = output.predicted_ttc_raw.detach().cpu().numpy().astype(np.float64)
        clipped = output.predicted_ttc_clipped.detach().cpu().numpy().astype(np.float64)
    else:
        inverse_tensor = getattr(output, "inverse_ttc_mean", None)
        if inverse_tensor is None or not hasattr(inverse_tensor, "detach"):
            raise TypeError("legacy model output lacks a raw inverse-TTC tensor")
        inverse = inverse_tensor.detach().cpu().numpy().astype(np.float64)
        raw = np.reciprocal(inverse)
        phase = -np.log1p(-delta_t_s * inverse)
        clipped = np.clip(raw, -60.0, 60.0)
    if not all(np.isfinite(values).all() for values in (phase, inverse, raw, clipped)):
        raise ValueError("model prediction cannot be exported as finite raw coordinates")
    return phase, inverse, raw, clipped


def evaluate_outer_dev_once(
    frozen: FrozenCollisionClockCheckpoint,
    batches: Iterable[CollisionClockOuterDevBatch],
    *,
    arm_id: str,
    seed: int,
    config_sha256: str,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate a frozen checkpoint on typed outer-dev batches exactly once."""

    if not isinstance(frozen, FrozenCollisionClockCheckpoint):
        raise TypeError("outer-dev evaluator requires a frozen checkpoint capability")
    if compute_file_hash(str(frozen.checkpoint_path)) != frozen.checkpoint_file_sha256:
        raise ValueError("frozen checkpoint bytes changed before outer-dev evaluation")
    if not frozen.external_official_a5:
        require_frozen_checkpoint(frozen.checkpoint_path)
    rows: list[dict[str, Any]] = []
    delta_metric = float(protocol["metric"]["metric_delta_t_s"])
    model_device = next(frozen.model.parameters()).device
    with torch.no_grad():  # pyright: ignore[reportAttributeAccessIssue]
        for batch in batches:
            if type(batch) is not CollisionClockOuterDevBatch:
                raise TypeError(
                    "outer-dev evaluator accepts CollisionClockOuterDevBatch values only"
                )
            output = frozen.model(
                batch.inputs.to(model_device),
                batch.delta_t_s.to(model_device),
            )
            phase, inverse, raw, clipped = _prediction_coordinates(output, delta_t_s=delta_metric)
            target = batch.target_ttc_seconds.detach().cpu().numpy().astype(np.float64)
            target_phase = -np.log1p(-delta_metric / target)
            mid = 1.0e4 * np.abs(target_phase - phase)
            failure = ~np.isfinite(raw) | (np.abs(raw) < 0.1)
            saturated = np.abs(raw) > 60.0
            weights = batch.sample_weights.detach().cpu().numpy().astype(np.float64)
            for index, token in enumerate(batch.sample_tokens):
                rows.append(
                    {
                        "sample_token": token,
                        "sequence_id": batch.sequence_ids[index],
                        "track_id": batch.track_ids[index],
                        "outer_fold": batch.outer_fold,
                        "target_ttc_s": target[index],
                        "target_benchmark_phase": target_phase[index],
                        "predicted_benchmark_phase": phase[index],
                        "predicted_inverse_ttc_raw": inverse[index],
                        "predicted_ttc_raw": raw[index],
                        "predicted_ttc_clipped": clipped[index],
                        "is_clip_saturated": bool(saturated[index]),
                        "scientific_mid_per_row": mid[index],
                        "scientific_failure": bool(failure[index]),
                        "sample_weight": weights[index],
                        "arm_id": arm_id,
                        "seed": seed,
                        "checkpoint_sha256": frozen.checkpoint_file_sha256,
                        "config_sha256": config_sha256,
                        "protocol_sha256": protocol["artifact_sha256"],
                        "cache_manifest_sha256": protocol["cache_binding"]["file_sha256"],
                        "split_manifest_sha256": protocol["split_binding"]["file_sha256"],
                    }
                )
    if not rows:
        raise ValueError("outer-dev evaluation received no rows")
    result = pd.DataFrame(rows).loc[:, ROW_LEVEL_OOF_COLUMNS]
    if not np.isfinite(
        result[
            [
                "target_benchmark_phase",
                "predicted_benchmark_phase",
                "predicted_inverse_ttc_raw",
                "predicted_ttc_raw",
                "predicted_ttc_clipped",
                "scientific_mid_per_row",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("outer-dev export contains a non-finite scientific coordinate")
    return result


def run_outer_folds(
    *,
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reference: dict[str, Any],
    cache_root: Path,
    source_root: Path,
    output_root: Path,
    device: torch.device,
    cache_mode: CacheMode = "auto",
    resume_campaign: bool = False,
    resume_checkpoint_every: int = 100,
    milestone_updates: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 6840),
    rich_log_every: int = 25,
) -> list[dict[str, Any]]:
    """Execute all frozen folds; caller must separately authorize this function."""

    if config["arm_id"] == "X0-DYN-W" or config.get("execution_authorized") is not True:
        raise PermissionError("arm is not executable")
    dirty = bool(_git(repo, "status", "--porcelain=v1", "--untracked-files=no"))
    if dirty:
        raise ValueError("scientific OOF requires a clean versioned worktree")
    commit = _git(repo, "rev-parse", "HEAD")
    config_sha = compute_file_hash(str(config_path))
    arm_manifest_path = output_root / "campaign_manifest.json"
    arm_identity = {
        "artifact_type": "eclock_x0_arm_campaign_manifest_v1",
        "git_commit": commit,
        "arm_id": config["arm_id"],
        "seed": 7,
        "folds": [0, 1, 2],
        "config_sha256": config_sha,
        "protocol_sha256": protocol["artifact_sha256"],
        "reference_sha256": reference["artifact_sha256"],
        "cache_manifest_sha256": protocol["cache_binding"]["file_sha256"],
        "cache_mode_requested": cache_mode,
        "checkpoint_policy": "last_update_fixed_budget",
    }
    if output_root.exists():
        if not resume_campaign:
            raise FileExistsError(
                "run output root already exists; use explicit resume only after "
                "identity verification"
            )
        if not arm_manifest_path.is_file():
            raise ValueError("existing campaign root lacks a signed arm manifest")
        observed = _load_signed_state(
            arm_manifest_path, artifact_type="eclock_x0_arm_campaign_manifest_v1"
        )
        if {k: v for k, v in observed.items() if k != "artifact_sha256"} != arm_identity:
            raise ValueError("resume campaign identity mismatch")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _write_signed_state(arm_manifest_path, arm_identity)
    adapter = CollisionClockTrain8192Cache(cache_root, protocol, cache_mode=cache_mode)
    adapter.verify_and_index()
    summaries = []
    for fold in (0, 1, 2):
        train_view, dev_view = adapter.outer_views(fold)
        if adapter.requested_cache_mode == "auto":
            adapter.select_mode_for_train_view(train_view)
        fold_root = output_root / f"fold-{fold}"
        fold_root.mkdir(parents=True, exist_ok=resume_campaign)
        state_path = fold_root / "fold_state.json"
        summary_path = fold_root / "fold_summary.json"
        if summary_path.is_file():
            if not resume_campaign:
                raise ValueError("completed fold exists without resume authorization")
            summary = _load_signed_state(summary_path, artifact_type="eclock_x0_fold_summary_v2")
            if (
                summary.get("arm_id") != config["arm_id"]
                or summary.get("outer_fold") != fold
                or summary.get("seed") != 7
                or summary.get("outer_dev_evaluations") != 1
                or summary.get("git_commit") != commit
                or summary.get("config_sha256") != config_sha
                or summary.get("protocol_sha256") != protocol["artifact_sha256"]
                or summary.get("reference_sha256") != reference["artifact_sha256"]
            ):
                raise ValueError("completed fold summary identity mismatch")
            existing_oof = Path(str(summary.get("oof_path", "")))
            if not existing_oof.is_file() or compute_file_hash(str(existing_oof)) != summary.get(
                "oof_file_sha256"
            ):
                raise ValueError("completed fold OOF physical identity mismatch")
            existing_checkpoint = Path(str(summary.get("checkpoint_path", "")))
            if compute_file_hash(str(existing_checkpoint)) != summary.get("checkpoint_file_sha256"):
                raise ValueError("completed fold checkpoint physical identity mismatch")
            summaries.append(summary)
            continue
        if state_path.is_file():
            state = _load_signed_state(state_path, artifact_type="eclock_x0_fold_state_v1")
            if state.get("git_commit") != commit or state.get("arm_id") != config["arm_id"]:
                raise ValueError("fold resume state identity mismatch")
            if state.get("status") in {"evaluation_in_progress", "evaluation_complete"}:
                raise ValueError(
                    "outer-dev may have been opened; fail closed instead of reevaluating"
                )
        else:
            state = _write_signed_state(
                state_path,
                {
                    "artifact_type": "eclock_x0_fold_state_v1",
                    "status": "planned",
                    "git_commit": commit,
                    "arm_id": config["arm_id"],
                    "outer_fold": fold,
                    "seed": 7,
                    "outer_dev_opened": False,
                },
            )
        fold_started = time.time()
        official_path: Path | None = None
        official_sha: str | None = None
        if config["arm_id"] in {"X0-A5-REPLAY", "X0-PAIR-U"}:
            official_path, official_sha = _official_checkpoint(
                fold=fold, reference=reference, source_root=source_root
            )
            source_a5 = load_causal_scale_replay_checkpoint(official_path, device=device)
        torch.manual_seed(7)  # pyright: ignore[reportAttributeAccessIssue]
        if config["arm_id"] == "X0-A5-REPLAY":
            model: nn.Module = X0A5Replay(source_a5).to(device)
            assert official_path is not None and official_sha is not None
            frozen = _freeze_official_a5(model, official_path, official_sha)
        elif config["arm_id"] == "X0-PAIR-U":
            values = dict(config["model"])
            values["feature_source"] = config["feature_source"]
            values["motion_feature_mode"] = config["motion_feature_mode"]
            model = X0PairDirectPhase(source_a5, CollisionClockConfig(**values)).to(device)
        else:
            model = _direct_model(config).to(device)
        checkpoint_path = fold_root / "resume_latest.pt"
        training_result = None
        if config["arm_id"] != "X0-A5-REPLAY":
            training = config["training"]
            identity = CollisionClockScientificIdentity(
                git_commit_observed=commit,
                git_dirty_observed=False,
                arm_id=config["arm_id"],
                scientific_role=config["scientific_role"],
                reference_family=("official_a5_oof" if config["arm_id"] == "X0-PAIR-U" else None),
                seed=7,
                outer_fold=fold,
                motion_feature_mode=config["motion_feature_mode"],
                model_class=model.__class__.__name__,
                model_topology_sha256=module_topology_sha256(model),
                initialization_sha256=tensor_state_sha256(model),
                config_path=str(config_path),
                config_sha256=config_sha,
                protocol_path=config["protocol_path"],
                protocol_sha256=protocol["artifact_sha256"],
                reference_path=config["reference_path"],
                reference_sha256=reference["artifact_sha256"],
                split_manifest_path=protocol["split_binding"]["path"],
                split_manifest_sha256=protocol["split_binding"]["file_sha256"],
                cache_manifest_path=protocol["cache_binding"]["path"],
                cache_manifest_sha256=protocol["cache_binding"]["file_sha256"],
                ordered_token_identity_sha256=protocol["canonical_hashes"]["token_identity_sha256"],
                target_sha256=protocol["canonical_hashes"]["target_sha256"],
                fold_assignment_sha256=protocol["canonical_hashes"]["fold_assignment_sha256"],
                sample_weight_sha256=protocol["canonical_hashes"]["sample_weight_sha256"],
                train_token_subset_sha256=train_view.ordered_token_identity_sha256,
                dev_token_subset_sha256=dev_view.ordered_token_identity_sha256,
                optimizer_config={
                    "name": "AdamW",
                    "learning_rate": training["learning_rate"],
                    "weight_decay": training["weight_decay"],
                },
                scheduler_config={"name": training["scheduler"]},
                precision_mode=training["precision_mode"],
                update_budget=training["update_budget"],
                checkpoint_policy=training["checkpoint_policy"],
            )
            schedule = CollisionClockOuterTrainSequence(
                adapter, train_view, batch_size=training["batch_size"]
            )
            progress_path = fold_root / "progress.jsonl"
            if resume_campaign and not checkpoint_path.is_file() and progress_path.is_file():
                abandoned = fold_root / "progress.aborted-before-first-checkpoint.jsonl"
                if abandoned.exists():
                    raise ValueError("pre-checkpoint interruption evidence already exists")
                progress_path.replace(abandoned)
            adapter.stage_view(train_view)
            _write_signed_state(
                state_path,
                {
                    "artifact_type": "eclock_x0_fold_state_v1",
                    "status": "training_in_progress",
                    "git_commit": commit,
                    "arm_id": config["arm_id"],
                    "outer_fold": fold,
                    "seed": 7,
                    "outer_dev_opened": False,
                    "cache_mode": adapter.cache_mode,
                },
            )
            training_result = train_collision_clock_updates(
                model,
                schedule,
                config=CollisionClockTrainingConfig(
                    arm_id=config["arm_id"],
                    update_budget=training["update_budget"],
                    learning_rate=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                    precision_mode=training["precision_mode"],
                    checkpoint_policy=training["checkpoint_policy"],
                ),
                scientific_identity=identity,
                checkpoint_path=checkpoint_path,
                resume=checkpoint_path.is_file(),
                checkpoint_every=resume_checkpoint_every,
                milestone_updates=milestone_updates,
                progress_path=progress_path,
                rich_log_every=rich_log_every,
            )
            adapter.release_staged_view()
            frozen = _freeze_local(model, training_result.checkpoint_path)
            _write_signed_state(
                state_path,
                {
                    "artifact_type": "eclock_x0_fold_state_v1",
                    "status": "training_complete_checkpoint_frozen",
                    "git_commit": commit,
                    "arm_id": config["arm_id"],
                    "outer_fold": fold,
                    "seed": 7,
                    "outer_dev_opened": False,
                    "checkpoint_path": str(training_result.checkpoint_path),
                    "checkpoint_sha256": frozen.checkpoint_file_sha256,
                },
            )
        _write_signed_state(
            state_path,
            {
                "artifact_type": "eclock_x0_fold_state_v1",
                "status": "evaluation_in_progress",
                "git_commit": commit,
                "arm_id": config["arm_id"],
                "outer_fold": fold,
                "seed": 7,
                "outer_dev_opened": True,
                "outer_dev_evaluation_count": 1,
            },
        )
        adapter.stage_view(dev_view)
        dev_batches = adapter.iter_outer_dev_batches(
            dev_view,
            batch_size=config.get("training", {}).get("batch_size", 32),
        )
        predictions = evaluate_outer_dev_once(
            frozen,
            dev_batches,
            arm_id=config["arm_id"],
            seed=7,
            config_sha256=config_sha,
            protocol=protocol,
        )
        adapter.release_staged_view()
        oof_path = fold_root / "oof_predictions.csv"
        predictions.to_csv(oof_path, index=False)
        summary = sign_artifact(
            {
                "artifact_type": "eclock_x0_fold_summary_v2",
                "status": "completed_after_frozen_checkpoint",
                "arm_id": config["arm_id"],
                "outer_fold": fold,
                "seed": 7,
                "checkpoint_policy": "last_update_fixed_budget",
                "checkpoint_path": str(frozen.checkpoint_path),
                "checkpoint_file_sha256": frozen.checkpoint_file_sha256,
                "checkpoint_bytes": frozen.checkpoint_path.stat().st_size,
                "checkpoint_manifest_sha256": frozen.checkpoint_manifest_sha256,
                "external_official_a5": frozen.external_official_a5,
                "oof_path": str(oof_path),
                "oof_file_sha256": compute_file_hash(str(oof_path)),
                "oof_bytes": oof_path.stat().st_size,
                "row_count": len(predictions),
                "outer_train_token_sha256": train_view.ordered_token_identity_sha256,
                "outer_dev_token_sha256": dev_view.ordered_token_identity_sha256,
                "outer_dev_evaluations": 1,
                "outer_dev_used_during_training": False,
                "outer_dev_used_for_selection": False,
                "git_commit": commit,
                "config_sha256": config_sha,
                "protocol_sha256": protocol["artifact_sha256"],
                "reference_sha256": reference["artifact_sha256"],
                "cache_mode": adapter.cache_mode,
                "cache_engineering": adapter.engineering_stats(),
                "progress_path": (
                    str(training_result.progress_path) if training_result is not None else None
                ),
                "updates_completed": (
                    training_result.completed_updates if training_result is not None else 0
                ),
                "fold_wall_seconds": time.time() - fold_started,
            }
        )
        _write_signed_state(
            state_path,
            {
                "artifact_type": "eclock_x0_fold_state_v1",
                "status": "evaluation_complete",
                "git_commit": commit,
                "arm_id": config["arm_id"],
                "outer_fold": fold,
                "seed": 7,
                "outer_dev_opened": True,
                "outer_dev_evaluation_count": 1,
                "fold_summary_sha256": summary["artifact_sha256"],
            },
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
    return summaries


def run_outer_train_smoke(
    *,
    repo: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reference: dict[str, Any],
    cache_root: Path,
    source_root: Path,
    outer_fold: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one explicitly authorized non-scientific outer-train update only."""

    arm_id = str(config["arm_id"])
    if arm_id not in {"X0-PAIR-U", "X0-BASE-U", "X0-DYN-U"}:
        raise PermissionError("real-data outer-train smoke is unavailable for this arm")
    if config.get("execution_authorized") is not True or outer_fold not in (0, 1, 2):
        raise PermissionError("outer-train smoke identity is not authorized")
    adapter = CollisionClockTrain8192Cache(cache_root, protocol)
    train_view, _dev_view = adapter.outer_views(outer_fold)
    torch.manual_seed(7)
    if arm_id == "X0-PAIR-U":
        official_path, _official_sha = _official_checkpoint(
            fold=outer_fold, reference=reference, source_root=source_root
        )
        source_a5 = load_causal_scale_replay_checkpoint(official_path, device=device)
        values = dict(config["model"])
        values["feature_source"] = config["feature_source"]
        values["motion_feature_mode"] = config["motion_feature_mode"]
        model: nn.Module = X0PairDirectPhase(source_a5, CollisionClockConfig(**values)).to(device)
    else:
        model = _direct_model(config).to(device)
    training = config["training"]
    schedule = CollisionClockOuterTrainSequence(
        adapter, train_view, batch_size=training["batch_size"]
    )
    batch = schedule[0]
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.inputs.to(device), batch.delta_t_s.to(device))
    target_phase, valid = ttc_to_benchmark_phase(
        batch.target_ttc_seconds.to(device, dtype=torch.float64), metric_delta_t_s=0.1
    )
    if not bool(valid.all()) or not bool(torch.isfinite(target_phase).all()):
        raise ValueError("outer-train smoke target lies outside the phase domain")
    phase = getattr(output, "benchmark_phase_mean", None)
    if phase is None:
        raise TypeError("outer-train smoke model lacks benchmark phase output")
    loss = uniform_benchmark_phase_loss(phase, target_phase.to(phase.dtype))
    if not bool(torch.isfinite(loss)):
        raise ValueError("outer-train smoke loss is non-finite")
    loss.backward()
    optimizer.step()
    return sign_artifact(
        {
            "artifact_type": "eclock_x0_outer_train_smoke_v2",
            "evidence_class": "smoke",
            "scientific_result": False,
            "arm_id": arm_id,
            "seed": 7,
            "outer_fold": outer_fold,
            "git_commit_observed": _git(repo, "rev-parse", "HEAD"),
            "git_dirty_observed": bool(
                _git(repo, "status", "--porcelain=v1", "--untracked-files=no")
            ),
            "cache_manifest_sha256": protocol["cache_binding"]["file_sha256"],
            "outer_train_token_sha256": train_view.ordered_token_identity_sha256,
            "outer_train_rows_consumed": len(batch.sample_tokens),
            "optimizer_updates": 1,
            "finite_loss": True,
            "outer_dev_opened": False,
            "outer_dev_evaluated": False,
            "creates_scientific_result": False,
        }
    )


def load_runner_contracts(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load config/protocol/reference with their strict v2 schemas."""

    from e_jepa_ttc.evaluation.collision_clock_config import load_x0_config

    config = load_x0_config(
        config_path,
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_config_v2.schema.json",
    )
    protocol_path = repo / config["protocol_path"]
    reference_path = repo / config["reference_path"]
    protocol = load_signed_json(
        protocol_path,
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_protocol_v2.schema.json",
    )
    reference = load_signed_json(
        reference_path,
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_reference_v2.schema.json",
    )
    validate_protocol_reference_binding(protocol, reference, protocol_path=protocol_path)
    return config, protocol, reference


__all__ = [
    "FrozenCollisionClockCheckpoint",
    "dry_run_dag",
    "evaluate_outer_dev_once",
    "load_runner_contracts",
    "run_outer_folds",
    "run_outer_train_smoke",
]

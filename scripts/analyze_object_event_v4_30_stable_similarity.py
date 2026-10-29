#!/usr/bin/env python3
# ruff: noqa: E402, I001
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
"""Train-cache-only v4.30 grouped-OOF orchestrator.

Development paths may be supplied or required by the CLI, but are not
stat/read/materialized unless a genuine full-mode OOF champion triggers
development. There is no eAP, EvTTC, RGB, or official-test input. Preflight is
run before output handling. OOF implementation callers must supply all three
fixed EMA teacher checkpoints and retain every seed; no best checkpoint or seed
selection exists in this phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
FULL_OUTPUT_ROOT = (ROOT / "artifacts" / "debug" / "object_event_v4_30_stable_similarity").resolve()
DIAGNOSTIC_OUTPUT_ROOT = (ROOT / "artifacts" / "debug" / "object_event_v4_30_diagnostic").resolve()
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from e_jepa_ttc.training.object_event_v4_30 import (
    choose_rank_winner,
    compute_oof_metrics,
    gate_constituents_and_median,
    oof_gates,
    promoted_champion,
    stabilization_gate,
)  # noqa: E402
from scripts.preflight_object_event_v4_30 import (
    canonical_protocol_sha256,
    parse_seed_paths,
    sha256,
    validate_checkpoints,
    validate_config,
)  # noqa: E402
from scripts.analyze_object_event_v4_24_orchestrator import _sequence_folds, _subset_split  # noqa: E402
from scripts.train_e_jepa_object_event_v4_6 import _materialize  # noqa: E402
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import _load_backbone  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble, _read_ensemble  # noqa: E402
from e_jepa_ttc.models.object_event_v4_30 import ObjectEventTTCV430, ObjectEventV430Config  # noqa: E402
from e_jepa_ttc.training.object_event_v4_30 import (
    ObjectEventV430LossConfig,
    object_event_v4_30_loss,
    posterior_kl,
)  # noqa: E402

PREDICTION_FIELDS: tuple[str, ...] = (
    "prediction",
    "log_eta",
    "posterior_variance",
    "unknown",
    "fit12_effective_support_mass",
    "kappa",
    "rotation_radians",
    "translation_magnitude",
    "fit_residual",
    "correlation_entropy",
    "correlation_confidence",
    "boundary_probability",
    "normal_flow_residual",
    "cycle_matrix_error",
    "cycle_translation_error",
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, PAPER_MID_WEIGHTS  # noqa: E402

PHYSICAL_PREDICTION_FIELDS: tuple[str, ...] = tuple(
    field for field in PREDICTION_FIELDS if field != "unknown"
)
CONTROL_NAMES: tuple[str, ...] = (
    "zero_event",
    "temporal_shuffle",
    "endpoint_swap",
)

TEACHER_CONSENSUS_CACHE_SCHEMA = "object_event_v4_30_teacher_consensus_v1"


@dataclass(frozen=True)
class TeacherConsensusCache:
    """CPU float32 consensus rows, indexed in the selected split's row order."""

    probabilities: Mapping[int, torch.Tensor]
    row_count: int
    schema: str
    consensus_cache_sha256: str
    consensus_config_sha256: str
    checkpoint_file_sha256: Mapping[str, str]
    teacher_backbone_forward_batches: int
    consensus_build_count: int
    elapsed_seconds: float

    def for_indices(
        self, indices: torch.Tensor | np.ndarray, device: torch.device
    ) -> dict[int, torch.Tensor]:
        """Return a device-local batch without re-running any frozen teacher."""
        index = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if len(index) == 0 or bool((index < 0).any()) or bool((index >= self.row_count).any()):
            raise IndexError("teacher consensus indices are outside the selected split")
        return {
            scale: probability.index_select(0, index).to(device, non_blocking=True)
            for scale, probability in self.probabilities.items()
        }


def _safe_output_root(output: Path) -> Path:
    target = output.resolve()
    if target != FULL_OUTPUT_ROOT:
        raise ValueError("--output-dir must be the exact v4.30 full-OOF artifact root")
    return target


def _prepare_output_target(target: Path, *, force: bool) -> None:
    """Recreate only an exact v4.30 output root, including a safely empty abort root."""
    resolved = target.resolve()
    if resolved not in {FULL_OUTPUT_ROOT, DIAGNOSTIC_OUTPUT_ROOT}:
        raise ValueError("v4.30 may create or remove only its exact full or diagnostic output root")
    if resolved.exists():
        if any(resolved.iterdir()):
            if not force:
                raise FileExistsError("v4.30 output exists; use --force only for its exact root")
            import shutil

            shutil.rmtree(resolved)
        else:
            resolved.rmdir()
    resolved.mkdir(parents=True, exist_ok=False)


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except Exception:
            return "unknown"

    return {"commit": run("git", "rev-parse", "HEAD"), "status": run("git", "status", "--short")}


def posterior_stability_metrics(
    student_probabilities: list[Mapping[int, np.ndarray]], offsets: Mapping[int, np.ndarray]
) -> dict[str, float]:
    """Pairwise JS/displacement stability over held-out student posteriors."""
    if len(student_probabilities) != 3:
        raise ValueError("v4.30 requires all three student seeds")
    js: list[np.ndarray] = []
    displacement: list[np.ndarray] = []
    for left in range(3):
        for right in range(left + 1, 3):
            for scale in (1, 2, 4):
                p, q = student_probabilities[left][scale], student_probabilities[right][scale]
                mean = 0.5 * (p + q)
                js.append(
                    0.5
                    * (
                        np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(mean, 1e-12)), axis=1)
                        + np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(mean, 1e-12)), axis=1)
                    )
                )
                expected_p = np.einsum("bkhw,ki->bhwi", p, offsets[scale])
                expected_q = np.einsum("bkhw,ki->bhwi", q, offsets[scale])
                displacement.append(np.linalg.norm(expected_p - expected_q, axis=-1))
    return {
        "js_median": float(np.median(np.concatenate([x.reshape(-1) for x in js]))),
        "js_p95": float(np.percentile(np.concatenate([x.reshape(-1) for x in js]), 95)),
        "expected_displacement_p95": float(
            np.percentile(np.concatenate([x.reshape(-1) for x in displacement]), 95)
        ),
    }


def make_summary(
    *,
    raw: Mapping[str, Any],
    checkpoint_hashes: Mapping[str, str],
    cache_manifest: Path,
    stabilization: Mapping[str, float],
    arm_metrics: Mapping[str, Mapping[str, object]],
    constituent_metrics: Mapping[str, list[Mapping[str, object]]] | None = None,
    constituent_gate_checks: Mapping[str, list[Mapping[str, bool]]] | None = None,
    median_gate_checks: Mapping[str, Mapping[str, bool]] | None = None,
    arm_passed: Mapping[str, bool] | None = None,
    teacher_consensus_cache: Mapping[str, object] | None = None,
    config_path: Path | None = None,
    v48_config_path: Path | None = None,
    v429_summary_path: Path | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Pure decision tree used by grouped OOF after predictions are written."""
    stable = stabilization_gate(
        stabilization["js_median"],
        stabilization["js_p95"],
        stabilization["expected_displacement_p95"],
    )
    status = (
        "stabilization_gate_failed" if not all(stable.values()) else "completed_oof_gate_failed"
    )
    rankable = {
        name: metrics
        for name, metrics in arm_metrics.items()
        if isinstance(metrics.get("pearson"), (float, int))
        and np.isfinite(float(metrics["pearson"]))
    }
    rank = choose_rank_winner(rankable) if rankable else None
    passed = dict(arm_passed or {})
    promotion_metrics = {
        name: metrics for name, metrics in arm_metrics.items() if passed.get(name, False)
    }
    champion = (
        promoted_champion(promotion_metrics)
        if all(stable.values()) and not diagnostic_only and promotion_metrics
        else None
    )
    if diagnostic_only:
        status = "diagnostic_only"
        rank = None
    elif champion is not None:
        status = "promotion_requires_development_validation"
    return {
        "artifact_type": "object_event_v4_30_stable_similarity",
        "status": status,
        "config": raw,
        "config_file_sha256": sha256(config_path) if config_path is not None else None,
        "canonical_protocol_sha256": canonical_protocol_sha256(dict(raw)),
        "v48_config_file_sha256": sha256(v48_config_path) if v48_config_path is not None else None,
        "v429_summary_file_sha256": sha256(v429_summary_path)
        if v429_summary_path is not None
        else None,
        "checkpoint_file_sha256": dict(checkpoint_hashes),
        "cache_manifest_sha256": sha256(cache_manifest),
        "teacher_consensus_cache": dict(teacher_consensus_cache or {}),
        "rng_schedule": raw["train"]["optimization_seed_by_fold"],
        "stabilization": {**stabilization, "checks": stable},
        "rank_winner": rank,
        "promoted_champion": champion,
        "median_metrics": dict(arm_metrics),
        "constituent_metrics": dict(constituent_metrics or {}),
        "constituent_gate_checks": dict(constituent_gate_checks or {}),
        "median_gate_checks": dict(
            median_gate_checks
            or {name: oof_gates(metrics) for name, metrics in arm_metrics.items()}
        ),
        "arm_passed": passed,
        "next_action": (
            "Run the locked one-time development-validation phase before any sealed data."
            if status == "promotion_requires_development_validation"
            else None
        ),
        "scientific_contract": {
            "sealed_data_materialized": False,
            "development_validation_materialized_at_most_once_after_oof_champion": False,
            "official_eap_test_opened": False,
            "evttc_opened": False,
            "all_seed_constituents_retained": True,
            "ema_epochs_fixed": raw["stabilization"]["checkpoint_ema_epochs"],
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "git": _git_state(),
            "time_utc": datetime.now(UTC).isoformat(),
        },
    }


def _teacher_consensus_cache_audit(cache: TeacherConsensusCache) -> dict[str, object]:
    """Strict-JSON audit metadata for the single event-only cache build."""
    return {
        "schema": cache.schema,
        "consensus_cache_sha256": cache.consensus_cache_sha256,
        "consensus_config_sha256": cache.consensus_config_sha256,
        "checkpoint_file_sha256": dict(cache.checkpoint_file_sha256),
        "row_count": cache.row_count,
        "teacher_backbone_forward_batches": cache.teacher_backbone_forward_batches,
        "consensus_build_count": cache.consensus_build_count,
        "elapsed_seconds": cache.elapsed_seconds,
    }


def _rng(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _teacher_pairs(
    teachers: list[object], events: torch.Tensor
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Event-only dense pairs for all frozen teachers; no annotations enter here."""
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    foregrounds: list[torch.Tensor] = []
    # Activity is raw 0:10 and deliberately independent of every teacher.
    raw = events[:, 2, :10].abs().sum(dim=1)
    positive = raw > 0
    denom = (
        (raw.square() * positive).sum((-2, -1), keepdim=True)
        / positive.sum((-2, -1), keepdim=True).clamp_min(1)
    ).sqrt()
    activity = (raw / denom.clamp_min(1e-6)).clamp(0.0, 4.0)
    for backbone in teachers:
        maps, _, fg, _ = backbone._foreground_and_features(events)  # type: ignore[attr-defined]
        temporal = backbone.temporal_projection(backbone._temporal_maps(maps))  # type: ignore[attr-defined]
        temporal = torch.nn.functional.interpolate(
            temporal, size=maps.shape[-2:], mode="bilinear", align_corners=False
        )
        dense = torch.cat((maps, temporal[:, None].expand(-1, 3, -1, -1, -1)), dim=2)
        pairs.append((dense[:, 1], dense[:, 2]))
        foregrounds.append(
            torch.nn.functional.interpolate(
                fg, size=maps.shape[-2:], mode="bilinear", align_corners=False
            )[:, 2]
        )
    if len(foregrounds) != 3:
        raise ValueError("v4.30 foreground consensus requires all three locked teachers")
    activity = torch.nn.functional.interpolate(
        activity[:, None], size=foregrounds[0].shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(1)
    return pairs, torch.stack(foregrounds).mean(dim=0), activity


def _consensus(
    model: ObjectEventTTCV430, teachers: list[object], events: torch.Tensor
) -> dict[int, torch.Tensor]:
    with torch.no_grad():
        pairs, foreground, activity = _teacher_pairs(teachers, events)
        return model.locked_teacher_consensus(pairs, foreground, activity)


def _cache_config_sha256(cfg: Mapping[str, object]) -> str:
    """Hash only the event-derived consensus configuration, never labels or metadata."""
    model_cfg = {key: value for key, value in cfg.items() if key != "batch_size"}
    return hashlib.sha256(json.dumps(model_cfg, sort_keys=True).encode("utf-8")).hexdigest()


def _teacher_consensus_cache_sha256(
    probabilities: Mapping[int, torch.Tensor],
    *,
    consensus_config_sha256: str,
    checkpoint_file_sha256: Mapping[str, str],
    row_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(TEACHER_CONSENSUS_CACHE_SCHEMA.encode("utf-8"))
    digest.update(str(row_count).encode("ascii"))
    digest.update(consensus_config_sha256.encode("ascii"))
    for seed, checkpoint_hash in sorted(checkpoint_file_sha256.items()):
        digest.update(seed.encode("utf-8"))
        digest.update(checkpoint_hash.encode("ascii"))
    for scale in (1, 2, 4):
        value = probabilities[scale]
        digest.update(f"{scale}:{tuple(value.shape)}:{value.dtype}".encode("ascii"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_teacher_consensus_cache(
    split: object,
    checkpoints: Mapping[int, Path],
    *,
    checkpoint_hashes: Mapping[str, str],
    v48_config: Path,
    cfg: Mapping[str, object],
    device: torch.device,
) -> TeacherConsensusCache:
    """Build one event-only frozen-teacher consensus table for the selected rows."""
    started = perf_counter()
    teachers = _frozen_teachers(checkpoints, v48_config=v48_config, device=device)
    model_cfg = {key: value for key, value in cfg.items() if key != "batch_size"}
    consensus_model = (
        ObjectEventTTCV430(teachers[0], ObjectEventV430Config(**model_cfg)).to(device).eval()
    )
    batch_size = int(cfg["batch_size"])
    rows: dict[int, list[torch.Tensor]] = {scale: [] for scale in (1, 2, 4)}
    forward_batches = 0
    with torch.no_grad():
        for start in range(0, len(split.events), batch_size):
            events = split.events[start : start + batch_size].to(device, torch.float32)
            consensus = _consensus(consensus_model, teachers, events)
            forward_batches += len(teachers)
            for scale in rows:
                rows[scale].append(consensus[scale].detach().to("cpu", torch.float32))
    probabilities = {scale: torch.cat(value, dim=0).contiguous() for scale, value in rows.items()}
    if any(
        len(value) != len(split.events) or not bool(torch.isfinite(value).all())
        for value in probabilities.values()
    ):
        raise FloatingPointError("teacher consensus cache is incomplete or nonfinite")
    consensus_config_sha256 = _cache_config_sha256(cfg)
    digest = _teacher_consensus_cache_sha256(
        probabilities,
        consensus_config_sha256=consensus_config_sha256,
        checkpoint_file_sha256=checkpoint_hashes,
        row_count=len(split.events),
    )
    del consensus_model
    del teachers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return TeacherConsensusCache(
        probabilities=probabilities,
        row_count=len(split.events),
        schema=TEACHER_CONSENSUS_CACHE_SCHEMA,
        consensus_cache_sha256=digest,
        consensus_config_sha256=consensus_config_sha256,
        checkpoint_file_sha256=dict(checkpoint_hashes),
        teacher_backbone_forward_batches=forward_batches,
        consensus_build_count=1,
        elapsed_seconds=perf_counter() - started,
    )


def _averaged_head(
    states: list[dict[str, torch.Tensor]], checkpoint_ema_epochs: list[int]
) -> dict[str, torch.Tensor]:
    if len(states) != len(checkpoint_ema_epochs):
        raise RuntimeError("v4.30 head snapshots differ from the configured EMA epochs")
    return {key: torch.stack([state[key] for state in states]).mean(dim=0) for key in states[0]}


def _head_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _fresh_arm_model(
    checkpoint: Path,
    state: Mapping[str, torch.Tensor],
    *,
    v48_config: Path,
    arm: str,
    arm_config: Mapping[str, object],
    device: torch.device,
) -> ObjectEventTTCV430:
    """Fresh paired arm start from immutable CPU stage-1 head weights."""
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model_config = {key: value for key, value in arm_config.items() if key != "batch_size"}
    model = ObjectEventTTCV430(backbone, ObjectEventV430Config(arm=arm, **model_config)).to(device)
    model.local_projection.load_state_dict(dict(state), strict=True)
    return model


def _stage1(
    split: object,
    held: np.ndarray,
    checkpoint: Path,
    *,
    consensus_cache: TeacherConsensusCache,
    v48_config: Path,
    cfg: Mapping[str, object],
    train_cfg: Mapping[str, object],
    stabilization_cfg: Mapping[str, object],
    seed: int,
    fold_seed: int,
    device: torch.device,
) -> tuple[ObjectEventTTCV430, list[dict[str, float]], dict[int, np.ndarray]]:
    """Fit-only configured posterior distillation with configured head snapshots."""
    if consensus_cache.row_count != len(split.events):
        raise ValueError("stage-1 consensus cache does not align to the selected split")
    effective_seed = fold_seed + seed
    sampler = _rng(effective_seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model_config = {key: value for key, value in cfg.items() if key != "batch_size"}
    model = ObjectEventTTCV430(backbone, ObjectEventV430Config(**model_config)).to(device)
    batch_size = int(cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    learning_rate = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    checkpoint_ema_epochs = [
        int(epoch) for epoch in cast(list[object], stabilization_cfg["checkpoint_ema_epochs"])
    ]
    optimizer = torch.optim.AdamW(
        model.head_parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    fit = np.setdiff1d(np.arange(len(split.events)), held)
    snapshots: list[dict[str, torch.Tensor]] = []
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        order = fit.copy()
        sampler.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            events = split.events[index].to(device, torch.float32)
            consensus = consensus_cache.for_indices(index, device)
            output = model(events)
            loss = torch.stack(
                [
                    posterior_kl(consensus[s], output.posteriors_12[s].probabilities)
                    for s in (1, 2, 4)
                ]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite v4.30 stabilization loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch + 1 in checkpoint_ema_epochs:
            snapshots.append(
                {
                    k: v.detach().cpu().clone()
                    for k, v in model.local_projection.state_dict().items()
                }
            )
        history.append({"epoch": float(epoch + 1), "distill_kl": float(np.mean(losses))})
    model.local_projection.load_state_dict(
        _averaged_head(snapshots, checkpoint_ema_epochs), strict=True
    )
    model.eval()
    posterior: dict[int, list[np.ndarray]] = {s: [] for s in (1, 2, 4)}
    with torch.no_grad():
        for start in range(0, len(held), batch_size):
            event = split.events[held[start : start + batch_size]].to(device, torch.float32)
            result = model(event)
            for scale in posterior:
                posterior[scale].append(result.posteriors_12[scale].probabilities.cpu().numpy())
    return model, history, {scale: np.concatenate(values) for scale, values in posterior.items()}


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV430,
    split: object,
    device: torch.device,
    *,
    batch_size: int | None = None,
    control: str | None = None,
    controls: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Return one diagnostic value per supplied row, preserving split order.

    ``control`` only modifies the event tensor.  It never sees labels, boxes,
    IDs, or metadata, so the saved control arrays are suitable for a later
    metric implementation without reopening a held fold.
    """
    rows: dict[str, list[np.ndarray]] = {key: [] for key in PREDICTION_FIELDS}
    resolved_batch_size = len(split.events) if batch_size is None else batch_size
    if resolved_batch_size <= 0:
        raise ValueError("prediction batch size must be positive")
    for start in range(0, len(split.events), resolved_batch_size):
        events = split.events[start : start + resolved_batch_size].to(device, torch.float32)
        if control is not None:
            if controls is None:
                raise ValueError("stage-2 controls are required for controlled prediction")
            events = _apply_event_control(events, control, controls)
        output = model(events)
        values = {
            "prediction": output.expansion,
            "log_eta": output.log_eta,
            "posterior_variance": output.posterior_variance,
            "unknown": output.unknown,
            "kappa": output.fit_12.kappa,
            "rotation_radians": output.rotation_radians,
            "translation_magnitude": output.translation_magnitude,
            "fit12_effective_support_mass": output.fit_12.effective_mass,
            "fit_residual": output.fit_12.residual,
            "correlation_entropy": output.correlation_entropy,
            "correlation_confidence": output.correlation_confidence,
            "boundary_probability": output.boundary_probability,
            "normal_flow_residual": output.normal_flow_residual,
            "cycle_matrix_error": output.cycle_matrix_error,
            "cycle_translation_error": output.cycle_translation_error,
        }
        for key, value in values.items():
            rows[key].append(value.detach().cpu().numpy().reshape(-1))
    result = {key: np.concatenate(value) for key, value in rows.items()}
    _validate_prediction_result(result, len(split.events), context=control or "unperturbed")
    return result


def _apply_event_control(
    events: torch.Tensor, control: str, controls: Mapping[str, object]
) -> torch.Tensor:
    """Apply a locked event-only control to a ``[B,3,C,H,W]`` input tensor."""
    if events.ndim != 5 or events.shape[1] != 3:
        raise ValueError("stage-2 controls require events [B,3,C,H,W]")
    if control == "zero_event":
        if controls.get("zero_event") is not True:
            raise ValueError("zero-event control is not locked on")
        return torch.zeros_like(events)
    permutation_key = f"{control}_permutation"
    permutation = controls.get(permutation_key)
    expected = {
        "temporal_shuffle": [2, 0, 1],
        "endpoint_swap": [0, 2, 1],
    }.get(control)
    if expected is None or permutation != expected:
        raise ValueError(f"invalid locked {control} permutation")
    return events.index_select(1, torch.as_tensor(permutation, device=events.device))


def _validate_prediction_result(
    result: Mapping[str, np.ndarray], length: int, *, context: str
) -> None:
    """Reject schema or row-count drift before an OOF row can be written."""
    if tuple(result) != PREDICTION_FIELDS:
        raise RuntimeError(f"{context} prediction schema changed")
    if length <= 0 or any(
        np.asarray(result[field]).reshape(-1).shape[0] != length for field in result
    ):
        raise RuntimeError(f"{context} prediction length differs from held rows")


def _assert_nonzero_unperturbed(result: Mapping[str, np.ndarray]) -> None:
    """The v4.30 nonzero-event forward contract permits no invalid OOF row."""
    if bool(np.any(result["unknown"])):
        raise RuntimeError("nonzero unperturbed OOF rows unexpectedly returned UNKNOWN")
    if any(not bool(np.isfinite(result[field]).all()) for field in PHYSICAL_PREDICTION_FIELDS):
        raise FloatingPointError("nonzero unperturbed OOF diagnostics are nonfinite")


def _assert_zero_event_contract(result: Mapping[str, np.ndarray]) -> None:
    """Zero event input must produce explicit UNKNOWN and no physical estimate."""
    if not bool(np.asarray(result["unknown"], dtype=bool).all()):
        raise RuntimeError("zero-event control failed to return UNKNOWN for every held row")
    required_nan = ("prediction", "log_eta", "posterior_variance")
    if any(not bool(np.isnan(result[field]).all()) for field in required_nan):
        raise RuntimeError("zero-event control emitted a physical prediction")


def _t1_t2_boxes(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Return the loss contract's t1/t2 boxes from the materialized annotations.

    The v4.30 loss receives exactly ``[B,2,4]``.  Older materializations can
    still expose the unambiguous three-frame history ``[t0,t1,t2]``; only at
    this annotation adapter boundary it is narrowed to ``[t1,t2]``.
    """
    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("v4.30 boxes must have shape [B,2,4]")
    if boxes_xyxy.shape[1] == 2:
        return boxes_xyxy
    if boxes_xyxy.shape[1] == 3:
        return boxes_xyxy[:, 1:]
    raise ValueError("v4.30 boxes must have shape [B,2,4]")


def _empty_oof_prediction(length: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create original-row storage with an independent coverage guard."""
    if length <= 0:
        raise ValueError("OOF collection requires at least one row")
    arrays = {
        field: (np.zeros(length, dtype=bool) if field == "unknown" else np.full(length, np.nan))
        for field in PREDICTION_FIELDS
    }
    return arrays, np.zeros(length, dtype=bool)


def _accumulate_oof(
    destination: Mapping[str, np.ndarray],
    coverage: np.ndarray,
    held: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    context: str,
) -> None:
    """Write held-order predictions into original OOF positions exactly once."""
    held = np.asarray(held, dtype=np.int64).reshape(-1)
    _validate_prediction_result(values, len(held), context=context)
    if (
        len(held) == 0
        or np.any(held < 0)
        or np.any(held >= len(coverage))
        or len(np.unique(held)) != len(held)
        or bool(coverage[held].any())
    ):
        raise RuntimeError(f"{context} has duplicate, missing, or invalid held indices")
    for field in PREDICTION_FIELDS:
        destination[field][held] = values[field]
    coverage[held] = True


def _assert_complete_coverage(coverage: np.ndarray, *, context: str) -> None:
    """Fail closed rather than silently dropping any original OOF row."""
    if not bool(coverage.all()):
        missing = np.flatnonzero(~coverage)
        raise RuntimeError(f"{context} missing {len(missing)} OOF rows")


def _oof_metadata(split: object) -> dict[str, np.ndarray]:
    """Return explicit original-row targets and provenance for saved OOF arrays."""
    length = len(split.events)
    values: dict[str, np.ndarray] = {
        "oof_row_index": np.arange(length, dtype=np.int64),
        "sample_token": np.asarray(split.sample_tokens, dtype=str),
        "sequence_id": np.asarray(split.sequence_ids, dtype=str),
        "track_id": np.asarray(split.track_ids, dtype=str),
        "target_expansion": (split.delta_t_s / split.target_ttc_s)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False),
        "delta_t_s": split.delta_t_s.detach().cpu().numpy().astype(np.float64, copy=False),
        "target_ttc_s": split.target_ttc_s.detach().cpu().numpy().astype(np.float64, copy=False),
    }
    values["target_log_eta"] = np.log1p(-values["target_expansion"])
    if any(np.asarray(value).reshape(-1).shape[0] != length for value in values.values()):
        raise RuntimeError("OOF metadata length differs from original rows")
    return values


def _save_oof_npz(
    path: Path,
    metadata: Mapping[str, np.ndarray],
    unperturbed: Mapping[str, np.ndarray],
    controls: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """Persist an explicit flat schema that does not rely on positional joins."""
    payload = {**metadata, **unperturbed}
    for control, values in controls.items():
        payload.update({f"{control}_{field}": value for field, value in values.items()})
    lengths = {np.asarray(value).reshape(-1).shape[0] for value in payload.values()}
    if len(lengths) != 1:
        raise RuntimeError("refusing to save OOF payload with mismatched array lengths")
    np.savez(path, **payload)


def _median_prediction(rows: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Take a rowwise median without removing UNKNOWN or nonfinite evidence."""
    if len(rows) != 3:
        raise ValueError("v4.30 requires exactly three seed payloads")
    length = len(rows[0]["prediction"])
    for index, row in enumerate(rows):
        _validate_prediction_result(row, length, context=f"median seed {index}")
    result: dict[str, np.ndarray] = {}
    for field in PREDICTION_FIELDS:
        values = np.stack([row[field] for row in rows])
        result[field] = np.any(values, axis=0) if field == "unknown" else np.median(values, axis=0)
    _validate_prediction_result(result, length, context="median")
    return result


def _metrics_from_payload(
    payload: Mapping[str, np.ndarray],
    metadata: Mapping[str, np.ndarray],
    seed_predictions: np.ndarray,
    controls: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    """Call the locked pure metric API with one complete, aligned OOF payload."""
    metrics = compute_oof_metrics(
        target_g=np.asarray(metadata["target_expansion"], dtype=np.float64),
        target_log_eta=np.asarray(metadata["target_log_eta"], dtype=np.float64),
        prediction=np.asarray(payload["prediction"], dtype=np.float64),
        predicted_log_eta=np.asarray(payload["log_eta"], dtype=np.float64),
        posterior_variance=np.asarray(payload["posterior_variance"], dtype=np.float64),
        support=np.asarray(payload["fit12_effective_support_mass"], dtype=np.float64),
        sequence_ids=np.asarray(metadata["sequence_id"], dtype=str).tolist(),
        track_ids=np.asarray(metadata["track_id"], dtype=str).tolist(),
        seed_predictions=np.asarray(seed_predictions, dtype=np.float64),
        shuffle_prediction=np.asarray(controls["temporal_shuffle"]["prediction"], dtype=np.float64),
        endpoint_prediction=np.asarray(controls["endpoint_swap"]["prediction"], dtype=np.float64),
        zero_unknown=np.asarray(controls["zero_event"]["unknown"], dtype=bool),
        zero_prediction=np.asarray(controls["zero_event"]["prediction"], dtype=np.float64),
        delta_t_s=np.asarray(metadata["delta_t_s"], dtype=np.float64),
    )
    metrics["finite_predictions"] = float(np.isfinite(payload["prediction"]).all())
    metrics["finite_posterior_variances"] = float(np.isfinite(payload["posterior_variance"]).all())
    buckets = metrics.get("buckets")
    if not isinstance(buckets, Mapping):
        raise RuntimeError("locked OOF metric API did not return magnitude buckets")
    for index in range(4):
        bucket = buckets.get(str(index))
        metrics[f"magnitude_ratio_{index}"] = (
            bucket.get("ratio") if isinstance(bucket, Mapping) else None
        )
    return metrics


def _aggregate_arm_oof(
    seed_payloads: list[Mapping[str, np.ndarray]],
    seed_controls: list[Mapping[str, Mapping[str, np.ndarray]]],
    metadata: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Compute all seed and median metrics from aligned OOF arrays, without filtering rows."""
    if len(seed_payloads) != 3 or len(seed_controls) != 3:
        raise ValueError("v4.30 arm aggregation requires all three seed payloads and controls")
    length = len(np.asarray(metadata["target_expansion"]).reshape(-1))
    if any(len(payload["prediction"]) != length for payload in seed_payloads):
        raise RuntimeError("seed OOF payload length differs from target metadata")
    if any(set(controls) != set(CONTROL_NAMES) for controls in seed_controls):
        raise RuntimeError("seed OOF controls are incomplete")
    stack = np.stack([payload["prediction"] for payload in seed_payloads])
    median = _median_prediction(seed_payloads)
    median_controls = {
        control: _median_prediction([controls[control] for controls in seed_controls])
        for control in CONTROL_NAMES
    }
    constituent_metrics = [
        _metrics_from_payload(payload, metadata, stack, controls)
        for payload, controls in zip(seed_payloads, seed_controls, strict=True)
    ]
    median_metrics = _metrics_from_payload(median, metadata, stack, median_controls)
    combined = gate_constituents_and_median(constituent_metrics, median_metrics)
    constituent_gates = [
        oof_gates(
            {key: value for key, value in metric.items() if isinstance(value, (float, int, bool))}
        )
        for metric in constituent_metrics
    ]
    median_gates = oof_gates(
        {
            key: value
            for key, value in median_metrics.items()
            if isinstance(value, (float, int, bool))
        }
    )
    return {
        "seed_predictions": stack,
        "median_payload": median,
        "median_controls": median_controls,
        "constituent_metrics": constituent_metrics,
        "median_metrics": median_metrics,
        "constituent_gate_checks": constituent_gates,
        "median_gate_checks": median_gates,
        "arm_passed": bool(combined["constituents"] and combined["median"]),
        "gate_constituents_and_median": combined,
    }


def _inject_arm_b_gains(arms: Mapping[str, dict[str, object]]) -> None:
    """Add preregistered Arm-B gains from actual Arm-A/B median metrics."""
    a = arms["stable_multiscale_similarity"]["median_metrics"]
    b = arms["stable_multiscale_similarity_normal_flow"]["median_metrics"]
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError("v4.30 Arm-B gains require flat median metric dictionaries")
    a_sequences, b_sequences = a.get("per_sequence"), b.get("per_sequence")
    if (
        not isinstance(a_sequences, Mapping)
        or not isinstance(b_sequences, Mapping)
        or set(a_sequences) != set(b_sequences)
    ):
        b["paired_sequence_pearson_gain"] = None
    else:
        paired: list[float] = []
        for sequence in sorted(a_sequences):
            left = a_sequences[sequence]
            right = b_sequences[sequence]
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                paired = []
                break
            left_value, right_value = left.get("pearson"), right.get("pearson")
            if (
                not isinstance(left_value, (float, int))
                or not isinstance(right_value, (float, int))
                or not np.isfinite(float(left_value))
                or not np.isfinite(float(right_value))
            ):
                paired = []
                break
            paired.append(float(right_value) - float(left_value))
        b["paired_sequence_pearson_gain"] = float(np.mean(paired)) if paired else None
    gain_pairs = {
        "high_bucket_pearson_gain": ("high_bucket_pearson",),
        "negative_track_macro_gain": ("negative_track_macro_accuracy",),
        "shuffle_ratio_reduction": ("shuffle_ratio",),
    }
    for output, (field,) in gain_pairs.items():
        left, right = a.get(field), b.get(field)
        if not isinstance(left, (float, int)) or not isinstance(right, (float, int)):
            b[output] = None
        elif output == "shuffle_ratio_reduction":
            b[output] = float(left) - float(right)
        else:
            b[output] = float(right) - float(left)


def _json_safe(value: object) -> object:
    """Convert NumPy values and invalid numbers to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _frozen_teachers(
    checkpoints: Mapping[int, Path], *, v48_config: Path, device: torch.device
) -> list[object]:
    """Load the fixed three-teacher ensemble without selecting a constituent."""
    teachers: list[object] = []
    for seed in (7, 13, 23):
        teacher, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoints[seed])
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers.append(teacher)
    return teachers


def _stage1_full(
    split: object,
    checkpoint: Path,
    *,
    consensus_cache: TeacherConsensusCache,
    v48_config: Path,
    cfg: Mapping[str, object],
    train_cfg: Mapping[str, object],
    stabilization_cfg: Mapping[str, object],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    """Run the configured stabilization stage on every train row once."""
    if consensus_cache.row_count != len(split.events):
        raise ValueError("full-train consensus cache does not align to the selected split")
    sampler = _rng(seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model_config = {key: value for key, value in cfg.items() if key != "batch_size"}
    model = ObjectEventTTCV430(backbone, ObjectEventV430Config(**model_config)).to(device)
    batch_size = int(cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    learning_rate = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    checkpoint_ema_epochs = [
        int(epoch) for epoch in cast(list[object], stabilization_cfg["checkpoint_ema_epochs"])
    ]
    optimizer = torch.optim.AdamW(
        model.head_parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    all_rows = np.arange(len(split.events))
    snapshots: list[dict[str, torch.Tensor]] = []
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        order = all_rows.copy()
        sampler.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            events = split.events[index].to(device, torch.float32)
            consensus = consensus_cache.for_indices(index, device)
            output = model(events)
            loss = torch.stack(
                [
                    posterior_kl(consensus[scale], output.posteriors_12[scale].probabilities)
                    for scale in (1, 2, 4)
                ]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite full-train v4.30 stabilization loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch + 1 in checkpoint_ema_epochs:
            snapshots.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.local_projection.state_dict().items()
                }
            )
        history.append({"epoch": float(epoch + 1), "distill_kl": float(np.mean(losses))})
    state = _averaged_head(snapshots, checkpoint_ema_epochs)
    return state, history


def _train_full_head(
    split: object,
    checkpoint: Path,
    state: Mapping[str, torch.Tensor],
    *,
    consensus_cache: TeacherConsensusCache,
    v48_config: Path,
    arm: str,
    arm_config: Mapping[str, object],
    raw: Mapping[str, Any],
    seed: int,
    final_seed: int,
    device: torch.device,
) -> tuple[ObjectEventTTCV430, list[dict[str, float]]]:
    """Freshly construct and train the champion head on every configured train epoch."""
    if consensus_cache.row_count != len(split.events):
        raise ValueError("final-head consensus cache does not align to the selected split")
    model = _fresh_arm_model(
        checkpoint,
        state,
        v48_config=v48_config,
        arm=arm,
        arm_config=arm_config,
        device=device,
    )
    _rng(final_seed)
    sampler = np.random.default_rng(final_seed)
    batch_size = int(arm_config["batch_size"])
    learning_rate = float(raw["train"]["learning_rate"])
    weight_decay = float(raw["train"]["weight_decay"])
    max_grad_norm = float(raw["train"]["max_grad_norm"])
    optimizer = torch.optim.AdamW(
        model.head_parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    all_rows = np.arange(len(split.events))
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(int(raw["train"]["final_epochs"])):
        order = all_rows.copy()
        sampler.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            events = split.events[index].to(device, torch.float32)
            output = model(events)
            consensus = consensus_cache.for_indices(index, device)
            loss, _ = object_event_v4_30_loss(
                output,
                split.delta_t_s[index].to(device),
                split.target_ttc_s[index].to(device),
                consensus_posteriors=consensus,
                sequence_ids=[split.sequence_ids[int(item)] for item in index],
                track_ids=[split.track_ids[int(item)] for item in index],
                config=ObjectEventV430LossConfig(arm=arm, **raw["loss"]),
                visible_heights_px=split.visible_heights_px[index].to(device),
                boxes_xyxy=_t1_t2_boxes(split.boxes_xyxy[index]).to(device),
                image_height=int(split.source_height),
                image_width=int(split.source_width),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite full-train v4.30 champion loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
    return model.eval(), history


def _paper_weighted_mid(
    target_ttc_s: np.ndarray, prediction_expansion: np.ndarray, delta_t_s: np.ndarray
) -> float | None:
    """Strict official signed-bin MiD with the preregistered .5/.3/.1/.1 weights."""
    target = np.asarray(target_ttc_s, dtype=np.float64)
    prediction = np.asarray(prediction_expansion, dtype=np.float64)
    delta = np.asarray(delta_t_s, dtype=np.float64)
    if target.shape != prediction.shape or target.shape != delta.shape:
        raise ValueError("MiD inputs must have aligned shapes")
    if not (
        np.isfinite(target).all() and np.isfinite(prediction).all() and np.isfinite(delta).all()
    ):
        return None
    total = 0.0
    for name, lower, upper in BUCKETS:
        selected = (target > lower) & (target <= upper)
        if not bool(selected.any()):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            true_eta = 1.0 - delta[selected] / target[selected]
            mid = np.abs(np.log(true_eta) - np.log(1.0 - prediction[selected])) * 1e4
        if not bool(np.isfinite(mid).all()):
            return None
        total += PAPER_MID_WEIGHTS[name] * float(np.mean(mid))
    return float(total)


def _finite_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a correlation only when both aligned arrays contain a signal."""
    if (
        len(left) < 2
        or not (np.isfinite(left).all() and np.isfinite(right).all())
        or np.std(left) <= 1e-12
        or np.std(right) <= 1e-12
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _development_metrics(
    metadata: Mapping[str, np.ndarray], payload: Mapping[str, np.ndarray]
) -> dict[str, object]:
    """Compute validation-only criteria on exactly the materialized rows."""
    target = np.asarray(metadata["target_expansion"], dtype=np.float64)
    prediction = np.asarray(payload["prediction"], dtype=np.float64)
    log_eta = np.asarray(payload["log_eta"], dtype=np.float64)
    target_log_eta = np.asarray(metadata["target_log_eta"], dtype=np.float64)
    sequence_ids = np.asarray(metadata["sequence_id"], dtype=str)
    track_ids = np.asarray(metadata["track_id"], dtype=str)
    complete = not bool(np.asarray(payload["unknown"], dtype=bool).any()) and all(
        bool(np.isfinite(payload[field]).all()) for field in PHYSICAL_PREDICTION_FIELDS
    )
    negative = target < 0.0
    positive = ~negative

    def accuracy(mask: np.ndarray) -> float | None:
        return (
            float(np.mean(np.sign(prediction[mask]) == np.sign(target[mask])))
            if bool(mask.any())
            else None
        )

    per_sequence = [
        _finite_corr(prediction[sequence_ids == name], target[sequence_ids == name])
        for name in sorted(set(sequence_ids))
    ]
    track_values: list[float] = []
    for name in sorted(set(track_ids)):
        mask = (track_ids == name) & negative
        if int(mask.sum()) >= 4:
            value = accuracy(mask)
            if value is not None:
                track_values.append(value)
    negative_accuracy, positive_accuracy = accuracy(negative), accuracy(positive)
    return {
        "complete_finite_validation_coverage": complete,
        "pearson": _finite_corr(prediction, target),
        "negative_accuracy": negative_accuracy,
        "balanced_sign_accuracy": (
            None
            if negative_accuracy is None or positive_accuracy is None
            else 0.5 * (negative_accuracy + positive_accuracy)
        ),
        "log_eta_pearson": _finite_corr(log_eta, target_log_eta),
        "minimum_sequence_pearson": min(
            (item for item in per_sequence if item is not None), default=None
        ),
        "negative_track_macro_accuracy": float(np.mean(track_values)) if track_values else None,
        "paper_weighted_mid": _paper_weighted_mid(
            np.asarray(metadata["target_ttc_s"]), prediction, np.asarray(metadata["delta_t_s"])
        ),
    }


def _development_decision(
    candidate: Mapping[str, object], baseline: Mapping[str, object], criteria: Mapping[str, object]
) -> dict[str, bool]:
    """Apply the fixed development criteria; absent or invalid comparators fail closed."""

    def finite(mapping: Mapping[str, object], key: str) -> float | None:
        value = mapping.get(key)
        return (
            float(value) if isinstance(value, (float, int)) and np.isfinite(float(value)) else None
        )

    candidate_mid, baseline_mid = (
        finite(candidate, "paper_weighted_mid"),
        finite(baseline, "paper_weighted_mid"),
    )
    checks = {
        "complete_finite_validation_coverage": candidate.get("complete_finite_validation_coverage")
        is True,
        "pearson_gain": False,
        "negative_accuracy_gain": False,
        "balanced_sign_gain": False,
        "log_eta_pearson": False,
        "negative_track_macro_gain": False,
        "relative_paper_weighted_mid_improvement": False,
    }
    for key, criterion, check in (
        ("pearson", "minimum_pearson_gain_over_v410", "pearson_gain"),
        ("negative_accuracy", "minimum_negative_accuracy_gain_over_v410", "negative_accuracy_gain"),
        ("balanced_sign_accuracy", "minimum_balanced_sign_gain_over_v410", "balanced_sign_gain"),
        (
            "negative_track_macro_accuracy",
            "minimum_negative_track_macro_gain_over_v410",
            "negative_track_macro_gain",
        ),
    ):
        value, reference = finite(candidate, key), finite(baseline, key)
        threshold = finite(criteria, criterion)
        checks[check] = (
            value is not None
            and reference is not None
            and threshold is not None
            and value >= reference + threshold
        )
    log_eta, minimum_log_eta = (
        finite(candidate, "log_eta_pearson"),
        finite(criteria, "minimum_log_eta_pearson"),
    )
    checks["log_eta_pearson"] = (
        log_eta is not None and minimum_log_eta is not None and log_eta >= minimum_log_eta
    )
    mid_threshold = finite(criteria, "minimum_relative_paper_weighted_mid_improvement_over_v410")
    checks["relative_paper_weighted_mid_improvement"] = (
        candidate_mid is not None
        and baseline_mid is not None
        and baseline_mid > 0.0
        and mid_threshold is not None
        and (baseline_mid - candidate_mid) / baseline_mid >= mid_threshold
    )
    return checks


def _run_development_validation(
    *,
    train: object,
    manifest: Mapping[str, object],
    checkpoints: Mapping[int, Path],
    consensus_cache: TeacherConsensusCache,
    raw: Mapping[str, Any],
    champion: str,
    v48_config: Path,
    cache_manifest: Path,
    v410_summary_path: Path,
    ensemble_validation_path: Path,
    base_input_size: object,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    """Execute the one-time full-train then validation procedure after OOF promotion only."""
    full_records: list[dict[str, object]] = []
    models: dict[int, ObjectEventTTCV430] = {}
    for seed in (7, 13, 23):
        state, stabilization_history = _stage1_full(
            train,
            checkpoints[seed],
            consensus_cache=consensus_cache,
            v48_config=v48_config,
            cfg=raw["arms"]["stable_multiscale_similarity"],
            train_cfg=raw["train"],
            stabilization_cfg=raw["stabilization"],
            seed=seed,
            device=device,
        )
        final_seed = int(raw["train"]["final_training_seed_by_student"][seed])
        model, final_history = _train_full_head(
            train,
            checkpoints[seed],
            state,
            consensus_cache=consensus_cache,
            v48_config=v48_config,
            arm=champion,
            arm_config=raw["arms"][champion],
            raw=raw,
            seed=seed,
            final_seed=final_seed,
            device=device,
        )
        models[seed] = model
        full_records.append(
            {
                "seed": seed,
                "stabilization_train_rows": len(train.events),
                "stabilization_epochs": int(raw["train"]["epochs"]),
                "stabilization_ema_epochs": raw["stabilization"]["checkpoint_ema_epochs"],
                "stabilization_history": stabilization_history,
                "averaged_head_sha256": _head_state_hash(state),
                "final_training_seed": final_seed,
                "final_train_rows": len(train.events),
                "final_epochs": int(raw["train"]["final_epochs"]),
                "final_history": final_history,
                "teacher_consensus_cache": _teacher_consensus_cache_audit(consensus_cache),
                "training_controls": {
                    "stabilization": {
                        "epochs": int(raw["train"]["epochs"]),
                        "batch_size": int(
                            raw["arms"]["stable_multiscale_similarity"]["batch_size"]
                        ),
                        "learning_rate": float(raw["train"]["learning_rate"]),
                        "weight_decay": float(raw["train"]["weight_decay"]),
                        "max_grad_norm": float(raw["train"]["max_grad_norm"]),
                        "checkpoint_ema_epochs": raw["stabilization"]["checkpoint_ema_epochs"],
                    },
                    "final": {
                        "epochs": int(raw["train"]["final_epochs"]),
                        "batch_size": int(raw["arms"][champion]["batch_size"]),
                        "learning_rate": float(raw["train"]["learning_rate"]),
                        "weight_decay": float(raw["train"]["weight_decay"]),
                        "max_grad_norm": float(raw["train"]["max_grad_norm"]),
                        "seed": final_seed,
                    },
                },
            }
        )
    # These file reads are deliberately deferred until every full-train model exists.
    audit: dict[str, object] = {
        "full_models_ready_before_development_reads": len(models) == 3,
        "v410_summary_read": False,
        "ensemble_validation_read": False,
        "validation_materialize_calls": 0,
        "v410_summary_file_sha256": None,
        "ensemble_validation_file_sha256": None,
    }
    if not v410_summary_path.is_file() or not ensemble_validation_path.is_file():
        raise FileNotFoundError("OOF promotion requires both locked v4.10 development references")
    v410_summary = json.loads(v410_summary_path.read_text(encoding="utf-8"))
    audit["v410_summary_read"] = True
    audit["v410_summary_file_sha256"] = sha256(v410_summary_path)
    if v410_summary.get("artifact_type") != "object_event_v4_10_true_seed_fixed_fusion_robustness":
        raise ValueError("--v410-summary is not the locked v4.10 development artifact")
    ensemble = _read_ensemble(ensemble_validation_path)
    audit["ensemble_validation_read"] = True
    audit["ensemble_validation_file_sha256"] = sha256(ensemble_validation_path)
    validation, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=base_input_size
    )
    audit["validation_materialize_calls"] = 1
    aligned = _align_ensemble(validation, ensemble)
    metadata = _oof_metadata(validation)
    seed_payloads: dict[int, dict[str, np.ndarray]] = {
        seed: _predict(
            model, validation, device, batch_size=int(raw["arms"][champion]["batch_size"])
        )
        for seed, model in models.items()
    }
    median = _median_prediction(list(seed_payloads.values()))
    for seed, payload in seed_payloads.items():
        np.savez(output / f"validation_{champion}_seed{seed}.npz", **metadata, **payload)
    np.savez(output / f"validation_{champion}_median.npz", **metadata, **median)
    candidate_metrics = _development_metrics(metadata, median)
    v410_prediction = aligned["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    v410_payload = dict(median)
    v410_payload["prediction"] = v410_prediction
    with np.errstate(divide="ignore", invalid="ignore"):
        v410_payload["log_eta"] = np.log1p(-v410_prediction)
    v410_payload["unknown"] = np.zeros(len(v410_prediction), dtype=bool)
    baseline_metrics = _development_metrics(metadata, v410_payload)
    checks = _development_decision(candidate_metrics, baseline_metrics, raw["development"])
    table = pd.DataFrame(metadata)
    for seed, payload in seed_payloads.items():
        table[f"prediction_seed_{seed}"] = payload["prediction"]
        table[f"log_eta_seed_{seed}"] = payload["log_eta"]
    table["prediction_median"] = median["prediction"]
    table["log_eta_median"] = median["log_eta"]
    table["v410_prediction"] = v410_prediction
    table.to_csv(output / "development_validation_rows.csv", index=False)
    (output / "development_validation_metrics.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "candidate": candidate_metrics,
                    "v410": baseline_metrics,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return {
        "status": (
            "development_validation_completed_passed"
            if all(checks.values())
            else "development_validation_completed_failed"
        ),
        "development_validation_materialized_once": True,
        "validation_manifest": validation_manifest,
        "full_train_seed_records": full_records,
        "validation_constituent_metrics": {
            str(seed): _development_metrics(metadata, payload)
            for seed, payload in seed_payloads.items()
        },
        "validation_median_metrics": candidate_metrics,
        "v410_validation_metrics": baseline_metrics,
        "development_gate_checks": checks,
        "development_passed": all(checks.values()),
        "development_audit": audit,
    }


def _development_requested(summary: Mapping[str, object], *, diagnostic: bool) -> bool:
    """Keep every development filesystem operation behind genuine full-mode promotion."""
    return (
        not diagnostic
        and summary.get("status") == "promotion_requires_development_validation"
        and isinstance(summary.get("promoted_champion"), str)
    )


def _diagnostic_indices(sequence_ids: list[str], requested: int) -> np.ndarray:
    """Bound diagnostic rows while retaining three nonempty grouped-fold constituents."""
    if requested < 6:
        raise ValueError("--diagnostic-samples requires at least six rows")
    groups = {
        sequence: np.flatnonzero(np.asarray(sequence_ids, dtype=str) == sequence)
        for sequence in sorted(set(sequence_ids))
    }
    eligible = [sequence for sequence, rows in groups.items() if len(rows) >= 2]
    if len(eligible) < 3:
        raise ValueError("diagnostic selection requires three sequences with at least two rows")
    selected_sequences = eligible[:3]
    if requested > sum(len(groups[sequence]) for sequence in selected_sequences):
        raise ValueError("diagnostic request exceeds the bounded three-sequence selection")
    quota = {sequence: 2 for sequence in selected_sequences}
    remaining = requested - 2 * len(selected_sequences)
    cursor = 0
    while remaining:
        sequence = selected_sequences[cursor % len(selected_sequences)]
        if quota[sequence] < len(groups[sequence]):
            quota[sequence] += 1
            remaining -= 1
        cursor += 1
        if cursor > requested * len(selected_sequences):
            raise RuntimeError("diagnostic round-robin selection exhausted available rows")
    chosen = np.concatenate(
        [groups[sequence][: quota[sequence]] for sequence in selected_sequences]
    )
    return np.sort(chosen)


def _verify_grouped_folds(
    folds: list[np.ndarray], sequence_ids: list[str], total_rows: int
) -> None:
    """Reject empty or overlapping grouped folds before any diagnostic training."""
    if len(folds) != 3 or any(len(held) == 0 or len(held) == total_rows for held in folds):
        raise RuntimeError("grouped diagnostic folds require nonempty held and fit rows")
    held_sequences = [set(np.asarray(sequence_ids, dtype=str)[held].tolist()) for held in folds]
    if any(
        left & right
        for index, left in enumerate(held_sequences)
        for right in held_sequences[index + 1 :]
    ):
        raise RuntimeError("grouped diagnostic folds overlap by sequence")
    held_rows = np.concatenate(folds)
    if len(np.unique(held_rows)) != total_rows or set(held_rows.tolist()) != set(range(total_rows)):
        raise RuntimeError("grouped diagnostic folds do not cover each selected row exactly once")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v429-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--v410-summary", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--diagnostic-samples", type=int)
    args = parser.parse_args()
    v429 = json.loads(args.v429_summary.read_text(encoding="utf-8"))
    if v429.get("status") != "completed_oof_gate_failed":
        raise ValueError("v4.30 requires the sealed v4.29 OOF-failure reference")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(raw)
    checkpoint_hashes = validate_checkpoints(
        parse_seed_paths(args.adapted_checkpoint), args.v48_config
    )
    diagnostic = args.diagnostic_samples is not None
    target = DIAGNOSTIC_OUTPUT_ROOT if diagnostic else _safe_output_root(args.output_dir)
    _prepare_output_target(target, force=args.force)
    base, _, _, _, _ = _load_v48_config(args.v48_config)
    train, manifest = _materialize(args.cache_manifest, "train", input_size=base.input_size)
    if diagnostic:
        train = _subset_split(
            train, _diagnostic_indices(train.sequence_ids, int(args.diagnostic_samples))
        )
    elif len(train.events) != int(raw["selection"]["oof_rows_per_constituent"]):
        raise RuntimeError("full v4.30 OOF train-row coverage differs from the locked protocol")
    folds = _sequence_folds(np.asarray(train.sequence_ids), 3, 430)
    if diagnostic:
        _verify_grouped_folds(folds, train.sequence_ids, len(train.events))
    checkpoints = parse_seed_paths(args.adapted_checkpoint)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    consensus_cache = _build_teacher_consensus_cache(
        train,
        checkpoints,
        checkpoint_hashes=checkpoint_hashes,
        v48_config=args.v48_config,
        cfg=raw["arms"]["stable_multiscale_similarity"],
        device=device,
    )
    cache_audit = _teacher_consensus_cache_audit(consensus_cache)
    staged: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    posterior: dict[int, dict[int, list[np.ndarray]]] = {
        seed: {scale: [] for scale in (1, 2, 4)} for seed in (7, 13, 23)
    }
    records: list[dict[str, object]] = []
    for fold, held in enumerate(folds):
        for seed in (7, 13, 23):
            model, history, got = _stage1(
                train,
                held,
                checkpoints[seed],
                consensus_cache=consensus_cache,
                v48_config=args.v48_config,
                cfg=raw["arms"]["stable_multiscale_similarity"],
                train_cfg=raw["train"],
                stabilization_cfg=raw["stabilization"],
                seed=seed,
                fold_seed=int(raw["train"]["optimization_seed_by_fold"][fold]),
                device=device,
            )
            staged[(fold, seed)] = {
                key: value.detach().cpu().clone()
                for key, value in model.local_projection.state_dict().items()
            }
            for scale in got:
                posterior[seed][scale].append(got[scale])
            records.append(
                {
                    "stage": "stabilization",
                    "fold": fold,
                    "seed": seed,
                    "effective_seed": int(raw["train"]["optimization_seed_by_fold"][fold]) + seed,
                    "held": len(held),
                    "history": history,
                    "ema_epochs": raw["stabilization"]["checkpoint_ema_epochs"],
                    "teacher_consensus_cache": cache_audit,
                    "training_controls": {
                        "epochs": int(raw["train"]["epochs"]),
                        "batch_size": int(
                            raw["arms"]["stable_multiscale_similarity"]["batch_size"]
                        ),
                        "learning_rate": float(raw["train"]["learning_rate"]),
                        "weight_decay": float(raw["train"]["weight_decay"]),
                        "max_grad_norm": float(raw["train"]["max_grad_norm"]),
                        "optimization_seed": int(raw["train"]["optimization_seed_by_fold"][fold]),
                    },
                }
            )
    probabilities = [
        {scale: np.concatenate(posterior[seed][scale]) for scale in (1, 2, 4)}
        for seed in (7, 13, 23)
    ]
    offsets = {
        scale: np.stack(
            np.meshgrid(np.arange(-scale, scale + 1), np.arange(-scale, scale + 1), indexing="xy"),
            -1,
        ).reshape(-1, 2)
        * scale
        for scale in (1, 2, 4)
    }
    stabilization = posterior_stability_metrics(probabilities, offsets)
    if not all(stabilization_gate(**stabilization).values()):
        summary = make_summary(
            raw=raw,
            checkpoint_hashes=checkpoint_hashes,
            cache_manifest=args.cache_manifest,
            stabilization=stabilization,
            arm_metrics={},
            teacher_consensus_cache=cache_audit,
            config_path=args.config,
            v48_config_path=args.v48_config,
            v429_summary_path=args.v429_summary,
            diagnostic_only=diagnostic,
        )
        summary.update(
            {
                "records": records,
                "train_manifest": manifest,
                "diagnostic_only": diagnostic,
                "rank_winner": None,
                "promoted_champion": None,
            }
        )
    else:
        # Geometry OOF is deliberately head-only and starts from the fixed stage-1 mean.
        metadata = _oof_metadata(train)
        arms: dict[str, dict[str, object]] = {}
        for arm in raw["arms"]:
            seed_payloads: list[dict[str, np.ndarray]] = []
            seed_controls: list[dict[str, dict[str, np.ndarray]]] = []
            for seed in (7, 13, 23):
                unperturbed, coverage = _empty_oof_prediction(len(train.events))
                controlled = {
                    control: _empty_oof_prediction(len(train.events)) for control in CONTROL_NAMES
                }
                for fold, held in enumerate(folds):
                    state = staged[(fold, seed)]
                    start_hash = _head_state_hash(state)
                    fold_seed = int(raw["train"]["optimization_seed_by_fold"][fold])
                    effective_seed = fold_seed + seed
                    rng = _rng(effective_seed)
                    model = _fresh_arm_model(
                        checkpoints[seed],
                        state,
                        v48_config=args.v48_config,
                        arm=arm,
                        arm_config=raw["arms"][arm],
                        device=device,
                    )
                    model.train()
                    batch_size = int(raw["arms"][arm]["batch_size"])
                    optimizer = torch.optim.AdamW(
                        model.head_parameters(),
                        lr=float(raw["train"]["learning_rate"]),
                        weight_decay=float(raw["train"]["weight_decay"]),
                    )
                    fit = np.setdiff1d(np.arange(len(train.events)), held)
                    for _ in range(int(raw["train"]["final_epochs"])):
                        order = fit.copy()
                        rng.shuffle(order)
                        for start in range(0, len(order), batch_size):
                            index = torch.as_tensor(
                                order[start : start + batch_size], dtype=torch.long
                            )
                            event = train.events[index].to(device, torch.float32)
                            out = model(event)
                            consensus = consensus_cache.for_indices(index, device)
                            loss, _ = object_event_v4_30_loss(
                                out,
                                train.delta_t_s[index].to(device),
                                train.target_ttc_s[index].to(device),
                                consensus_posteriors=consensus,
                                sequence_ids=[train.sequence_ids[int(i)] for i in index],
                                track_ids=[train.track_ids[int(i)] for i in index],
                                config=ObjectEventV430LossConfig(arm=arm, **raw["loss"]),
                                visible_heights_px=train.visible_heights_px[index].to(device),
                                boxes_xyxy=_t1_t2_boxes(train.boxes_xyxy[index]).to(device),
                                image_height=int(train.source_height),
                                image_width=int(train.source_width),
                            )
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("nonfinite v4.30 geometry loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.head_parameters(),
                                float(raw["train"]["max_grad_norm"]),
                                error_if_nonfinite=True,
                            )
                            optimizer.step()
                    records.append(
                        {
                            "stage": "geometry",
                            "arm": arm,
                            "fold": fold,
                            "seed": seed,
                            "checkpoint_seed": seed,
                            "effective_seed": effective_seed,
                            "starting_head_sha256": start_hash,
                            "teacher_consensus_cache": cache_audit,
                            "training_controls": {
                                "epochs": int(raw["train"]["final_epochs"]),
                                "batch_size": batch_size,
                                "learning_rate": float(raw["train"]["learning_rate"]),
                                "weight_decay": float(raw["train"]["weight_decay"]),
                                "max_grad_norm": float(raw["train"]["max_grad_norm"]),
                                "fold_seed": fold_seed,
                                "effective_seed": effective_seed,
                            },
                        }
                    )
                    held_split = _subset_split(train, held)
                    output = _predict(model.eval(), held_split, device, batch_size=batch_size)
                    _assert_nonzero_unperturbed(output)
                    _accumulate_oof(
                        unperturbed,
                        coverage,
                        held,
                        output,
                        context=f"{arm}/seed{seed}/fold{fold}/unperturbed",
                    )
                    fold_controls: dict[str, dict[str, np.ndarray]] = {}
                    for control in CONTROL_NAMES:
                        control_output = _predict(
                            model,
                            held_split,
                            device,
                            batch_size=batch_size,
                            control=control,
                            controls=raw["controls"],
                        )
                        if control == "zero_event":
                            _assert_zero_event_contract(control_output)
                        destination, control_coverage = controlled[control]
                        _accumulate_oof(
                            destination,
                            control_coverage,
                            held,
                            control_output,
                            context=f"{arm}/seed{seed}/fold{fold}/{control}",
                        )
                        fold_controls[control] = control_output
                    fold_metadata = {key: value[held] for key, value in metadata.items()}
                    _save_oof_npz(
                        target / f"oof_{arm}_seed{seed}_fold{fold}_controls.npz",
                        fold_metadata,
                        output,
                        fold_controls,
                    )
                _assert_complete_coverage(coverage, context=f"{arm}/seed{seed}/unperturbed")
                for control, (_, control_coverage) in controlled.items():
                    _assert_complete_coverage(
                        control_coverage, context=f"{arm}/seed{seed}/{control}"
                    )
                _assert_nonzero_unperturbed(unperturbed)
                _assert_zero_event_contract(controlled["zero_event"][0])
                seed_payloads.append(unperturbed)
                seed_controls.append({control: values[0] for control, values in controlled.items()})
                _save_oof_npz(
                    target / f"oof_{arm}_seed{seed}.npz",
                    metadata,
                    unperturbed,
                    seed_controls[-1],
                )
            arms[arm] = _aggregate_arm_oof(seed_payloads, seed_controls, metadata)
        _inject_arm_b_gains(arms)
        median_metrics = {arm: values["median_metrics"] for arm, values in arms.items()}
        summary = make_summary(
            raw=raw,
            checkpoint_hashes=checkpoint_hashes,
            cache_manifest=args.cache_manifest,
            stabilization=stabilization,
            arm_metrics=median_metrics,
            constituent_metrics={
                arm: values["constituent_metrics"] for arm, values in arms.items()
            },
            constituent_gate_checks={
                arm: values["constituent_gate_checks"] for arm, values in arms.items()
            },
            median_gate_checks={arm: values["median_gate_checks"] for arm, values in arms.items()},
            arm_passed={arm: bool(values["arm_passed"]) for arm, values in arms.items()},
            teacher_consensus_cache=cache_audit,
            config_path=args.config,
            v48_config_path=args.v48_config,
            v429_summary_path=args.v429_summary,
            diagnostic_only=diagnostic,
        )
        summary.update(
            {"records": records, "train_manifest": manifest, "diagnostic_only": diagnostic}
        )
        champion = summary.get("promoted_champion")
        if _development_requested(summary, diagnostic=diagnostic):
            assert isinstance(champion, str)
            development = _run_development_validation(
                train=train,
                manifest=manifest,
                checkpoints=checkpoints,
                consensus_cache=consensus_cache,
                raw=raw,
                champion=champion,
                v48_config=args.v48_config,
                cache_manifest=args.cache_manifest,
                v410_summary_path=args.v410_summary,
                ensemble_validation_path=args.ensemble_validation,
                base_input_size=base.input_size,
                device=device,
                output=target,
            )
            summary.update(development)
            summary["next_action"] = None
            summary["scientific_contract"].update(
                {
                    "development_validation_materialized_at_most_once_after_oof_champion": True,
                    "official_eap_test_opened": False,
                    "evttc_opened": False,
                }
            )
    (target / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rank_winner": summary.get("rank_winner"),
                "promoted_champion": summary.get("promoted_champion"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Replay frozen V8 A5/C2F checkpoints; never reuse historical prediction CSVs.

The input is a signed, portable event-only payload materialized from the frozen
V8 cache.  It deliberately contains endpoint tensors and row identities, not
predictions.  Every counterfactual below calls the loaded checkpoint again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (  # noqa: E402
    FACTORIAL_A5_CELLS,
    assert_causal_prefix_invariance,
    canonical_json_sha256,
    load_causal_scale_replay_checkpoint,
    replay_factorial_a5,
    replay_output_frame,
    sha256_file,
    validate_causal_scale_replay_input,
    validate_counterfactual_identity,
    validate_oof_frame,  # noqa: E402
)
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleReplayControl,
    CausalScaleTTC,
    CausalScaleTTCConfig,
    CausalScaleTTCOutput,
)
from e_jepa_ttc.models.garl_ttc_replica import (  # noqa: E402
    GarlTTCConfig,
    GarlTTCOutput,
    GarlTTCReplica,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def _checkpoint_config_provenance(checkpoint: Path, model: CausalScaleTTC) -> dict[str, str]:
    """Bind both historical raw config and current effective config.

    Older A5/C2F checkpoints may predate fields later added to
    ``CausalScaleTTCConfig`` with backward-compatible defaults.  Hashing
    ``model.checkpoint_config()`` and comparing it directly to the raw checkpoint
    config therefore creates a false mismatch.  We preserve the raw hash for
    exact provenance, separately hash the effective/default-expanded config, and
    require the loaded model to equal that deterministic expansion.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_config"), dict):
        raise ValueError("causal-scale replay checkpoint lacks a raw model_config")
    raw_config = dict(payload["model_config"])
    expected_effective = CausalScaleTTCConfig(**raw_config)
    effective_config = model.checkpoint_config()
    expected_effective_config = expected_effective.__dict__.copy()
    if effective_config != expected_effective_config:
        raise ValueError(
            "loaded model effective config differs from checkpoint config expanded with current defaults"
        )
    return {
        "raw_checkpoint_config_sha256": canonical_json_sha256(raw_config),
        "effective_model_config_sha256": canonical_json_sha256(effective_config),
    }


def _signed_json(path: Path) -> dict[str, Any]:
    from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"unsigned or malformed artifact: {path}")
    return value


def _control_replay(
    model: CausalScaleTTC, events: torch.Tensor, delta_t_s: torch.Tensor, *, name: str
) -> CausalScaleTTCOutput:
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleReplayControl

    controls = {
        "residual_zero": CausalScaleReplayControl(residual_enabled=False),
        "transport_zero": CausalScaleReplayControl(transport_enabled=False),
        "pair_current_only": CausalScaleReplayControl(temporal_blend="current_only"),
        "pair_previous_only": CausalScaleReplayControl(temporal_blend="previous_only"),
        "blend_neutral": CausalScaleReplayControl(temporal_blend="neutral"),
    }
    if name not in controls:
        raise ValueError(f"unknown replay control {name}")
    with torch.inference_mode():
        return model(events, delta_t_s, replay_control=controls[name])


def _garl_frame(
    output: GarlTTCOutput,
    payload: dict[str, Any],
    *,
    config_sha256: str,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(output.ttc_seconds.detach().float().cpu().tolist()):
        finite = bool(torch.isfinite(output.ttc_seconds[index]).detach().cpu())
        rows.append(
            {
                "token_id": payload["token_id"][index],
                "sequence_id": payload["sequence_id"][index],
                "track_id": payload["track_id"][index],
                "outer_fold": payload["outer_fold"][index],
                "seed": payload["seed"][index],
                "target_ttc": float(payload["target_ttc"][index]),
                "sample_weight": float(payload["sample_weight"][index]),
                "prediction_ttc": float(prediction) if finite else float("nan"),
                "prediction_log_variance": 0.0 if finite else float("nan"),
                "finite": finite,
                "failure_reason": "" if finite else "non_finite_garl_checkpoint_output",
                "event_count": int(torch.count_nonzero(payload["events"][index]).cpu()),
                "event_rate": float(torch.count_nonzero(payload["events"][index]).cpu()),
                "support_ms": float(payload["garl_delta_t_s"][index] * 1_000.0),
                "model_name": "garl",
                "config_sha256": config_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    return validate_oof_frame(pd.DataFrame(rows))


def run_garl_replay(
    *, checkpoint: Path, replay_input: Path, output_dir: Path, config_sha256: str, device_name: str
) -> dict[str, Any]:
    """Replay a real local Garl replica checkpoint over the frozen V8 row payload."""

    raw = torch.load(replay_input, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Garl replay input must be a mapping")
    payload = validate_causal_scale_replay_input(raw)
    if "garl_event_roi" not in raw or "garl_delta_t_s" not in raw:
        raise ValueError(
            "Garl replay requires garl_event_roi and garl_delta_t_s in the cache payload"
        )
    device = torch.device(device_name)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict) or not isinstance(
        checkpoint_payload.get("model_config"), dict
    ):
        raise ValueError("Garl replay checkpoint lacks model_config")
    model = GarlTTCReplica(GarlTTCConfig(**checkpoint_payload["model_config"]))
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    model.to(device).eval()
    event_roi = torch.as_tensor(raw["garl_event_roi"], dtype=torch.float32, device=device)
    elapsed = torch.as_tensor(raw["garl_delta_t_s"], dtype=torch.float32, device=device).reshape(-1)
    if event_roi.shape[0] != len(payload["token_id"]) or elapsed.shape != (event_roi.shape[0],):
        raise ValueError("Garl replay tensors do not match the frozen identity population")
    with torch.inference_mode():
        output = model(event_roi, elapsed)
    payload["garl_delta_t_s"] = elapsed.detach().cpu()
    frame = _garl_frame(
        output, payload, config_sha256=config_sha256, checkpoint_sha256=sha256_file(checkpoint)
    )
    path = output_dir / "baseline.csv"
    csv_sha256 = _save_frame(path, frame)
    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_garl_replay_v1",
        "status": "completed_replay_without_optimizer_steps",
        "model_name": "garl",
        "checkpoint": {"path": checkpoint.as_posix(), "sha256": sha256_file(checkpoint)},
        "replay_input": {"path": replay_input.as_posix(), "sha256": sha256_file(replay_input)},
        "config_sha256": config_sha256,
        "device": str(device),
        "causality_checks": {"optimizer_steps": 0, "two_endpoint_contract": True},
        "interventions": {"baseline": {"path": path.name, "sha256": csv_sha256}},
    }
    sign_artifact(manifest)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _run_causal_batch_frames(
    *,
    model: CausalScaleTTC,
    payload_raw: dict[str, Any],
    model_name: str,
    config_sha256: str,
    checkpoint_sha256: str,
    device: torch.device,
    dropout_levels: tuple[float, ...],
    dropout_seed_offset: int = 0,
) -> tuple[dict[str, pd.DataFrame], bool]:
    """Replay one bounded batch and immediately move every intervention to CPU.

    This is deliberately streaming: no fold-sized event tensor and no collection
    of many GPU-resident CausalScaleTTCOutput objects is retained.
    """

    payload = validate_causal_scale_replay_input(payload_raw)
    events = payload["events"].to(device=device, dtype=torch.float32, non_blocking=False)
    delta_t_s = payload["delta_t_s"].to(device=device, dtype=torch.float32, non_blocking=False)
    if events.shape[2] != model.config.in_channels:
        raise ValueError("replay input channel count differs from checkpoint model_config")
    if events.shape[1] != 3:
        raise ValueError("V8 A5/C2F autopsy requires exactly three causal endpoints")

    assert_causal_prefix_invariance(model, events, delta_t_s)
    rollback_rejected = False
    if "endpoint_us" in payload:
        rollback_payload = dict(payload)
        rollback_endpoints = payload["endpoint_us"].clone()
        rollback_endpoints[:, -1] = rollback_endpoints[:, -2]
        rollback_payload["endpoint_us"] = rollback_endpoints
        try:
            validate_causal_scale_replay_input(rollback_payload)
        except ValueError:
            rollback_rejected = True
        if not rollback_rejected:
            raise RuntimeError("timestamp rollback fixture was accepted by the replay contract")

    frames: dict[str, pd.DataFrame] = {}
    baseline_frame: pd.DataFrame | None = None

    def capture(name: str, output: CausalScaleTTCOutput) -> None:
        nonlocal baseline_frame
        frame = replay_output_frame(
            output,
            payload,
            model_name=model_name,
            config_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
        if baseline_frame is None:
            if name != "baseline":
                raise RuntimeError("baseline must be captured before counterfactuals")
            baseline_frame = frame
        else:
            validate_counterfactual_identity(baseline_frame, frame)
        frames[name] = frame
        del output

    with torch.inference_mode():
        capture("baseline", model(events, delta_t_s))
        for cell in FACTORIAL_A5_CELLS:
            capture(
                f"factorial_{cell.name}",
                model(
                    events,
                    delta_t_s,
                    replay_control=CausalScaleReplayControl(
                        residual_enabled=cell.residual_enabled,
                        transport_enabled=cell.transport_enabled,
                        temporal_blend=cell.temporal_blend,  # type: ignore[arg-type]
                    ),
                ),
            )
        for name, control in (
            ("residual_zero", CausalScaleReplayControl(residual_enabled=False)),
            ("transport_zero", CausalScaleReplayControl(transport_enabled=False)),
            ("pair_current_only", CausalScaleReplayControl(temporal_blend="current_only")),
            ("pair_previous_only", CausalScaleReplayControl(temporal_blend="previous_only")),
            ("blend_neutral", CausalScaleReplayControl(temporal_blend="neutral")),
        ):
            capture(name, model(events, delta_t_s, replay_control=control))
        capture("events_zero", model(torch.zeros_like(events), delta_t_s))
        capture("temporal_order_reversed", model(events.flip(1), delta_t_s.flip(1)))
        capture("spatial_permutation", model(events.flip(-1), delta_t_s))
        for level in dropout_levels:
            generator = torch.Generator(device=events.device).manual_seed(
                8_000 + int(round(level * 1_000)) + int(dropout_seed_offset)
            )
            keep = torch.rand(events.shape, device=events.device, generator=generator) >= level
            capture(
                f"event_dropout_{int(round(level * 100)):02d}",
                model(events * keep.to(events.dtype), delta_t_s),
            )

    del events, delta_t_s
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frames, rollback_rejected


def run_replay_sharded(
    *,
    checkpoint: Path,
    input_manifest_path: Path,
    output_dir: Path,
    model_name: str,
    config_sha256: str,
    device_name: str = "cpu",
    dropout_levels: tuple[float, ...] = (0.1, 0.3, 0.5),
) -> dict[str, Any]:
    """Replay a signed fold manifest part-by-part with bounded CPU/GPU memory."""

    input_manifest = _signed_json(input_manifest_path)
    if input_manifest.get("artifact_type") != "scientific_recovery_v8_autopsy_replay_input_sharded_v2":
        raise ValueError("run_replay_sharded requires a V8 sharded replay-input manifest")
    parts = input_manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("sharded replay-input manifest has no parts")

    device = torch.device(device_name)
    model = load_causal_scale_replay_checkpoint(checkpoint, device=device)
    checkpoint_sha256 = sha256_file(checkpoint)
    config_provenance = _checkpoint_config_provenance(checkpoint, model)
    if config_sha256 != config_provenance["raw_checkpoint_config_sha256"]:
        raise ValueError(
            "declared config_sha256 does not match the raw model_config stored in the checkpoint"
        )

    intervention_frames: dict[str, list[pd.DataFrame]] = {}
    rollback_checks: list[bool] = []
    total_rows = 0
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError("sharded replay-input part must be a mapping")
        part_path = input_manifest_path.parent / str(part["path"])
        if not part_path.is_file() or sha256_file(part_path) != str(part.get("sha256")):
            raise ValueError(f"replay-input part hash mismatch: {part_path}")
        raw = torch.load(part_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(f"replay-input part is not a mapping: {part_path}")
        validated = validate_causal_scale_replay_input(raw)
        expected_part_rows = int(part.get("rows", -1))
        if len(validated["token_id"]) != expected_part_rows:
            raise ValueError(f"replay-input part row count mismatch: {part_path}")
        frames, rollback_rejected = _run_causal_batch_frames(
            model=model,
            payload_raw=raw,
            model_name=model_name,
            config_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
            dropout_levels=dropout_levels,
            dropout_seed_offset=part_index * 10_000,
        )
        for intervention, frame in frames.items():
            intervention_frames.setdefault(intervention, []).append(frame)
        rollback_checks.append(rollback_rejected)
        total_rows += expected_part_rows
        del raw, validated, frames

    if total_rows != int(input_manifest.get("rows", -1)):
        raise ValueError("sharded replay-input total row count mismatch")

    manifest_rows: dict[str, Any] = {}
    baseline: pd.DataFrame | None = None
    for intervention, chunks in intervention_frames.items():
        frame = validate_oof_frame(pd.concat(chunks, ignore_index=True))
        if baseline is None:
            if intervention != "baseline":
                raise RuntimeError("baseline intervention missing or unordered")
            baseline = frame
        else:
            validate_counterfactual_identity(baseline, frame)
        path = output_dir / f"{intervention}.csv"
        manifest_rows[intervention] = {
            "path": path.name,
            "sha256": _save_frame(path, frame),
            "rows": len(frame),
        }
    factorial = {f"factorial_{cell.name}" for cell in FACTORIAL_A5_CELLS}
    if not factorial.issubset(manifest_rows):
        raise RuntimeError("replay did not produce all five frozen A5 factorial cells")

    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_mechanism_replay_v2_sharded",
        "status": "completed_replay_without_optimizer_steps",
        "model_name": model_name,
        "checkpoint": {"path": checkpoint.as_posix(), "sha256": checkpoint_sha256},
        "replay_input": {
            "manifest": input_manifest_path.as_posix(),
            "manifest_sha256": sha256_file(input_manifest_path),
            "parts": len(parts),
            "rows": total_rows,
        },
        "config_sha256": config_sha256,
        "config_provenance": config_provenance,
        "device": str(device),
        "causality_checks": {
            "future_prefix_invariance": True,
            "future_prefix_invariance_method": "same_shape_future_endpoint_and_interval_perturbation_v1",
            "timestamp_rollback_rejected": bool(rollback_checks) and all(rollback_checks),
            "state_reset_contract": "not_applicable_stateless_checkpoint",
            "optimizer_steps": 0,
        },
        "factorial_cells": [cell.name for cell in FACTORIAL_A5_CELLS],
        "interventions": manifest_rows,
    }
    sign_artifact(manifest)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def run_protocol_replays(
    *,
    protocol_path: Path,
    replay_input_root: Path,
    output_dir: Path,
    models: tuple[str, ...],
    device_name: str,
) -> dict[str, Any]:
    """Resolve all parent checkpoints/folds from the signed V8 protocol."""

    protocol = _signed_json(protocol_path)
    if protocol.get("status") != "frozen_before_v8_training":
        raise ValueError("canonical V8 replay requires the frozen pre-training protocol")
    parents = protocol.get("parent_checkpoints")
    if not isinstance(parents, dict):
        raise ValueError("V8 protocol lacks parent_checkpoints")
    expected_rows = protocol.get("sample_contract", {}).get("row_count_contract", {}).get("by_outer_fold")
    if not isinstance(expected_rows, dict):
        raise ValueError("V8 protocol lacks frozen per-fold row counts")
    result: dict[str, Any] = {}
    for model_name in models:
        checkpoints = parents.get(model_name)
        if not isinstance(checkpoints, list) or len(checkpoints) != 3:
            raise ValueError(f"protocol lacks exactly three parent checkpoints for {model_name}")
        combined: dict[str, list[pd.DataFrame]] = {}
        fold_sources: list[dict[str, Any]] = []
        for item in sorted(checkpoints, key=lambda value: int(value["fold"])):
            fold = int(item["fold"])
            checkpoint = ROOT / str(item["path"])
            if not checkpoint.is_file() or sha256_file(checkpoint) != item.get("sha256"):
                raise ValueError(f"checkpoint hash mismatch or missing: {checkpoint}")
            sidecar = replay_input_root / f"fold{fold}.manifest.json"
            input_manifest = _signed_json(sidecar)
            if (
                input_manifest.get("protocol_artifact_sha256") != protocol["artifact_sha256"]
                or int(input_manifest.get("outer_fold", -1)) != fold
                or int(input_manifest.get("rows", -1)) != int(expected_rows[str(fold)])
            ):
                raise ValueError(f"signed replay input binding mismatch for fold {fold}")
            config_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(config_payload, dict) or not isinstance(config_payload.get("model_config"), dict):
                raise ValueError(f"checkpoint has no model_config: {checkpoint}")
            fold_output = output_dir / model_name / f"fold{fold}"
            if input_manifest.get("artifact_type") == "scientific_recovery_v8_autopsy_replay_input_sharded_v2":
                manifest = run_replay_sharded(
                    checkpoint=checkpoint,
                    input_manifest_path=sidecar,
                    output_dir=fold_output,
                    model_name=model_name,
                    config_sha256=canonical_json_sha256(config_payload["model_config"]),
                    device_name=device_name,
                )
            else:
                replay_input = replay_input_root / f"fold{fold}.pt"
                if (
                    input_manifest.get("input_sha256") != sha256_file(replay_input)
                ):
                    raise ValueError(f"legacy replay input binding mismatch for fold {fold}")
                payload = torch.load(replay_input, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict):
                    raise ValueError(f"replay input is not a mapping: {replay_input}")
                validated = validate_causal_scale_replay_input(payload)
                if len(validated["token_id"]) != int(expected_rows[str(fold)]):
                    raise ValueError(f"replay input row count differs from frozen fold {fold}")
                if set(validated["outer_fold"]) != {fold}:
                    raise ValueError(f"replay input contains rows outside frozen outer fold {fold}")
                manifest = run_replay(
                    checkpoint=checkpoint,
                    replay_input=replay_input,
                    output_dir=fold_output,
                    model_name=model_name,
                    config_sha256=canonical_json_sha256(config_payload["model_config"]),
                    device_name=device_name,
                )
            for intervention, source in manifest["interventions"].items():
                frame = pd.read_csv(fold_output / source["path"])
                combined.setdefault(intervention, []).append(frame)
            fold_sources.append(
                {
                    "fold": fold,
                    "checkpoint_sha256": item["sha256"],
                    "input_manifest": str(sidecar.relative_to(ROOT)),
                    "replay_manifest": str((fold_output / "manifest.json").relative_to(ROOT)),
                }
            )
        root = output_dir / model_name
        interventions: dict[str, Any] = {}
        for intervention, frames in combined.items():
            frame = pd.concat(frames, ignore_index=True)
            path = root / f"{intervention}.csv"
            interventions[intervention] = {"path": path.name, "sha256": _save_frame(path, frame)}
        causal_manifests = [
            _signed_json(ROOT / source["replay_manifest"])
            for source in fold_sources
            if model_name != "garl"
        ]
        manifest = {
            "artifact_type": "scientific_recovery_v8_combined_mechanism_replay_v2",
            "status": "completed_replay_without_optimizer_steps",
            "model_name": model_name,
            "protocol_artifact_sha256": protocol["artifact_sha256"],
            "fold_sources": fold_sources,
            "interventions": interventions,
            "causality_checks": {
                "future_prefix_invariance": bool(causal_manifests)
                and all(source["causality_checks"].get("future_prefix_invariance") is True for source in causal_manifests),
                "future_prefix_invariance_method": (
                    "same_shape_future_endpoint_and_interval_perturbation_v1"
                    if causal_manifests else "not_applicable_garl_pair_checkpoint"
                ),
                "timestamp_rollback_rejected": bool(causal_manifests)
                and all(source["causality_checks"].get("timestamp_rollback_rejected") is True for source in causal_manifests),
                "state_reset_contract": (
                    "not_applicable_stateless_checkpoint" if causal_manifests else "not_applicable_garl_pair_checkpoint"
                ),
            },
        }
        sign_artifact(manifest)
        _atomic_json(root / "manifest.json", manifest)
        result[model_name] = {"path": str((root / "manifest.json").relative_to(ROOT))}
    return result


def run_replay(
    *,
    checkpoint: Path,
    replay_input: Path,
    output_dir: Path,
    model_name: str,
    config_sha256: str,
    device_name: str = "cpu",
    dropout_levels: tuple[float, ...] = (0.1, 0.3, 0.5),
) -> dict[str, Any]:
    """Run all frozen A5/C2F interventions for one small portable payload."""

    device = torch.device(device_name)
    model = load_causal_scale_replay_checkpoint(checkpoint, device=device)
    payload_raw = torch.load(replay_input, map_location="cpu", weights_only=False)
    if not isinstance(payload_raw, dict):
        raise ValueError("replay input must be a torch mapping materialized from the V8 cache")
    checkpoint_sha256 = sha256_file(checkpoint)
    input_sha256 = sha256_file(replay_input)
    config_provenance = _checkpoint_config_provenance(checkpoint, model)
    if config_sha256 != config_provenance["raw_checkpoint_config_sha256"]:
        raise ValueError(
            "declared config_sha256 does not match the raw model_config stored in the checkpoint"
        )
    frames, rollback_rejected = _run_causal_batch_frames(
        model=model,
        payload_raw=payload_raw,
        model_name=model_name,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        dropout_levels=dropout_levels,
    )
    manifest_rows: dict[str, Any] = {}
    for intervention, frame in frames.items():
        path = output_dir / f"{intervention}.csv"
        manifest_rows[intervention] = {"path": path.name, "sha256": _save_frame(path, frame)}
    factorial = {f"factorial_{cell.name}" for cell in FACTORIAL_A5_CELLS}
    if not factorial.issubset(manifest_rows):
        raise RuntimeError("replay did not produce all five frozen A5 factorial cells")
    manifest = {
        "artifact_type": "scientific_recovery_v8_mechanism_replay_v1",
        "status": "completed_replay_without_optimizer_steps",
        "model_name": model_name,
        "checkpoint": {"path": checkpoint.as_posix(), "sha256": checkpoint_sha256},
        "replay_input": {"path": replay_input.as_posix(), "sha256": input_sha256},
        "config_sha256": config_sha256,
        "config_provenance": config_provenance,
        "device": str(device),
        "causality_checks": {
            "future_prefix_invariance": True,
            "future_prefix_invariance_method": "same_shape_future_endpoint_and_interval_perturbation_v1",
            "timestamp_rollback_rejected": rollback_rejected,
            "state_reset_contract": "not_applicable_stateless_checkpoint",
            "optimizer_steps": 0,
        },
        "factorial_cells": [cell.name for cell in FACTORIAL_A5_CELLS],
        "interventions": manifest_rows,
    }
    sign_artifact(manifest)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--replay-input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=("a5", "c2f", "garl"))
    parser.add_argument("--config-sha256")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--replay-input-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=("a5", "c2f", "garl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.protocol is not None:
        if args.replay_input_root is None or not args.models:
            raise ValueError("--protocol requires --replay-input-root and --models")
        if args.dry_run:
            protocol = _signed_json(args.protocol)
            print(
                json.dumps(
                    {"status": "validated_protocol", "protocol": protocol["artifact_sha256"]}
                )
            )
            return
        result = run_protocol_replays(
            protocol_path=args.protocol,
            replay_input_root=args.replay_input_root,
            output_dir=args.output_dir,
            models=tuple(args.models),
            device_name=args.device,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if (
        args.checkpoint is None
        or args.replay_input is None
        or args.model_name is None
        or not args.config_sha256
    ):
        raise ValueError(
            "single-model mode requires --checkpoint --replay-input --model-name --config-sha256"
        )
    if args.dry_run:
        payload = torch.load(args.replay_input, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("replay input must be a mapping")
        validate_causal_scale_replay_input(payload)
        print(
            json.dumps({"status": "validated_no_inference", "replay_input": str(args.replay_input)})
        )
        return
    result = (
        run_garl_replay(
            checkpoint=args.checkpoint,
            replay_input=args.replay_input,
            output_dir=args.output_dir,
            config_sha256=args.config_sha256,
            device_name=args.device,
        )
        if args.model_name == "garl"
        else run_replay(
            checkpoint=args.checkpoint,
            replay_input=args.replay_input,
            output_dir=args.output_dir,
            model_name=args.model_name,
            config_sha256=args.config_sha256,
            device_name=args.device,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Freeze preregistered A5 runtime configs without confounding the causal screen.

Scientific contract
-------------------
* The 2048-row A5-CORR seed7/13/23 and CAP-S/CAP-M screens keep the exact A4
  endpoint-DINO weight (lambda=4).  This is required for a clean causal
  comparison against the immutable A4 seed7 result.
* A train-only lambda selected by the 8192-row A4 CV may be supplied only for
  8192/16384 scale follow-ups.  Those runs must be compared against an A4-S1
  control trained at the same lambda when making a transport claim.
* Validation remains the original frozen 2048-row manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCREEN_A4_LAMBDA = 4.0
TEMPLATES = {
    "seed7": ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_corr_v1.yaml",
    "seed13": ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_corr_v1_seed13.yaml",
    "seed23": ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_corr_v1_seed23.yaml",
    "cap_s": ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_cap_s_v1.yaml",
    "cap_m": ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_cap_m_v1.yaml",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )




def _load_transport_selection(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    artifact_type = payload.get("artifact_type")
    allowed_types = {
        "a5_transport_preflight_train_only_v2",
        "a5_transport_preflight_train_only_v3_confirmation",
    }
    if artifact_type not in allowed_types:
        raise ValueError("--preflight-selection must be an authorized A5 transport preflight artifact")
    scope = payload.get("scope", {})
    decision = payload.get("decision", {})
    if (
        scope.get("public_train_only") is not True
        or scope.get("validation_or_test_opened") is not False
        or int(scope.get("optimizer_steps", -1)) != 0
    ):
        raise ValueError("A5 preflight did not satisfy train-only zero-training contract")
    if decision.get("a5_corr_authorized") is not True:
        raise ValueError("A5 preflight did not authorize A5-CORR")
    radius = int(decision.get("selected_radius", -1))
    temperature = float(decision.get("selected_temperature", float("nan")))
    if radius not in (1, 2, 4):
        raise ValueError(f"invalid selected transport radius: {radius}")
    if temperature not in (0.02, 0.04, 0.07, 0.10):
        raise ValueError(f"invalid selected transport temperature: {temperature}")
    return {
        "path": path,
        "artifact_sha256": payload.get("artifact_sha256"),
        "file_sha256": _sha(path),
        "radius": radius,
        "temperature": temperature,
        "decision": decision,
        "artifact_type": artifact_type,
    }


def _apply_transport_selection(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    name: str,
    selection: dict[str, Any] | None,
) -> str | None:
    if selection is None:
        return None
    model_path = (ROOT / str(payload["model_config"])).resolve(strict=True)
    model_payload = _read_yaml(model_path)
    if model_payload.get("transport_enabled") is not True:
        raise ValueError(f"A5 runtime model must enable transport: {model_path}")
    model_payload["transport_radius"] = int(selection["radius"])
    model_payload["transport_temperature"] = float(selection["temperature"])
    model_target = output_dir / f"model_{name}.yaml"
    _write(model_target, model_payload)
    payload["model_config"] = model_target.relative_to(ROOT).as_posix()

    contract = payload.get("decision_contract")
    if not isinstance(contract, dict):
        raise ValueError("A5 template lacks decision_contract")
    change = contract.get("representation_change")
    if isinstance(change, dict):
        radius = int(selection["radius"])
        change["transport_radius"] = radius
        change["transport_candidates_per_position"] = (2 * radius + 1) ** 2
        change["transport_temperature"] = float(selection["temperature"])
    contract["preflight_contract"] = {
        "artifact_type": selection["artifact_type"],
        "artifact": _repo_path(selection["path"]),
        "artifact_sha256": selection.get("artifact_sha256"),
        "file_sha256": selection["file_sha256"],
        "selected_radius": int(selection["radius"]),
        "selected_temperature": float(selection["temperature"]),
        "selection_rule": (
            "v3_disjoint_confirmation_of_v2_soft_epe_candidate"
            if selection["artifact_type"] == "a5_transport_preflight_train_only_v3_confirmation"
            else "smallest_physics_covered_radius_surviving_two_nulls_then_largest_safe_tau"
        ),
        "public_train_only": True,
        "optimizer_steps": 0,
        "a5_training_requires_preflight_pass": True,
    }
    contract["transport_radius_selected_before_a5_validation"] = True
    contract["transport_temperature_selected_before_a5_validation"] = True
    return model_target.relative_to(ROOT).as_posix()

def _freeze_screen_lambda(payload: dict[str, Any]) -> None:
    training = payload.get("training")
    contract = payload.get("decision_contract")
    if not isinstance(training, dict) or not isinstance(contract, dict):
        raise ValueError("template lacks training/decision_contract")
    observed = float(training.get("representation_distillation_weight", float("nan")))
    if not math.isclose(observed, SCREEN_A4_LAMBDA, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"A5 causal screen must preserve immutable A4 lambda={SCREEN_A4_LAMBDA}, got {observed}"
        )
    contract["a4_endpoint_dino_lambda_source"] = "immutable_A4_seed7_screen"
    contract["a4_endpoint_dino_lambda_frozen"] = SCREEN_A4_LAMBDA
    contract["lambda_cv_must_not_change_2048_causal_screen"] = True


def _freeze_scale_lambda(payload: dict[str, Any], dino_lambda: float) -> None:
    training = payload.get("training")
    contract = payload.get("decision_contract")
    if not isinstance(training, dict) or not isinstance(contract, dict):
        raise ValueError("template lacks training/decision_contract")
    training["representation_distillation_weight"] = float(dino_lambda)
    contract["a4_endpoint_dino_lambda_source"] = "train_only_A4_lambda_CV_selected_before_A5_scale_validation"
    contract["a4_endpoint_dino_lambda_frozen"] = float(dino_lambda)
    contract["scale_transport_claim_requires_A4_control_at_same_lambda"] = True


def _find_existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def _scale_config(
    base: dict[str, Any],
    *,
    rows: int,
    train_manifest: Path,
    teacher_manifest: Path,
    validation_manifest: Path,
    dino_lambda: float,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base))
    _freeze_scale_lambda(payload, dino_lambda)
    event_meta = _read_json(train_manifest)
    teacher_meta = _read_json(teacher_manifest)
    val_meta = _read_json(validation_manifest)
    data = payload["data"]
    experiment = payload["experiment"]
    contract = payload["decision_contract"]
    data["cache_manifest"] = train_manifest.relative_to(ROOT).as_posix()
    data["cache_manifest_sha256"] = _sha(train_manifest)
    data["cache_artifact_sha256"] = event_meta.get("artifact_sha256")
    data["expected_train_rows"] = rows
    data["validation_cache_manifest"] = validation_manifest.relative_to(ROOT).as_posix()
    data["validation_cache_manifest_sha256"] = _sha(validation_manifest)
    data["validation_cache_artifact_sha256"] = val_meta.get("artifact_sha256")
    data["expected_validation_rows"] = 2048
    data["dinov3_relational_teacher"] = {
        "manifest": teacher_manifest.relative_to(ROOT).as_posix(),
        "manifest_sha256": _sha(teacher_manifest),
        "artifact_sha256": teacher_meta.get("artifact_sha256"),
    }
    payload["training"]["representation_teacher_cache_artifact_sha256"] = teacher_meta.get("artifact_sha256")
    experiment["name"] = f"e_jepa_garl_event_causal_scale_a5_corr_v1_train{rows}_seed7"
    experiment["protocol_version"] = f"causal_scale_eap_a5_event_local_transport_train{rows}_v1"
    experiment["parent_arm"] = f"A4_S1_train{rows}_same_lambda_control_required_for_transport_claim"
    experiment["single_scientific_difference"] = "A5_local_transport_vs_same_scale_A4_control"
    contract["data_scale_rows"] = rows
    contract["frozen_validation_rows"] = 2048
    contract["scale_changes_model_architecture"] = False
    contract["scale_changes_transport_radius"] = False
    contract["scale_uses_new_rgb_teacher_only_for_train_rows"] = True
    contract["same_scale_A4_control_required"] = True
    return payload


def run(
    output_dir: Path,
    scale_dino_lambda: float | None,
    include_scale: bool,
    preflight_selection_path: Path | None = None,
) -> dict[str, Any]:
    if include_scale and scale_dino_lambda is None:
        raise ValueError("--include-scale requires --scale-dino-lambda from the train-only A4 CV")
    if scale_dino_lambda is not None and (
        not math.isfinite(scale_dino_lambda) or scale_dino_lambda <= 0
    ):
        raise ValueError("--scale-dino-lambda must be finite and positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    selection = _load_transport_selection(preflight_selection_path)
    written: dict[str, str] = {}
    frozen: dict[str, dict[str, Any]] = {}
    for name, template in TEMPLATES.items():
        payload = _read_yaml(template)
        _freeze_screen_lambda(payload)
        model_runtime = _apply_transport_selection(
            payload, output_dir=output_dir, name=name, selection=selection
        )
        if model_runtime is not None:
            written[f"model_{name}"] = model_runtime
        target = output_dir / f"{name}.yaml"
        _write(target, payload)
        written[name] = target.relative_to(ROOT).as_posix()
        frozen[name] = payload

    scale_status: dict[str, Any] = {}
    if include_scale:
        assert scale_dino_lambda is not None
        validation = ROOT / "artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json"
        specs = {
            8192: (
                [ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"],
                [ROOT / "artifacts/cache/dinov3_convnext_large_relational_a4_train8192_rgb_v1/manifest.json"],
            ),
            16384: (
                [
                    ROOT / "artifacts/cache/garl_object_event_common_roi_train16384_v1/manifest.json",
                    ROOT / "artifacts/cache/garl_object_event_common_roi_train16k_v1/manifest.json",
                ],
                [
                    ROOT / "artifacts/cache/dinov3_convnext_large_relational_a4_train16384_rgb_v1/manifest.json",
                    ROOT / "artifacts/cache/dinov3_convnext_large_relational_a4_train16k_rgb_v1/manifest.json",
                ],
            ),
        }
        for rows, (event_candidates, teacher_candidates) in specs.items():
            event_manifest = _find_existing(event_candidates)
            teacher_manifest = _find_existing(teacher_candidates)
            ready = validation.is_file() and event_manifest is not None and teacher_manifest is not None
            scale_status[str(rows)] = {"ready": ready}
            if not ready:
                continue
            scaled = _scale_config(
                frozen["seed7"],
                rows=rows,
                train_manifest=event_manifest,
                teacher_manifest=teacher_manifest,
                validation_manifest=validation,
                dino_lambda=scale_dino_lambda,
            )
            target = output_dir / f"scale_{rows}_seed7.yaml"
            _write(target, scaled)
            written[f"scale_{rows}"] = target.relative_to(ROOT).as_posix()
            scale_status[str(rows)].update({
                "event_manifest": event_manifest.relative_to(ROOT).as_posix(),
                "teacher_manifest": teacher_manifest.relative_to(ROOT).as_posix(),
                "config": target.relative_to(ROOT).as_posix(),
                "dino_lambda": scale_dino_lambda,
            })

    manifest = {
        "artifact_type": "a5_frozen_runtime_configs_v2",
        "screen_dino_endpoint_lambda": SCREEN_A4_LAMBDA,
        "screen_lambda_scope": "immutable_A4_comparator_for_clean_transport_attribution",
        "scale_dino_endpoint_lambda": scale_dino_lambda,
        "scale_lambda_scope": (
            "selected_train_only_before_A5_scale_validation" if scale_dino_lambda is not None else None
        ),
        "files": {name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in written.items()},
        "scale_status": scale_status,
        "transport_selection": (
            {
                "source": _repo_path(selection["path"]),
                "source_file_sha256": selection["file_sha256"],
                "source_artifact_sha256": selection.get("artifact_sha256"),
                "radius": selection["radius"],
                "temperature": selection["temperature"],
            } if selection is not None else None
        ),
        "private_test_opened": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-dino-lambda", type=float)
    parser.add_argument("--include-scale", action="store_true")
    parser.add_argument("--preflight-selection", type=Path)
    args = parser.parse_args()
    payload = run(
        args.output_dir.resolve(),
        args.scale_dino_lambda,
        args.include_scale,
        args.preflight_selection.resolve() if args.preflight_selection is not None else None,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

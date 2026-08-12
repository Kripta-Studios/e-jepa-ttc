#!/usr/bin/env python
"""Freeze A6-S1/A7-S1 configs from the already frozen A5-S1 contract.

This script never opens validation labels beyond paths already frozen in the source
config and never touches any private/test resource.  A6/A7 inherit the exact A5-S1
r=1, tau=.02, lambda=8, 8192/2048 data contract and initialize from A4-S1.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")


def model_config_kwargs(model_path: Path) -> dict[str, Any]:
    """Load a model YAML using the same risk-threshold normalization as training."""

    raw = read_yaml(model_path)
    raw.pop("model", None)
    thresholds = raw.get("risk_thresholds_s")
    if not isinstance(thresholds, list):
        raise ValueError("risk_thresholds_s must be a list")
    raw["risk_thresholds_s"] = tuple(float(value) for value in thresholds)
    return raw


def parameter_count(model_path: Path) -> int:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    raw = model_config_kwargs(model_path)
    return sum(p.numel() for p in CausalScaleTTC(CausalScaleTTCConfig(**raw)).parameters())


def mutate(
    base: dict[str, Any], *, arm: str, seed: int, checkpoint: Path, checkpoint_sha: str,
    num_workers: int, prefetch_factor: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["training"]["seed"] = seed
    cfg["training"]["num_workers"] = int(num_workers)
    cfg["training"]["prefetch_factor"] = int(prefetch_factor)
    cfg["training"]["foreground_warmup_epochs"] = 0
    cfg["training"]["initialization_checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
    cfg["training"]["initialization_checkpoint_sha256"] = checkpoint_sha
    cfg["training"]["initialization_mode"] = "shape_compatible"
    cfg["training"]["freeze_encoder"] = True

    change = cfg["decision_contract"]["representation_change"]
    change["transport_radius"] = 1
    change["transport_temperature"] = 0.02
    change["transport_candidates_per_position"] = 9

    common = {
        "initialization_mode": "shape_compatible",
        "initialization_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "initialization_checkpoint_sha256": checkpoint_sha,
        "parent_encoder_frozen_for_entire_run": True,
        "geometry_must_equal_parent_by_construction": True,
        "foreground_warmup_epochs": 0,
        "transport_radius": 1,
        "transport_temperature": 0.02,
        "dino_endpoint_lambda": 8.0,
        "train_rows": 8192,
        "validation_rows": 2048,
        "private_test_remains_closed": True,
    }
    if arm == "a6":
        model = ROOT / "configs/model/e_jepa_causal_scale_event_v10_transport_adapter_r1_t002_legacy.yaml"
        cfg["model_config"] = model.relative_to(ROOT).as_posix()
        cfg["experiment"]["name"] = f"e_jepa_garl_event_causal_scale_a6_s1_train8192_seed{seed}"
        cfg["experiment"]["protocol_version"] = "causal_scale_a6_s1_transport_adapter_train8192_v1"
        cfg["experiment"]["single_scientific_difference"] = "frozen_A4_S1_geometry_plus_identity_initialized_transport_adapter"
        change["type"] = "a4_frozen_endpoint_plus_adaptive_transport_adapter"
        cfg["decision_contract"]["adapter_contract"] = {
            **common,
            "transport_adapter_depth": 1,
            "adapter_is_transport_only": True,
            "adapter_identity_initialized": True,
        }
    elif arm == "a7":
        model = ROOT / "configs/model/e_jepa_causal_scale_event_v11_dual_transport_r1_t002_legacy.yaml"
        cfg["model_config"] = model.relative_to(ROOT).as_posix()
        cfg["experiment"]["name"] = f"e_jepa_garl_event_causal_scale_a7_s1_dual_transport_seed{seed}"
        cfg["experiment"]["protocol_version"] = "causal_scale_a7_s1_dual_transport_train8192_v1"
        cfg["experiment"]["single_scientific_difference"] = "frozen_A4_S1_geometry_plus_trainable_transport_encoder_copy"
        change["type"] = "a4_frozen_geometry_plus_trainable_transport_encoder"
        cfg["decision_contract"]["dual_stream_contract"] = {
            **common,
            "dual_stream_is_transport_only": True,
            "transport_encoder_initialized_from_parent": True,
            "primary_geometry_encoder_frozen": True,
            "transport_encoder_trainable": True,
            "transport_adapter_disabled": True,
        }
    else:
        raise ValueError(arm)
    cfg["decision_contract"]["expected_parameter_count"] = parameter_count(model)
    cfg["decision_contract"]["public_validation_does_not_authorize_sota_claim"] = True
    cfg["decision_contract"]["sota_requires_budget_matched_garl_and_sealed_test"] = True
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a5-s1-config", type=Path, required=True)
    ap.add_argument(
        "--a4-s1-checkpoint", type=Path,
        default=ROOT / "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed7/model_best.pt",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    args = ap.parse_args()
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("invalid DataLoader hardware profile")
    base_path = args.a5_s1_config.resolve()
    checkpoint = args.a4_s1_checkpoint.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base = read_yaml(base_path)
    if int(base.get("data", {}).get("expected_train_rows", -1)) != 8192:
        raise ValueError("A6/A7-S1 source must be the frozen A5 8192 config")
    if float(base.get("training", {}).get("representation_distillation_weight", -1)) != 8.0:
        raise ValueError("A6/A7-S1 require the train-only selected lambda=8")
    checkpoint_sha = sha(checkpoint)
    out = args.output_dir.resolve()
    files: dict[str, Any] = {}
    for arm in ("a6", "a7"):
        for seed in (7, 13, 23):
            payload = mutate(
                base, arm=arm, seed=seed, checkpoint=checkpoint, checkpoint_sha=checkpoint_sha,
                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
            )
            path = out / f"{arm}_s1_seed{seed}.yaml"
            write_yaml(path, payload)
            files[path.name] = {"sha256": sha(path), "parameter_count": payload["decision_contract"]["expected_parameter_count"]}
    manifest = {
        "artifact_type": "scientific_recovery_s1_frozen_configs_v1",
        "a5_s1_source": base_path.relative_to(ROOT).as_posix(),
        "a5_s1_source_sha256": sha(base_path),
        "a4_s1_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "a4_s1_checkpoint_sha256": checkpoint_sha,
        "files": files,
        "hardware_profile": {"num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor},
        "private_test_opened": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

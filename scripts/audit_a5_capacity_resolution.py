#!/usr/bin/env python
"""Static A5 capacity/resolution audit; no dataset access and no training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402

DEFAULT_MODELS = (
    ROOT / "configs/model/e_jepa_causal_scale_event_v8_t015_resize_conv.yaml",
    ROOT / "configs/model/e_jepa_causal_scale_event_v9_transport_r4.yaml",
    ROOT / "configs/model/e_jepa_causal_scale_event_v9_transport_cap_s.yaml",
    ROOT / "configs/model/e_jepa_causal_scale_event_v9_transport_cap_m.yaml",
)


def _model(path: Path) -> tuple[CausalScaleTTCConfig, int]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(path)
    raw.pop("model", None)
    raw["risk_thresholds_s"] = tuple(raw["risk_thresholds_s"])
    config = CausalScaleTTCConfig(**raw)
    count = sum(p.numel() for p in CausalScaleTTC(config).parameters())
    return config, count


def _cost_volume_mib(grid: int, radius: int, batch: int, pairs: int, bytes_per_value: int = 4) -> float:
    k = (2 * radius + 1) ** 2
    return batch * pairs * grid * grid * k * bytes_per_value / 2**20


def run(output: Path, *, garl_parameter_count: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for path in DEFAULT_MODELS:
        config, count = _model(path)
        models[path.name] = {
            "parameter_count": count,
            "hidden_dim": config.hidden_dim,
            "geometry_dim": config.geometry_dim,
            "residual_depth": config.residual_depth,
            "transport_enabled": config.transport_enabled,
            "transport_radius": config.transport_radius if config.transport_enabled else None,
            "vs_garl_parameter_ratio": count / float(garl_parameter_count),
        }
    payload: dict[str, Any] = {
        "artifact_type": "a5_capacity_resolution_audit_v1",
        "models": models,
        "garl_event_lhr_reference_parameter_count": garl_parameter_count,
        "cost_volume_fp32_mib": {
            "grid32_r4_batch32_two_pairs": _cost_volume_mib(32, 4, 32, 2),
            "grid64_r4_batch32_two_pairs": _cost_volume_mib(64, 4, 32, 2),
            "grid32_r2_batch32_two_pairs": _cost_volume_mib(32, 2, 32, 2),
        },
        "resolution_contract": {
            "current_student_dense_grid": [32, 32],
            "current_dino_relation_grid": [32, 32],
            "current_dino_input_size": 256,
            "cached_relations_can_be_upsampled_scientifically": False,
            "larger_dino_grid_requires_rematerialization_from_rgb": True,
            "student_64_grid_requires_new_encoder_or_high_resolution_branch": True,
        },
        "recommendation": {
            "first_a5_arm": "keep A4 width/depth and 32x32 grid; isolate transport",
            "post_gate_capacity_arm_order": ["cap_s", "cap_m"],
            "do_not_mix_larger_grid_with_first_capacity_arm": True,
            "larger_grid_next_only_if": "transport is useful but correspondence is spatially under-resolved",
            "reason": (
                "A4/A4D is ~0.355M params versus ~24.67M for Garl event-LHR, so capacity is a plausible "
                "secondary bottleneck; however increasing width/depth or DINO grid in A5-CORR-V1 would confound "
                "the causal test of p->q transport."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--garl-parameter-count", type=int, default=24674178)
    args = parser.parse_args()
    payload = run(args.output.resolve(), garl_parameter_count=args.garl_parameter_count)
    print(json.dumps(payload["recommendation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

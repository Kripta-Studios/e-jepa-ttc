#!/usr/bin/env python
"""Numerically audit prefix invariance for legacy vs causal temporal smoothing."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max().item())


def _model_config(path: Path) -> CausalScaleTTCConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model config must be a mapping")
    payload = dict(payload)
    payload.pop("model", None)
    thresholds = payload.get("risk_thresholds_s")
    if isinstance(thresholds, list):
        payload["risk_thresholds_s"] = tuple(float(x) for x in thresholds)
    return CausalScaleTTCConfig(**payload)


def _run_mode(base: CausalScaleTTCConfig, mode: str, seed: int, tolerance: float) -> dict[str, Any]:
    torch.manual_seed(seed)
    cfg = replace(base, foreground_temporal_smoothing_mode=mode)
    model = CausalScaleTTC(cfg).eval()
    # Test on CPU/fp32 deliberately: this is a numerical dependency audit, not a benchmark.
    x = torch.randn(2, 4, cfg.in_channels, 32, 32)
    dt2 = torch.full((2, 1), 0.1)
    dt3 = torch.full((2, 2), 0.1)
    dt4 = torch.full((2, 3), 0.1)
    with torch.inference_mode():
        o2 = model(x[:, :2], dt2)
        o3 = model(x[:, :3], dt3)
        o4 = model(x[:, :4], dt4)
    diffs = {
        "foreground_pair01_append_t2": _max_abs(o2.foreground_logits[:, :2], o3.foreground_logits[:, :2]),
        "foreground_pair01_append_t3": _max_abs(o2.foreground_logits[:, :2], o4.foreground_logits[:, :2]),
        "geometry_endpoint01_append_t2": _max_abs(o2.geometry_tokens[:, :2], o3.geometry_tokens[:, :2]),
        "geometry_endpoint01_append_t3": _max_abs(o2.geometry_tokens[:, :2], o4.geometry_tokens[:, :2]),
        "analytic_pair01_append_t2": _max_abs(o2.analytic_log_height_ratio[:, :1], o3.analytic_log_height_ratio[:, :1]),
        "residual_pair01_append_t2": _max_abs(o2.residual_log_height_ratio[:, :1], o3.residual_log_height_ratio[:, :1]),
        "final_pair01_append_t2": _max_abs(o2.pair_log_height_ratio[:, :1], o3.pair_log_height_ratio[:, :1]),
        "pair_ttc01_append_t2": _max_abs(o2.pair_ttc_seconds[:, :1], o3.pair_ttc_seconds[:, :1]),
        "pair_token01_append_t2": _max_abs(o2.pair_tokens[:, :1], o3.pair_tokens[:, :1]),
        "pair12_append_t3": _max_abs(o3.pair_log_height_ratio[:, 1:2], o4.pair_log_height_ratio[:, 1:2]),
    }
    prefix_invariant = max(diffs.values()) <= tolerance
    return {
        "mode": mode,
        "tolerance": tolerance,
        "prefix_invariant": prefix_invariant,
        "max_abs_difference": max(diffs.values()),
        "differences": diffs,
    }


def run(model_config: Path, output: Path, seed: int, tolerance: float) -> dict[str, Any]:
    base = _model_config(model_config)
    legacy = _run_mode(base, "symmetric_legacy", seed, tolerance)
    causal = _run_mode(base, "causal_left", seed, tolerance)
    none = _run_mode(base, "none", seed, tolerance)
    expected = (
        legacy["prefix_invariant"] is False
        and causal["prefix_invariant"] is True
        and none["prefix_invariant"] is True
    )
    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_prefix_causality_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if expected else "FAIL",
        "model_config": str(model_config),
        "seed": seed,
        "legacy_expected_to_fail_prefix_invariance": True,
        "results": {"symmetric_legacy": legacy, "causal_left": causal, "none": none},
        "interpretation": {
            "legacy_claim": "endpoint-window causal only; not strict prefix-streaming causal",
            "causal_left_claim": "model-level prefix-causal if this dynamic test passes",
            "full_pipeline_claim": "still oracle-ROI unless a causal non-oracle localizer is evaluated",
        },
        "sealed_sources": {"private_test_opened": False},
    }
    sign_artifact(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", type=Path, default=ROOT / "configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_legacy.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--tolerance", type=float, default=1e-6)
    args = p.parse_args()
    try:
        result = run(args.model_config.resolve(), args.output.resolve(), args.seed, args.tolerance)
    except Exception as exc:
        print(f"prefix audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

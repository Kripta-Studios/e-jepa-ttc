#!/usr/bin/env python
"""Stress an E-JEPA checkpoint against synthetic common-ROI mis-localization proxies.

This does NOT simulate a detector. It applies deterministic affine perturbations to the
already-materialized event ROI and quantifies sensitivity. The report labels the test
accordingly so it cannot be confused with an end-to-end localization benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset, collate_object_event_v4
from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics, signed_garl_metrics
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig


def _affine(events: torch.Tensor, tx: float, ty: float, sx: float, sy: float) -> torch.Tensor:
    b, t, c, h, w = events.shape
    flat = events.reshape(b * t, c, h, w)
    # affine_grid maps output -> input. Positive translation here is a deterministic
    # stress direction, not a calibrated detector coordinate error.
    theta = flat.new_zeros((b * t, 2, 3))
    theta[:, 0, 0] = 1.0 / sx
    theta[:, 1, 1] = 1.0 / sy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.reshape(b, t, c, h, w)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _evaluate(model: CausalScaleTTC, loader: DataLoader, device: torch.device, spec: dict[str, float]) -> dict[str, Any]:
    targets: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    seqs: list[str] = []
    knowns: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for host in loader:
            batch = host.to(device)
            events = _affine(batch.events, spec["tx"], spec["ty"], spec["sx"], spec["sy"])
            delta = batch.delta_t_s[:, None].expand(-1, events.shape[1] - 1)
            output = model(events, delta)
            prediction = torch.where(
                output.known_mask,
                output.ttc_mean_seconds.float(),
                torch.full_like(output.ttc_mean_seconds.float(), float("nan")),
            )
            targets.append(batch.target_ttc_s.float().cpu())
            preds.append(prediction.cpu())
            ratios.append(output.log_height_ratio[:, -1].float().cpu())
            knowns.append(output.known_mask.cpu())
            seqs.extend(batch.sequence_ids)
    target = torch.cat(targets).numpy().astype(np.float64)
    pred = torch.cat(preds).numpy().astype(np.float64)
    ratio = torch.cat(ratios).numpy().astype(np.float64)
    known = torch.cat(knowns).numpy().astype(bool)
    # Physical target ratio under the dataset's current endpoint delta.
    # For diagnostic correlation use TTC -> log(1 + dt/TTC) with nominal 0.1 only
    # when per-row dt is not exported by this compact evaluator; TTC metrics remain exact.
    return {
        "signed": signed_garl_metrics(target, pred),
        "sequence_macro": sequence_macro_signed_metrics(target, pred, np.asarray(seqs)),
        "known_coverage": float(known.mean()),
        "prediction_finite_fraction": float(np.isfinite(pred).mean()),
        "prediction_ttc_pearson": _pearson(target, pred),
        "mean_abs_log_ratio": float(np.mean(np.abs(ratio))),
    }


def run(checkpoint: Path, validation_manifest: Path, output: Path, device_name: str, batch_size: int, workers: int, prefetch: int) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError("Unsupported E-JEPA checkpoint payload")
    cfg_raw = dict(payload["model_config"])
    thresholds = cfg_raw.get("risk_thresholds_s")
    if isinstance(thresholds, list):
        cfg_raw["risk_thresholds_s"] = tuple(float(x) for x in thresholds)
    cfg = CausalScaleTTCConfig(**cfg_raw)
    model = CausalScaleTTC(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(device_name)
    model.to(device)
    dataset = GarlTTCObjectEventV4Dataset(str(validation_manifest), splits=("validation",))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=(prefetch if workers > 0 else None),
        collate_fn=collate_object_event_v4,
    )
    specs = {
        "baseline": {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0},
        "translate_x_pos_5pct": {"tx": 0.05, "ty": 0.0, "sx": 1.0, "sy": 1.0},
        "translate_x_neg_5pct": {"tx": -0.05, "ty": 0.0, "sx": 1.0, "sy": 1.0},
        "translate_y_pos_5pct": {"tx": 0.0, "ty": 0.05, "sx": 1.0, "sy": 1.0},
        "translate_y_neg_5pct": {"tx": 0.0, "ty": -0.05, "sx": 1.0, "sy": 1.0},
        "translate_xy_10pct": {"tx": 0.10, "ty": 0.10, "sx": 1.0, "sy": 1.0},
        "scale_in_10pct": {"tx": 0.0, "ty": 0.0, "sx": 0.90, "sy": 0.90},
        "scale_out_10pct": {"tx": 0.0, "ty": 0.0, "sx": 1.10, "sy": 1.10},
        "scale_in_20pct": {"tx": 0.0, "ty": 0.0, "sx": 0.80, "sy": 0.80},
        "scale_out_20pct": {"tx": 0.0, "ty": 0.0, "sx": 1.20, "sy": 1.20},
        "aspect_wide_10pct": {"tx": 0.0, "ty": 0.0, "sx": 1.10, "sy": 0.90},
        "aspect_tall_10pct": {"tx": 0.0, "ty": 0.0, "sx": 0.90, "sy": 1.10},
    }
    results = {name: _evaluate(model, loader, device, spec) for name, spec in specs.items()}
    base_mid = float(results["baseline"]["sequence_macro"]["sequence_macro_paper_MiD_overall"])
    for name, metrics in results.items():
        metrics["delta_sequence_macro_MiD_vs_baseline"] = float(metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"]) - base_mid
    report: dict[str, Any] = {
        "artifact_type": "oracle_roi_affine_stress_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "checkpoint": str(checkpoint.resolve()),
        "validation_manifest": str(validation_manifest.resolve()),
        "method": "post_materialization_common_roi_affine_proxy",
        "method_limitations": [
            "not a detector simulation",
            "does not remove the oracle-box dependency",
            "measures sensitivity to deterministic spatial mis-localization proxies",
        ],
        "results": results,
        "claim_contract": {
            "end_to_end_localizer_evaluated": False,
            "oracle_roi_claim_only": True,
            "private_test_opened": False,
        },
    }
    sign_artifact(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    args = p.parse_args()
    try:
        report = run(args.checkpoint.resolve(), args.validation_manifest.resolve(), args.output.resolve(), args.device, args.batch_size, args.num_workers, args.prefetch_factor)
    except Exception as exc:
        print(f"ROI stress failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Calibrate DINOv3 relational loss weight for A4.

Computes lambda_d = median_geometry / median_relation on a small fixed
subset of the train set (64 samples). The geometry loss is frozen exactly
to the A1-DF warmup definition:
    1.25 * extent + 1.25 * width + 2.5 * center

No optimizer steps are taken. The resulting weight is clipped to [0.25, 4.0].

Usage:
    python scripts/calibrate_a4_dinov3_relational_weight.py \
        --experiment-config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1.yaml \
        --samples 64 \
        --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (  # noqa: E402
    DINOv3RelationalTeacherDataset,
)
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC  # noqa: E402
from e_jepa_ttc.reproducibility import seed_everything  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import _targets  # noqa: E402


def _geometry_loss_per_sample(
    output: Any,
    targets: Any,
    beta: float = 0.02,
) -> torch.Tensor:
    """Compute EXACTLY 1.25 * extent + 1.25 * width + 2.5 * center per sample."""
    # This is a bit tricky as the original loss functions return the mean.
    # For calibration, we can just compute the mean per batch or reimplement
    # the unreduced versions here to get exactly per-sample values.

    if targets.geometry is None:
        raise ValueError("Calibration requires bbox geometry targets")

    valid = targets.geometry.valid.bool()
    if not valid.any():
        return output.visible_height_normalized.new_tensor([]), valid

    # Extent
    log_h_pred = output.visible_height_normalized.clamp_min(1.0e-6).log()
    log_h_tgt = targets.geometry.height_normalized.clamp_min(1.0e-6).log()
    extent = torch.nn.functional.smooth_l1_loss(log_h_pred, log_h_tgt, beta=beta, reduction="none")

    # Width
    log_w_pred = output.visible_width_normalized.clamp_min(1.0e-6).log()
    log_w_tgt = targets.geometry.width_normalized.clamp_min(1.0e-6).log()
    width = torch.nn.functional.smooth_l1_loss(log_w_pred, log_w_tgt, beta=beta, reduction="none")

    # Center
    pred_center = torch.stack(
        (
            output.diagnostics["foreground_centroid_x"],
            output.diagnostics["foreground_centroid_y"],
        ),
        dim=-1,
    )
    tgt_center = torch.stack(
        (
            targets.geometry.centroid_x_normalized,
            targets.geometry.centroid_y_normalized,
        ),
        dim=-1,
    )
    # sum over x,y dims for each sample
    center = torch.nn.functional.smooth_l1_loss(pred_center, tgt_center, beta=beta, reduction="none").mean(dim=-1)

    total = 1.25 * extent + 1.25 * width + 2.5 * center
    total_masked = total * valid.float()

    # average per row to get one scalar loss per sample_token
    valid_count = valid.float().sum(dim=1).clamp_min(1.0)
    per_row_geo = total_masked.sum(dim=1) / valid_count
    return per_row_geo, valid


def run(config_path: str, samples: int, seed: int, device_name: str, output_path: Path | None) -> None:
    seed_everything(seed)
    device = torch.device(device_name)
    config = OmegaConf.load(config_path)

    # 1. Dataset & Loader
    manifest_path = ROOT / str(config.data.cache_manifest)
    train_dataset = GarlTTCObjectEventV4Dataset(
        str(manifest_path), splits=("train",)
    )

    # Need DINO wrapper
    dino_teacher = config.data.get("dinov3_relational_teacher")
    if not dino_teacher or not dino_teacher.get("manifest"):
        raise ValueError("Missing dinov3_relational_teacher in config")
    if dino_teacher["manifest"] == "REPLACE_AFTER_MATERIALIZATION":
        raise ValueError("DINO teacher manifest not materialized yet. Run materialization first.")

    dino_manifest = ROOT / str(dino_teacher["manifest"])
    train_dataset = DINOv3RelationalTeacherDataset(
        train_dataset,
        manifest_path=dino_manifest,
        expected_artifact_sha256=str(dino_teacher["artifact_sha256"]),
        expected_manifest_sha256=str(dino_teacher["manifest_sha256"]),
    )

    # Equispaced sampling
    total_len = len(train_dataset)
    if total_len < samples:
        raise ValueError(f"Dataset has only {total_len} samples, need {samples}")

    indices = np.linspace(0, total_len - 1, samples, dtype=int).tolist()
    subset = torch.utils.data.Subset(train_dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=min(samples, config.training.batch_size),
        shuffle=False,
        collate_fn=collate_object_event_v4,
        num_workers=0,
    )

    # 2. Model
    model_cfg_path = ROOT / str(config.model_config)
    model_config = OmegaConf.load(model_cfg_path)
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig

    mc_dict = OmegaConf.to_container(model_config, resolve=True)
    if isinstance(mc_dict, dict):
        if "model" in mc_dict:
            del mc_dict["model"]
        if "risk_thresholds_s" in mc_dict:
            mc_dict["risk_thresholds_s"] = tuple(mc_dict["risk_thresholds_s"])
        mc = CausalScaleTTCConfig(**mc_dict)
    else:
        raise ValueError("Invalid model config")

    model = CausalScaleTTC(mc).to(device).eval()

    # 3. Calibration Loop
    geometry_losses = []
    relational_losses = []
    tokens_processed = []

    print(f"[calibrate] Starting calibration (target samples: {samples}, seed: {seed})")

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch in loader:
            batch = batch.to(device)
            targets = _targets(
                batch,
                mask_t0_as_proxy=config.training.mask_t0_as_proxy,
                foreground_supervision=config.training.foreground_supervision,
            )
            output = model(batch.events, targets.delta_t_s, return_dense_features=True)

            geo, geo_valid = _geometry_loss_per_sample(output, targets, beta=config.loss.smooth_l1_beta)
            if geo.numel() == 0:
                continue

            student_features = output.endpoint_dense_features[:, 1:3]
            from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
            student_rels = local_cosine_relation_maps(student_features)
            combined_valid = batch.dinov3_relation_valid.bool() & student_rels.valid
            error_map = (student_rels.values - batch.dinov3_relation_targets.float()).abs()

            B = student_features.shape[0]
            for i in range(B):
                valid_mask = combined_valid[i]
                if valid_mask.any() and geo_valid[i].any():
                    rel_val = error_map[i][valid_mask].mean().item()
                    relational_losses.append(rel_val)
                    geometry_losses.append(geo[i].item())
                    tokens_processed.append(batch.sample_tokens[i])

    if not geometry_losses or not relational_losses:
        raise RuntimeError("No valid samples for calibration")

    geo_all = np.array(geometry_losses)
    rel_all = np.array(relational_losses)

    print(f"[calibrate] Collected {len(geo_all)} samples.")

    med_geo = float(np.median(geo_all))
    med_rel = float(np.median(rel_all))

    print(f"  median_geometry: {med_geo:.6f}")
    print(f"  median_relation: {med_rel:.6f}")

    if med_rel == 0.0:
        raise ValueError("median_relation is exactly 0.0, cannot compute ratio")

    lambda_raw = med_geo / med_rel
    lambda_d = float(np.clip(lambda_raw, 0.25, 4.0))

    print(f"\n[calibrate] RAW LAMBDA (geo/rel):  {lambda_raw:.4f}")
    print(f"[calibrate] CLIPPED LAMBDA (A4): {lambda_d:.4f}")

    if output_path is not None:
        from datetime import datetime, UTC
        import json
        from e_jepa_ttc.artifacts.hashing import sign_artifact
        payload = {
            "artifact_type": "dinov3_relational_weight_calibration_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "scope": {
                "public_train_only": True,
                "validation_or_test_opened": False,
                "optimizer_steps": 0,
            },
            "teacher_artifact_sha256": str(dino_teacher["artifact_sha256"]),
            "samples_requested": samples,
            "samples_collected": len(geo_all),
            "seed": seed,
            "tokens": tokens_processed,
            "median_geometry": med_geo,
            "median_relation": med_rel,
            "lambda_raw": lambda_raw,
            "clamp_range": [0.25, 4.0],
            "selected_weight": lambda_d,
        }
        sign_artifact(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n[calibrate] Artifact written to {output_path}")

    print("\nPlease update `representation_distillation_weight` in the experiment config.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=str, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        run(args.experiment_config, args.samples, args.seed, device, args.output.resolve() if args.output else None)
    except Exception as e:
        print(f"\n[calibrate] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

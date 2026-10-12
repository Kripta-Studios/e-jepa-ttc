"""Native producer objective on the valid T3 suffix, with exact batch reductions."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

from e_jepa_ttc.data.object_event_v4 import (
    BoxGeometryTargets,
    ObjectEventV4Batch,
    box_geometry_targets,
)
from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig, causal_scale_ttc_loss
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, target_log_ratio_from_ttc
from operational.rgb_port_revision.loss import reduce_parts

COUNT_KEYS = {
    "log_ratio_nll": "physical_ratio",
    "log_ratio_huber": "physical_ratio",
    "foreground_bce": "foreground",
    "foreground_dice": "foreground",
    "foreground_extent": "foreground_extent",
    "foreground_width": "foreground_geometry",
    "foreground_center": "foreground_geometry",
    "foreground_pair_ratio": "foreground_pair_ratio",
    "risk_bce": "supervised_ttc",
    "auxiliary_inverse_ttc": "supervised_ttc",
    "temporal_consistency": "temporal_consistency",
}


def p2_geometry(batch: ObjectEventV4Batch) -> BoxGeometryTargets:
    """Both retained endpoints have their own real annotation validity."""
    valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0)
    endpoints = valid[:, None].expand(-1, 2)
    return box_geometry_targets(
        batch.boxes_xyxy,
        height=batch.events.shape[-2],
        width=batch.events.shape[-1],
        endpoint_valid=endpoints,
    )


def native_parts(
    model: CausalScaleTTC, batch: ObjectEventV4Batch, cfg: CausalScaleTTCLossConfig
) -> tuple[dict, dict, torch.Tensor]:
    """Compute each component and its true denominator before global top-k."""
    dt = batch.delta_t_s.reshape(len(batch.events), 1)
    valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0)
    out = model(batch.events, dt, return_dense_features=True)
    base = causal_scale_ttc_loss(
        out,
        target_ttc_seconds=batch.target_ttc_s,
        delta_t_s=dt,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=valid,
        target_geometry=p2_geometry(batch),
        config=cfg,
    )
    target, physical = target_log_ratio_from_ttc(batch.target_ttc_s, dt[:, -1])
    physical &= valid
    predicted = (
        out.pair_log_height_ratio
        if cfg.supervise_pair_ratio_before_temporal_blend
        else out.log_height_ratio
    )[:, -1]
    tail = functional.smooth_l1_loss(
        predicted[physical], target[physical], beta=cfg.smooth_l1_beta, reduction="none"
    )
    if (
        out.endpoint_dense_features is None
        or batch.dinov3_relation_targets is None
        or batch.dinov3_relation_valid is None
    ):
        raise ValueError("Native objective requires the admitted P-only teacher")
    rel = local_cosine_relation_maps(out.endpoint_dense_features)
    if rel.values.shape != batch.dinov3_relation_targets.shape:
        raise ValueError("P2 teacher relation shape mismatch")
    mask = rel.valid & batch.dinov3_relation_valid.bool()
    error = (rel.values - batch.dinov3_relation_targets)[mask]
    relational = error.abs().mean() if error.numel() else rel.values.sum() * 0
    components = {**base.components, "dinov3_relational_raw": relational}
    counts = {key: base.counts[value] for key, value in COUNT_KEYS.items()}
    counts["residual_regularization"] = out.residual_log_height_ratio.numel()
    counts["dinov3_relational_raw"] = error.numel()
    return components, counts, tail


def effective_native_loss(
    model: CausalScaleTTC,
    batches: list[ObjectEventV4Batch],
    cfg: CausalScaleTTCLossConfig,
    *,
    recompute: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve RNG while recomputing activations; reduce top-k only globally."""
    parts = []
    for batch in batches:
        if recompute and torch.is_grad_enabled():
            # Pass tensors explicitly so checkpoint preserves the correct CUDA RNG.
            # The closure defaults bind this microbatch independently of the loop.
            def calculate(events: torch.Tensor, item: ObjectEventV4Batch = batch) -> tuple:
                return native_parts(model, replace(item, events=events), cfg)

            parts.append(
                checkpoint(calculate, batch.events, use_reentrant=False, preserve_rng_state=True)
            )
        else:
            parts.append(native_parts(model, batch, cfg))
    components = reduce_parts(parts, cfg.log_ratio_tail_fraction)
    total = components["log_ratio_nll"].new_zeros(())
    for name, value in components.items():
        weight = (
            8.0
            if name == "dinov3_relational_raw"
            else getattr(cfg, f"{'risk' if name == 'risk_bce' else name}_weight")
        )
        total = total + weight * value
    return total, {name: value.detach() for name, value in components.items()}

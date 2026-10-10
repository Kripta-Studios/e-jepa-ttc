"""Exact effective-batch reductions for heterogeneous RGB T2/T3 inputs."""

# ruff: noqa: ANN401 -- adapters bridge the retained dynamic source protocol

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
from operational.rgb_port import train_producers as original


def reduce_parts(
    parts: list[tuple[dict[str, Tensor], dict[str, int], Tensor]], fraction: float
) -> dict[str, Tensor]:
    """Reduce means by their actual denominators; select the tail only once."""
    if not parts or not 0 < fraction <= 1:
        raise ValueError("Nonempty parts and a tail fraction in (0,1] are required")
    result = {}
    for name in parts[0][0]:
        if name == "log_ratio_tail":
            values = torch.cat([part[2] for part in parts])
            result[name] = (
                values.topk(max(1, math.ceil(fraction * len(values)))).values.mean()
                if values.numel()
                else parts[0][0][name] * 0
            )
        else:
            count = sum(part[1][name] for part in parts)
            result[name] = sum(part[0][name] * part[1][name] for part in parts) / max(1, count)
    return result


@dataclass
class EffectiveBatch:
    """Keep real temporal lengths; no padded frame reaches a producer."""

    parts: list[Any]

    def to(self, device: torch.device) -> EffectiveBatch:
        return EffectiveBatch([part.to(device) for part in self.parts])


class EffectiveSource:
    """Expose one effective batch to the frozen optimizer/update-journal loop."""

    def __init__(self, source: Any, recipe: Any, revision_sha256: str) -> None:
        self.source, self.recipe = source, recipe
        self.identity: Mapping[str, Any] = {
            **source.identity,
            "audit_revision_sha256": revision_sha256,
        }
        self.population_size: int = source.population_size
        self.frame_counts: Sequence[int] = source.frame_counts
        supplied_spans = getattr(source, "input_span_us", None)
        self.input_span_us: Sequence[int] = (
            supplied_spans
            if supplied_spans is not None
            else tuple(
                int(row["producer_sensor_times_us"][-1]) - int(row["producer_sensor_times_us"][0])
                for row in getattr(getattr(source, "dataset", None), "rows", ())
            )
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def batch(self, indices: Sequence[int], modality: str) -> EffectiveBatch:
        groups: dict[int, list[int]] = {}
        for index in indices:
            groups.setdefault(int(self.source.frame_counts[index]), []).append(index)
        return EffectiveBatch(
            [
                self.source.batch(group[start : start + self.recipe.microbatch_size], modality)
                for _, group in sorted(groups.items())
                for start in range(0, len(group), self.recipe.microbatch_size)
            ]
        )


def effective_loss(
    model: Any, batch: EffectiveBatch, config: Any
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine all differentiable component numerators before one backward pass."""
    from e_jepa_ttc.models.causal_scale_ttc import target_log_ratio_from_ttc

    parts = []
    for item in batch.parts:
        size, steps = item.events.shape[:2]
        delta = original._pair_intervals(item.delta_t_s, batch_size=size, steps=steps)
        endpoint_valid = torch.ones(
            item.boxes_xyxy.shape[:2], device=item.events.device, dtype=torch.bool
        )
        endpoint_valid[:, 0] = False
        valid = torch.isfinite(item.target_ttc_s) & (item.target_ttc_s != 0)
        geometry = original.box_geometry_targets(
            item.boxes_xyxy,
            height=item.events.shape[-2],
            width=item.events.shape[-1],
            endpoint_valid=endpoint_valid & valid[:, None],
        )
        output = model(item.events, delta, return_dense_features=True)
        base = original.causal_scale_ttc_loss(
            output,
            target_ttc_seconds=item.target_ttc_s,
            delta_t_s=delta,
            risk_thresholds_s=model.config.risk_thresholds_s,
            target_valid=valid,
            target_geometry=geometry,
            config=config,
        )
        target, physical = target_log_ratio_from_ttc(item.target_ttc_s, delta[:, -1])
        physical &= valid
        prediction = (
            output.pair_log_height_ratio
            if config.supervise_pair_ratio_before_temporal_blend
            else output.log_height_ratio
        )[:, -1]
        tail = functional.smooth_l1_loss(
            prediction[physical], target[physical], beta=config.smooth_l1_beta, reduction="none"
        )
        dense = output.endpoint_dense_features
        if (
            dense is None
            or item.dinov3_relation_targets is None
            or item.dinov3_relation_valid is None
        ):
            raise ValueError("RGB training requires dense endpoints and frozen teacher relations")
        relations = local_cosine_relation_maps(dense[:, -2:])
        if (
            item.dinov3_relation_targets.shape != relations.values.shape
            or item.dinov3_relation_valid.shape != relations.valid.shape
        ):
            raise ValueError("Teacher/student relation shapes differ")
        relation_valid = relations.valid & item.dinov3_relation_valid.bool()
        relation_error = (relations.values.float() - item.dinov3_relation_targets.float())[
            relation_valid
        ]
        if not bool(torch.isfinite(relation_error).all()):
            raise ValueError("Nonfinite teacher/student relation")
        relational = relation_error.abs().mean() if relation_error.numel() else dense.sum() * 0
        components = {**base.components, "dinov3_relational_raw": relational}
        keys = {
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
        counts = {name: base.counts[key] for name, key in keys.items()}
        counts["residual_regularization"] = output.residual_log_height_ratio.numel()
        counts["dinov3_relational_raw"] = relation_error.numel()
        parts.append((components, counts, tail))
    components = reduce_parts(parts, config.log_ratio_tail_fraction)
    aliases = {"risk_bce": "risk", "dinov3_relational_raw": None}
    total = components["log_ratio_nll"].new_zeros(())
    for name, value in components.items():
        total = total + value * (
            8.0
            if name == "dinov3_relational_raw"
            else getattr(config, f"{aliases.get(name, name)}_weight")
        )
    components["dinov3_relational_weighted"] = 8 * components["dinov3_relational_raw"]
    return total, {name: value.detach() for name, value in components.items()}

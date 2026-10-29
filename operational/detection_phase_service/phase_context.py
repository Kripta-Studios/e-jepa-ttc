"""Opt-in batched TRAIN40 context route exposing native PHASE coordinates.

The historical :class:`BatchedContextRuntime` remains the source of the first
four scalar outputs.  This additive route evaluates the same frozen R_CTX
width head and ``PhaseOnlyFinal.forward`` to transport native phase bounds.
Those bounds are phase coordinates, not a calibrated continuous TTC interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.model import TemporalRefiner
from operational.final_refit.training import PhaseOnlyFinal
from operational.train40_runtime.batched_context import (
    BatchedContextRuntime,
    ContextObservation,
    _BatchRefiner,
)
from operational.train40_runtime.context import HEAD


class _PhaseBatchRefiner(_BatchRefiner):
    """Return the historical 170-D block and its frozen width logits."""

    def forward(
        self, features: Tensor, times: Tensor, valid: Tensor, experts: Tensor
    ) -> tuple[Tensor, Tensor]:
        block = super().forward(features, times, valid, experts)

        # Reproduce the exact padded-eight hidden route used by _BatchRefiner.
        safe_x = torch.where(valid[..., None], features, torch.zeros_like(features))
        safe_t = torch.where(valid[..., None], times, torch.zeros_like(times))
        z = self.model.project(torch.cat((safe_x, safe_t), -1))
        first = z.new_zeros((len(z), 160))
        second = z.new_zeros((len(z), 160))
        for index in range(8):
            proposal = self.model.cells[0](z[:, index], first)
            first = torch.where(valid[:, index, None], proposal, first)
            proposal = self.model.cells[1](first, second)
            second = torch.where(valid[:, index, None], proposal, second)
        return block, self.model.width(second)


class _BatchPhaseCore(nn.Module):
    """Canonical native PHASE output over the same frozen batched states."""

    def __init__(self, models: Mapping[str, nn.Module]) -> None:
        super().__init__()
        event, rgb, head = models["E_CTX_MATCHED"], models["R_CTX"], models[HEAD]
        if not isinstance(event, TemporalRefiner) or not isinstance(rgb, TemporalRefiner):
            raise TypeError("Phase context requires frozen TemporalRefiner models")
        if not isinstance(head, PhaseOnlyFinal):
            raise TypeError("Phase context requires the frozen PhaseOnlyFinal head")
        self.event = _BatchRefiner(event)
        self.rgb = _PhaseBatchRefiner(rgb)
        self.head = head

    def forward(self, *inputs: Tensor) -> Tensor:
        event_block = self.event(*inputs[:4])
        rgb_block, rgb_width_logits = self.rgb(*inputs[4:8])
        native = self.head(torch.cat((rgb_block, event_block), -1), rgb_width_logits)
        finite = torch.stack(
            tuple(torch.isfinite(value).reshape(len(value), -1).all(1) for value in inputs[:8])
        ).all(0)
        finite = finite & torch.stack(
            tuple(torch.isfinite(native[name]) for name in ("ttc", "point_phase", "q10", "q90"))
        ).all(0)
        return torch.stack(
            (
                native["ttc"],
                native["point_phase"],
                native["q10"],
                native["q90"],
                finite.to(native["ttc"].dtype),
            ),
            dim=1,
        )


class BatchedPhaseContextRuntime(BatchedContextRuntime):
    """Historical batched TTC plus opt-in native PHASE bounds.

    CUDA graphs are deliberately unsupported by this first additive version.
    Missing RGB history emits an explicit unavailable reason; event-only phase
    bounds are never fabricated.
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        normalizers: Mapping[str, tuple[Tensor, Tensor]],
        device: str | torch.device,
        *,
        graphs: bool = False,
        max_objects: int = 20,
    ) -> None:
        if graphs:
            raise ValueError("Native phase-bound context currently requires graphs=False")
        super().__init__(models, normalizers, device, graphs=False, max_objects=max_objects)
        self.phase_core = _BatchPhaseCore(models).eval()

    def _prepare(
        self, observations: Sequence[ContextObservation]
    ) -> tuple[tuple[Tensor, ...], list[int]]:
        if not 1 <= len(observations) <= self.max_objects:
            raise ValueError("Object count is outside the configured one-to-twenty bound")
        ids = [item.object_id for item in observations]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(
            ids
        ):
            raise ValueError("Object IDs must be unique nonempty strings")
        contexts: list[tuple[Tensor, ...]] = []
        rgb_counts: list[int] = []
        for item in observations:
            event = self.reference._context(
                item.event_raw,
                item.event_experts,
                item.event_anchor_us,
                [item.query_us] * len(item.event_anchor_us),
                item.query_us,
                "event",
            )
            selected = [
                index
                for index, metadata in enumerate(item.rgb_metadata)
                if int(metadata["available_us"]) <= item.query_us
                and 0 <= item.query_us - int(metadata["first_sensor_ts_us"]) <= 650_000
            ][-8:]
            rgb_counts.append(len(selected))
            if selected:
                if item.rgb_raw is None or item.rgb_experts is None:
                    raise ValueError("Available RGB requires resident features")
                if len(item.rgb_raw) != len(item.rgb_metadata) or len(item.rgb_experts) != len(
                    item.rgb_metadata
                ):
                    raise ValueError("RGB features and metadata are not aligned")
                rgb = self.reference._context(
                    torch.stack([item.rgb_raw[index] for index in selected]),
                    torch.stack([item.rgb_experts[index] for index in selected]),
                    [int(item.rgb_metadata[index]["anchor_us"]) for index in selected],
                    [int(item.rgb_metadata[index]["available_us"]) for index in selected],
                    item.query_us,
                    "rgb",
                )
            else:
                rgb = tuple(torch.zeros_like(value) for value in event)
            contexts.append((*event, *rgb))
        combined = tuple(torch.cat([context[index] for context in contexts]) for index in range(8))
        has_rgb = torch.tensor([bool(count) for count in rgb_counts], device=self.device)
        return (*combined, has_rgb), rgb_counts

    @torch.inference_mode()
    def predict_many(self, observations: Sequence[ContextObservation]) -> list[dict[str, Any]]:
        """Preserve historical TTC and add canonical phase output when RGB exists."""
        inputs, rgb_counts = self._prepare(observations)
        historical = self._run(inputs)
        native = self.phase_core(*inputs)
        has_rgb = inputs[8]
        if not bool(torch.isfinite(historical).all()) or not bool((historical[:, 3] == 1).all()):
            raise ValueError("Nonfinite context input or prediction")
        if bool(has_rgb.any()):
            admitted = native[has_rgb]
            if not bool(torch.isfinite(admitted).all()) or not bool((admitted[:, 4] == 1).all()):
                raise ValueError("Nonfinite native phase prediction")
            if not torch.equal(historical[has_rgb, 0], admitted[:, 0]):
                raise ValueError("Native phase TTC differs from historical frozen TTC")
            if not bool((admitted[:, 2] <= admitted[:, 3]).all()):
                raise ValueError("Native phase bounds are reversed")
        values = historical.cpu().numpy()
        phases = native.cpu().numpy()
        result = []
        for index, item in enumerate(observations):
            row: dict[str, Any] = {
                "object_id": item.object_id,
                "predictions": {
                    "PHASE_ONLY_TRAIN40_S7": float(values[index, 0]),
                    "E_CTX_TRAIN40": float(values[index, 1]),
                    "R_CTX_TRAIN40": float(values[index, 2]) if rgb_counts[index] else None,
                },
                "history_lengths": {"event": len(item.event_anchor_us), "rgb": rgb_counts[index]},
                "rgb_available": bool(rgb_counts[index]),
            }
            if rgb_counts[index]:
                row["phase"] = {
                    "point_phase": float(phases[index, 1]),
                    "q10": float(phases[index, 2]),
                    "q90": float(phases[index, 3]),
                }
                row["phase_unavailable_reason"] = ""
            else:
                row["phase"] = None
                row["phase_unavailable_reason"] = "rgb_history_unavailable"
            result.append(row)
        return result


__all__ = ["BatchedPhaseContextRuntime"]

"""Padded TRAIN40 contexts evaluated together with object identities preserved."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.model import TemporalRefiner
from operational.final_refit.training import PhaseOnlyFinal
from operational.train40_runtime.context import (
    HEAD,
    ContextRuntime,
    _emitted_phase,
    _phase_to_ttc,
)


@dataclass(frozen=True)
class ContextObservation:
    """One object's resident features and their original causal clocks."""

    object_id: str
    event_raw: Tensor
    event_experts: Tensor
    event_anchor_us: Sequence[int]
    rgb_raw: Tensor | None
    rgb_experts: Tensor | None
    rgb_metadata: Sequence[Mapping[str, Any]]
    query_us: int


class _BatchRefiner(nn.Module):
    def __init__(self, model: TemporalRefiner) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor, times: Tensor, valid: Tensor, experts: Tensor) -> Tensor:
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
        anchor = (
            experts.sort(dim=-1, stable=True).values[:, 1]
            if experts.device.type == "cuda"
            else experts.median(-1).values
        )
        location = anchor + 0.03 * self.model.location(second).squeeze(-1)
        point = _emitted_phase(location)
        return torch.cat(
            (
                second,
                location[:, None],
                point[:, None],
                experts,
                times[:, -1],
                valid.sum(dim=1, keepdim=True).to(features.dtype) / 8,
            ),
            dim=-1,
        )


class _BatchCore(nn.Module):
    def __init__(self, models: Mapping[str, nn.Module]) -> None:
        super().__init__()
        event, rgb, head = models["E_CTX_MATCHED"], models["R_CTX"], models[HEAD]
        if not isinstance(event, TemporalRefiner) or not isinstance(rgb, TemporalRefiner):
            raise TypeError("Batched context requires frozen TemporalRefiner models")
        if not isinstance(head, PhaseOnlyFinal):
            raise TypeError("Batched context requires the frozen PhaseOnlyFinal head")
        self.event = _BatchRefiner(event)
        self.rgb = _BatchRefiner(rgb)
        self.head = head

    def forward(self, *inputs: Tensor) -> Tensor:
        event_block = self.event(*inputs[:4])
        rgb_block = self.rgb(*inputs[4:8])
        has_rgb = inputs[8]
        combined = torch.cat((rgb_block, event_block), -1)
        correction = self.head.output(self.head.mlp(self.head.inputs(combined)))
        location = combined[:, 160] + 0.03 * correction[:, 0]
        event_ttc = _phase_to_ttc(event_block[:, 161])
        rgb_ttc = _phase_to_ttc(rgb_block[:, 161])
        final = torch.where(has_rgb, _phase_to_ttc(location), event_ttc)
        finite_event = torch.isfinite(event_block).all(dim=1)
        finite_rgb = torch.isfinite(rgb_block).all(dim=1) & torch.isfinite(location)
        for value in inputs[:4]:
            finite_event = finite_event & torch.isfinite(value).reshape(len(value), -1).all(1)
        for value in inputs[4:8]:
            finite_rgb = finite_rgb & torch.isfinite(value).reshape(len(value), -1).all(1)
        finite = finite_event & (~has_rgb | finite_rgb)
        return torch.stack((final, event_ttc, rgb_ttc, finite.float()), dim=1)


@dataclass
class _Capture:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[Tensor, ...]
    output: Tensor


class BatchedContextRuntime:
    """Emit each object's three TTC values with one final host transfer.

    Batching changes GEMM reduction shapes, so admission against single-object
    frozen predictions remains mandatory. Instances have one owning thread and
    stream; their graph output is consumed before another invocation.
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
        if not isinstance(max_objects, int) or not 1 <= max_objects <= 20:
            raise ValueError("max_objects must be between one and twenty")
        self.reference = ContextRuntime(models, normalizers, device, graphs=False)
        self.device = self.reference.device
        if graphs and self.device.type != "cuda":
            raise ValueError("Context CUDA graphs require a CUDA device")
        self.graphs = graphs
        self.max_objects = max_objects
        self.core = _BatchCore(models).eval()
        self.captures: dict[int, _Capture] = {}

    def _run(self, inputs: tuple[Tensor, ...]) -> Tensor:
        if not self.graphs:
            return self.core(*inputs)
        count = len(inputs[0])
        entry = self.captures.get(count)
        if entry is None:
            static = tuple(value.clone() for value in inputs)
            stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.core(*static)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.core(*static)
            entry = _Capture(graph, static, output)
            self.captures[count] = entry
        for destination, value in zip(entry.inputs, inputs, strict=True):
            destination.copy_(value)
        entry.graph.replay()
        return entry.output

    @torch.inference_mode()
    def predict_many(self, observations: Sequence[ContextObservation]) -> list[dict[str, Any]]:
        """Apply causal selection independently; reject duplicate or empty IDs."""
        if not 1 <= len(observations) <= self.max_objects:
            raise ValueError("Object count is outside the configured one-to-twenty bound")
        ids = [item.object_id for item in observations]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(
            ids
        ):
            raise ValueError("Object IDs must be unique nonempty strings")
        contexts: list[tuple[Tensor, ...]] = []
        rgb_counts = []
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
                i
                for i, metadata in enumerate(item.rgb_metadata)
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
                    torch.stack([item.rgb_raw[i] for i in selected]),
                    torch.stack([item.rgb_experts[i] for i in selected]),
                    [int(item.rgb_metadata[i]["anchor_us"]) for i in selected],
                    [int(item.rgb_metadata[i]["available_us"]) for i in selected],
                    item.query_us,
                    "rgb",
                )
            else:
                rgb = tuple(torch.zeros_like(value) for value in event)
            contexts.append((*event, *rgb))
        combined = tuple(torch.cat([context[i] for context in contexts]) for i in range(8))
        has_rgb = torch.tensor([bool(count) for count in rgb_counts], device=self.device)
        values = self._run((*combined, has_rgb)).cpu().numpy()
        if not np.isfinite(values).all() or not np.all(values[:, 3] == 1):
            raise ValueError("Nonfinite context input or prediction")
        return [
            {
                "object_id": item.object_id,
                "predictions": {
                    "PHASE_ONLY_TRAIN40_S7": float(values[i, 0]),
                    "E_CTX_TRAIN40": float(values[i, 1]),
                    "R_CTX_TRAIN40": float(values[i, 2]) if rgb_counts[i] else None,
                },
                "history_lengths": {"event": len(item.event_anchor_us), "rgb": rgb_counts[i]},
                "rgb_available": bool(rgb_counts[i]),
            }
            for i, item in enumerate(observations)
        ]

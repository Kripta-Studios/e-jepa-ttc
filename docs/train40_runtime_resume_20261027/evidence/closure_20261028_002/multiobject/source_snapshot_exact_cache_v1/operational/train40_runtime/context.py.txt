"""Resident, inference-only context route for the sealed TRAIN40 chain.

The original scientific modules remain the oracle. This additive runtime removes
feature copies through NumPy and exposes the same context coordinates without
installing forward hooks. CUDA graphs are optional and cover only this context
route, never sensor preparation or the four producers. Instances are single-owner
and must not be called concurrently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.model import TemporalRefiner
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE
from operational.final_refit.training import PhaseOnlyFinal
from operational.rgb_port.infer_experts import clock_features

HEAD = "PHASE_ONLY_EVENT_S7_TRAIN40"
ContextTuple = tuple[Tensor, Tensor, Tensor, Tensor]


def _phase_to_ttc(phase: Tensor) -> Tensor:
    """Frozen phase arithmetic, with finite checks performed at the boundary."""
    phi = phase.clamp(MIN_PHASE, MAX_PHASE)
    q = -torch.expm1(-phi) / 0.1
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    return sign / q.abs().clamp_min(1.0 / 60.0)


def _emitted_phase(phase: Tensor) -> Tensor:
    return -torch.log1p(-0.1 / _phase_to_ttc(phase))


class _ResidentRefiner(nn.Module):
    """The frozen padded-eight GRU arithmetic, with hook outputs made explicit."""

    def __init__(self, model: TemporalRefiner) -> None:
        super().__init__()
        if (
            model.cfg.feature_count != 17
            or model.cfg.hidden != 160
            or model.cfg.backbone != "gru"
            or model.cfg.output_mode != "residual"
        ):
            raise ValueError("Resident context requires TRAIN40's 17/160/GRU/residual ABI")
        self.model = model

    def forward(
        self, features: Tensor, times: Tensor, valid: Tensor, experts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        safe_x = torch.where(valid[..., None], features, torch.zeros_like(features))
        safe_t = torch.where(valid[..., None], times, torch.zeros_like(times))
        z = self.model.project(torch.cat((safe_x, safe_t), -1))
        states = [z.new_zeros((1, 160)), z.new_zeros((1, 160))]
        for index in range(8):
            proposal = self.model.cells[0](z[:, index], states[0])
            states[0] = torch.where(valid[:, index, None], proposal, states[0])
            proposal = self.model.cells[1](states[0], states[1])
            states[1] = torch.where(valid[:, index, None], proposal, states[1])
        hidden = states[-1]
        anchor = (
            experts.sort(dim=-1, stable=True).values[:, 1]
            if experts.device.type == "cuda"
            else experts.median(-1).values
        )
        delta = 0.03 * self.model.location(hidden).squeeze(-1)
        location = anchor + delta
        width_logits = self.model.width(hidden)
        point = _emitted_phase(location)
        block = torch.cat(
            (
                hidden,
                location[:, None],
                point[:, None],
                experts,
                times[:, -1],
                valid.sum(dim=1, keepdim=True).to(features.dtype) / 8,
            ),
            dim=-1,
        )
        return block, width_logits, _phase_to_ttc(point)


class _ContextCore(nn.Module):
    """Static arithmetic for one padded-eight query, using the original weights."""

    def __init__(
        self, event: TemporalRefiner, rgb: TemporalRefiner, head: PhaseOnlyFinal, has_rgb: bool
    ) -> None:
        super().__init__()
        self.event = _ResidentRefiner(event)
        self.rgb = _ResidentRefiner(rgb)
        self.head = head
        self.has_rgb = has_rgb

    def forward(self, *inputs: Tensor) -> Tensor:
        event_block, _, event_ttc = self.event(*inputs[:4])
        finite = torch.stack([torch.isfinite(value).all() for value in inputs]).all()
        finite = finite & torch.isfinite(event_block).all()
        if not self.has_rgb:
            return torch.cat((event_ttc, event_ttc, event_ttc, finite.float()[None]))
        rgb_block, _, rgb_ttc = self.rgb(*inputs[4:])
        features = torch.cat((rgb_block, event_block), -1)
        correction = self.head.output(self.head.mlp(self.head.inputs(features)))
        location = features[:, 160] + 0.03 * correction[:, 0]
        finite = finite & torch.isfinite(rgb_block).all() & torch.isfinite(location).all()
        return torch.cat((_phase_to_ttc(location), event_ttc, rgb_ttc, finite.float()[None]))


@dataclass
class _CapturedContext:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[Tensor, ...]
    output: Tensor


class ContextRuntime:
    """Predict from resident producer features without an intermediate CPU copy.

    ``predict`` returns the three original scalar predictions and transfers only
    four scalars to the host. The fourth reports input finiteness, preserving the
    oracle's rejection of nonfinite features even when a finite median could
    otherwise mask a bad expert. Numerical mode is inherited from the caller;
    the frozen FP32, TF32-disabled mode must be configured before construction.
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        normalizers: Mapping[str, tuple[Tensor, Tensor]],
        device: str | torch.device,
        *,
        graphs: bool = False,
    ) -> None:
        self.device = torch.device(device)
        event, rgb, head = models["E_CTX_MATCHED"], models["R_CTX"], models[HEAD]
        if not isinstance(event, TemporalRefiner) or not isinstance(rgb, TemporalRefiner):
            raise TypeError("Resident context requires the sealed TemporalRefiner models")
        if not isinstance(head, PhaseOnlyFinal):
            raise TypeError("Resident context requires the sealed PhaseOnlyFinal model")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = next(event.parameters()).device
        for model in (event, rgb, head):
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("Resident context accepts only frozen eval models")
            if any(
                parameter.dtype != torch.float32 or parameter.device != self.device
                for parameter in model.parameters()
            ):
                raise ValueError("Context weights must be resident FP32 on the requested device")
        self.normalizers = normalizers
        for modality in ("event", "rgb"):
            mean, scale = normalizers[modality]
            if (
                mean.shape != (17,)
                or scale.shape != (17,)
                or mean.device != self.device
                or scale.device != self.device
                or mean.dtype != torch.float32
                or scale.dtype != torch.float32
                or not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all())
                or not bool((scale > 0).all())
            ):
                raise ValueError("Normalizer must contain finite resident FP32 mean/positive scale")
        if graphs and self.device.type != "cuda":
            raise ValueError("Context CUDA graphs require a CUDA device")
        self.graphs = graphs
        self.cores = {
            has_rgb: _ContextCore(event, rgb, head, has_rgb).eval() for has_rgb in (False, True)
        }
        self.captures: dict[bool, _CapturedContext] = {}

    def _context(
        self,
        raw: Tensor,
        experts: Tensor,
        anchors: Sequence[int],
        available: Sequence[int],
        query_us: int,
        modality: str,
    ) -> ContextTuple:
        count = len(anchors)
        if not 1 <= count <= 8 or len(available) != count:
            raise ValueError("Context requires one to eight aligned observations")
        if any(later <= earlier for earlier, later in zip(anchors[:-1], anchors[1:], strict=True)):
            raise ValueError("Context anchors must be strictly chronological")
        if raw.shape != (count, 17) or experts.shape != (count, 3):
            raise ValueError("Context raw/expert feature shapes differ")
        if any(
            tensor.dtype != torch.float32 or tensor.device != self.device
            for tensor in (raw, experts)
        ):
            raise ValueError("Producer features must be resident FP32 tensors")
        features = torch.zeros((1, 8, 17), dtype=torch.float32, device=self.device)
        mean, scale = self.normalizers[modality]
        features[0, -count:] = (raw - mean) / scale
        timing = torch.zeros((1, 8, 4), dtype=torch.float32, device=self.device)
        timing[0, -count:] = torch.from_numpy(clock_features(anchors, available, query_us)).to(
            self.device
        )
        valid = torch.zeros((1, 8), dtype=torch.bool, device=self.device)
        valid[0, -count:] = True
        return features, timing, valid, experts[-1:]

    def _run(self, inputs: tuple[Tensor, ...], has_rgb: bool) -> Tensor:
        core = self.cores[has_rgb]
        if not self.graphs:
            return core(*inputs)
        captured = self.captures.get(has_rgb)
        if captured is None:
            static = tuple(value.clone() for value in inputs)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    core(*static)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = core(*static)
            captured = _CapturedContext(graph, static, output)
            self.captures[has_rgb] = captured
        for destination, source in zip(captured.inputs, inputs, strict=True):
            destination.copy_(source)
        captured.graph.replay()
        return captured.output

    @torch.inference_mode()
    def predict(
        self,
        event_raw: Tensor,
        event_experts: Tensor,
        event_anchor_us: Sequence[int],
        rgb_raw: Tensor | None,
        rgb_experts: Tensor | None,
        rgb_metadata: Sequence[Mapping[str, Any]],
        query_us: int,
    ) -> dict[str, Any]:
        """Apply the frozen causal history selection and emit the original outputs."""
        event_context = self._context(
            event_raw,
            event_experts,
            event_anchor_us,
            [query_us] * len(event_anchor_us),
            query_us,
            "event",
        )
        selected = [
            index
            for index, item in enumerate(rgb_metadata)
            if int(item["available_us"]) <= query_us
            and 0 <= query_us - int(item["first_sensor_ts_us"]) <= 650_000
        ][-8:]
        inputs: tuple[Tensor, ...] = event_context
        if selected:
            if rgb_raw is None or rgb_experts is None:
                raise ValueError("Available RGB observations require resident features")
            if rgb_raw.shape[0] != len(rgb_metadata) or rgb_experts.shape[0] != len(rgb_metadata):
                raise ValueError("RGB features and metadata are not aligned")
            # Lists of per-row views avoid changing the arithmetic of producer batches.
            raw = torch.stack([rgb_raw[index] for index in selected])
            experts = torch.stack([rgb_experts[index] for index in selected])
            rgb_context = self._context(
                raw,
                experts,
                [int(rgb_metadata[index]["anchor_us"]) for index in selected],
                [int(rgb_metadata[index]["available_us"]) for index in selected],
                query_us,
                "rgb",
            )
            inputs += rgb_context
        values = self._run(inputs, bool(selected)).cpu().numpy()
        if not np.isfinite(values).all() or values[3] != 1:
            raise ValueError("Nonfinite context input or prediction")
        return {
            "predictions": {
                "PHASE_ONLY_TRAIN40_S7": float(values[0]),
                "E_CTX_TRAIN40": float(values[1]),
                "R_CTX_TRAIN40": float(values[2]) if selected else None,
            },
            "history_lengths": {"event": len(event_anchor_us), "rgb": len(selected)},
            "rgb_available": bool(selected),
        }

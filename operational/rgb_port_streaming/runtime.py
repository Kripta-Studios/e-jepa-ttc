"""Modality-specific V13 experts, causal feature reuse, and native head inputs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.features import (
    EVENT_PHASE17_SHA256,
    RGB_PHASE17_SHA256,
    event_phase17,
    event_statistics_from_channels12,
    raw_rgb_statistics,
)
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.rgb_port.infer_experts import clock_features
from operational.rgb_port.train_heads import pair_point_phase
from operational.rgb_port.train_producers import producer_features
from operational.streaming_revision.kernels import EncoderPrecision, GraphModule
from operational.streaming_revision.state import FeatureState, Observation, Policy, roi_distance

from .contracts import NativeObservation
from .transport import install_transport


class Producer(nn.Module):
    """Preserve V13 feature semantics inside an optional inference graph."""

    def __init__(self, model: nn.Module, *, graphs: bool = False) -> None:
        super().__init__()
        if graphs:
            graph = GraphModule(model)
            graph.__dict__["config"] = cast(CausalScaleTTC, model).config
            self.model = graph
        else:
            self.model = model

    def forward(self, sensor: Tensor, delta: Tensor) -> dict[str, Tensor]:
        """Return the same label-free fields as the canonical V13 producer."""
        return producer_features(cast(CausalScaleTTC, self.model), sensor, delta)


class NativeExperts:
    """Copy V13 weights for inference; never modify live training modules.

    Reduced precision applies only to inference encoders. V13 training remains
    under its existing BF16 recipe. A V12 student is never silently substituted.
    """

    def __init__(
        self,
        a5: nn.Module,
        c2f: nn.Module,
        pair: nn.Module,
        *,
        modality: str,
        parent_sha256: str,
        vectorized: bool = True,
        encoder_precision: str = "fp32",
        graphs: bool = False,
        batch_size: int = 2,
        student: nn.Module | None = None,
        student_binding: Mapping[str, str] | None = None,
    ) -> None:
        if modality not in {"event", "rgb"} or batch_size not in {1, 2, 4, 8}:
            raise ValueError("Explicit event/rgb modality and admitted fixed batch required")
        self.modality, self.parents, self.batch_size = modality, parent_sha256, batch_size
        self.device = next(a5.parameters()).device
        self.schema = EVENT_PHASE17_SHA256 if modality == "event" else RGB_PHASE17_SHA256
        if graphs and self.device.type != "cuda":
            raise ValueError("Inference graphs require CUDA")
        self.models = []
        for original in (a5, c2f):
            if cast(CausalScaleTTC, original).config.modality != modality:
                raise ValueError("Producer modality differs from the streaming schema")
            model = copy.deepcopy(original).eval().requires_grad_(False)
            if vectorized:
                install_transport(model)
            if encoder_precision != "fp32":
                model.encoder = EncoderPrecision(cast(nn.Module, model.encoder), encoder_precision)
            self.models.append(Producer(model, graphs=graphs))
        self.pair = copy.deepcopy(pair).eval().requires_grad_(False)
        self.student = copy.deepcopy(student).eval() if student is not None else None
        if self.student is not None:
            expected = {
                "modality": modality,
                "parent_sha256": parent_sha256,
                "schema_sha256": self.schema,
                "fit_role": "H",
            }
            if student_binding is None or any(
                student_binding.get(k) != v for k, v in expected.items()
            ):
                raise ValueError(
                    "Student must bind V13 parents, modality, schema and training role H"
                )
            self.student.to(self.device)

    @torch.inference_mode()
    def __call__(self, sensor: Tensor, delta: Tensor) -> Tensor:
        """Execute fixed-size producer batches and discard padding outputs."""
        if sensor.ndim != 5 or sensor.shape[1] not in {2, 3}:
            raise ValueError("Real two/three-frame producer inputs required")
        if sensor.shape[2] != (12 if self.modality == "event" else 3):
            raise ValueError("Sensor channels differ from modality")
        if delta.shape != (len(sensor), sensor.shape[1] - 1) or (delta <= 0).any():
            raise ValueError("Measured positive frame intervals required")
        if not torch.isfinite(sensor).all() or not torch.isfinite(delta).all():
            raise ValueError("Finite sensor input and measured intervals required")
        result = []
        for start in range(0, len(sensor), self.batch_size):
            x = sensor[start : start + self.batch_size].to(self.device).float()
            dt = delta[start : start + self.batch_size].to(self.device).float()
            n = len(x)
            if n < self.batch_size:
                x = torch.cat((x, x[-1:].expand(self.batch_size - n, -1, -1, -1, -1)))
                dt = torch.cat((dt, dt[-1:].expand(self.batch_size - n, -1)))
            a5 = self.models[0](x, dt)
            base = (
                event_statistics_from_channels12(x[:, -1])
                if self.modality == "event"
                else raw_rgb_statistics(x[:, -1])
            )
            diag_a5 = torch.stack([a5[k] for k in ("flow", "margin", "log_variance")], -1)
            if self.student is not None:
                raw = x.new_zeros((len(x), 17))
                raw[:, :2], raw[:, 2:5], raw[:, 8] = base, diag_a5, a5["point_phase"]
                feature = self.student(raw)
            else:
                c2f = self.models[1](x, dt)
                diag_c2f = torch.stack([c2f[k] for k in ("flow", "margin", "log_variance")], -1)
                pair_input = NativeObservation.from_output(a5).pair_input(dt[:, -1])
                pair_phase = emitted_phase(pair_point_phase(self.pair, pair_input))
                phases = torch.stack((a5["point_phase"], c2f["point_phase"], pair_phase), -1)
                feature = event_phase17(base, diag_a5, diag_c2f, phases)
            if feature.shape != (self.batch_size, 17) or not torch.isfinite(feature).all():
                raise ValueError("Invalid V13 PHASE17 inference output")
            result.append(feature[:n].clone())
        if not result:
            return torch.empty((0, 17), device=self.device)
        return torch.cat(result)


@dataclass(frozen=True)
class Request:
    """Causal observation identity plus lazy construction of its sensor inputs."""

    anchor: int
    available: int
    windows: tuple[tuple[int, int], ...]
    roi: tuple[float, ...]
    delta: tuple[float, ...]
    load: Callable[[], Tensor]  # [T,C,H,W], never targets

    def validate(self, now: int) -> None:
        """Bound all sensor history and ROI availability before calling the reader."""
        roi_distance(self.roi, self.roi)
        if not self.windows or not self.anchor <= self.available <= now:
            raise ValueError("Observation or ROI is unavailable at query time")
        if any(a >= b or b > self.anchor or now - a > 650000 for a, b in self.windows):
            raise ValueError("Observation exceeds causal 650 ms sensor history")
        if (
            len(self.delta) not in (1, 2)
            or not np.isfinite(self.delta).all()
            or min(self.delta) <= 0
        ):
            raise ValueError("Measured T2/T3 intervals required")


class NativeStream:
    """Bounded single-track inference with V13 normalization and real clocks."""

    def __init__(
        self, experts: NativeExperts, normalizer: FrozenNormalizer, *, policy: Policy | None = None
    ) -> None:
        normalizer.validate_endpoint(
            modality=experts.modality,
            fit_role="H",
            schema_sha256=experts.schema,
            producer_sha256=experts.parents,
        )
        self.experts, self.normalizer = experts, normalizer
        # Exact reuse is the default; approximation must be selected explicitly.
        self.state = FeatureState(
            policy or Policy(time_tolerance_us=0, min_roi_iou=1, max_log_scale=0)
        )
        self.deltas: dict[int, tuple[float, ...]] = {}

    @torch.inference_mode()
    def prepare(
        self, sequence: str, track: str, now: int, requests: Sequence[Request]
    ) -> dict[str, Any]:
        """Return inputs shared by native temporal heads and multimodal fusion."""
        if not requests or any(
            a.anchor >= b.anchor for a, b in zip(requests, requests[1:], strict=False)
        ):
            raise ValueError("Nonempty ordered distinct observations required")
        selected = list(requests)[-self.state.policy.history :]
        for request in selected:
            request.validate(now)
        if self.state.begin(sequence, track, now):
            self.deltas.clear()
        used: set[int] = set()
        observations: list[Observation | None] = [None] * len(selected)
        fresh = []
        missing: dict[tuple[int, ...], list[tuple[int, Request, Tensor]]] = {}
        exact = (
            self.state.policy.time_tolerance_us == 0
            and self.state.policy.min_roi_iou == 1
            and self.state.policy.max_log_scale == 0
        )
        for index, request in enumerate(selected):
            # Different real T2/T3 layouts cannot be compared by broadcasting.
            compatible = [o for o in self.state.entries if len(o.windows) == len(request.windows)]
            entries, self.state.entries = self.state.entries, compatible
            try:
                item = self.state.find(
                    anchor=request.anchor,
                    now=now,
                    windows=request.windows,
                    roi=request.roi,
                    delta=request.delta[-1],
                    used=used,
                )
            finally:
                self.state.entries = entries
            if item is not None and (
                self.deltas.get(id(item)) != request.delta
                or (exact and item.available != request.available)
            ):
                item = None
            if item is None:
                sensor = request.load()
                if sensor.ndim != 4 or len(sensor) != len(request.delta) + 1:
                    raise ValueError("Reader padded or changed the real frame count")
                missing.setdefault(tuple(sensor.shape), []).append((index, request, sensor))
            else:
                used.add(id(item))
                observations[index] = item
        for group in missing.values():
            features = self.experts(
                torch.stack([row[2] for row in group]),
                torch.tensor([row[1].delta for row in group]),
            )
            for (index, request, _), feature in zip(group, features, strict=True):
                item = Observation(
                    request.anchor,
                    request.windows,
                    request.roi,
                    request.delta[-1],
                    request.available,
                    now,
                    feature,
                )
                observations[index] = item
                fresh.append(item)
                self.deltas[id(item)] = request.delta
        if any(o is None for o in observations):
            raise RuntimeError("Incomplete observation execution")
        complete = cast(list[Observation], observations)
        raw = torch.stack([o.feature for o in complete])
        normalized = self.normalizer.transform(raw.cpu().numpy())
        timing = clock_features([o.anchor for o in complete], [o.available for o in complete], now)
        self.state.commit(fresh)
        retained = {id(o) for o in self.state.entries}
        self.deltas = {key: value for key, value in self.deltas.items() if key in retained}
        device = self.experts.device
        return {
            "features": torch.from_numpy(normalized)[None].to(device),
            "timing": torch.from_numpy(timing)[None].to(device),
            "valid": torch.ones((1, len(observations)), dtype=torch.bool, device=device),
            "expert_phase": raw[-1:, 8:11],
            "raw": raw,
            "reused": len(used),
            "computed": len(fresh),
            "anchors": [o.anchor for o in complete],
        }


def temporal_forward(
    head: nn.Module, prepared: Mapping[str, Any], *, history: int = 8, fixed_shape: bool = True
) -> dict[str, Tensor]:
    """Apply native temporal heads with optional masked leading padding."""
    from operational.streaming_revision.kernels import pad_head_inputs

    if history not in (1, 2, 4, 8):
        raise ValueError("Unsupported inference history")
    timing = prepared["timing"][:, -history:].clone()
    xs = (
        prepared["features"][:, -history:],
        timing,
        prepared["valid"][:, -history:],
        prepared["expert_phase"],
    )
    return head(*(pad_head_inputs(xs, history) if fixed_shape else xs))


def fusion_forward(
    head: nn.Module, event: Mapping[str, Any], rgb: Mapping[str, Any]
) -> dict[str, Tensor]:
    """Keep native multimodal argument order and separate modality normalizers."""
    return head(
        event["features"],
        event["timing"],
        event["valid"],
        event["expert_phase"],
        rgb["features"],
        rgb["timing"],
        rgb["valid"],
    )

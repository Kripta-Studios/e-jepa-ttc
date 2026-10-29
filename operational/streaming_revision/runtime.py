"""Experimental stream runner with honest feature timestamps and reduced histories."""

from __future__ import annotations

import copy
import time
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer.models import FrozenModels
from operational.ttc_revision.runtime import device_features

from .kernels import EncoderPrecision, GraphModule, install_transport, pad_head_inputs
from .preparation import IncrementalPreparer, Query
from .state import FeatureState, Observation, Policy, roi_distance


class StreamRuntime:
    """An independent experimental runtime; all modifications are on copied modules."""

    def __init__(
        self,
        frozen: FrozenModels,
        policy: Policy,
        *,
        precision: str = "fp32",
        vectorized: bool = False,
        compile_encoder: bool = False,
        student: nn.Module | None = None,
        compile_full: bool = False,
        producer_batch: int = 0,
        pad_heads: bool = False,
    ) -> None:
        if producer_batch not in (0, 1, 2, 4, 8):
            raise ValueError("producer batch must be dynamic or 1/2/4/8")
        self.producer_batch = producer_batch
        self.pad_heads = pad_heads
        self.device = frozen.device
        self.models = {name: copy.deepcopy(model) for name, model in frozen.models.items()}
        for name in ("A5", "C2F"):
            model = self.models[name]
            if vectorized:
                install_transport(model)
            if precision != "fp32" or compile_encoder:
                model.encoder = EncoderPrecision(
                    cast(nn.Module, model.encoder), precision, compile_graph=compile_encoder
                )
        self.heads = frozen.heads
        if compile_full:
            self.models = {
                name: GraphModule(model) if name != "PAIR" else model
                for name, model in self.models.items()
            }
            self.heads = {
                seed: GraphModule(copy.deepcopy(head)) for seed, head in frozen.heads.items()
            }
        self.mean = torch.as_tensor(frozen.mean, dtype=torch.float64, device=self.device)
        self.scale = torch.as_tensor(frozen.scale, dtype=torch.float64, device=self.device)
        self.state = FeatureState(policy)
        self.student = student.to(self.device).eval() if student is not None else None
        self.configuration = {
            "precision": precision,
            "vectorized": vectorized,
            "compile_encoder": compile_encoder,
            "compile_full": compile_full,
            "producer_batch": producer_batch,
            "pad_heads": pad_heads,
            "distilled": student is not None,
        }

    def synchronize(self) -> None:
        """Synchronize only at measurement boundaries."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def extract(self, events: Tensor, delta: float) -> Tensor:
        """Run the required producers only for newly encoded observations."""
        events = events.to(self.device)
        if self.producer_batch:
            parts = []
            for part in events.split(self.producer_batch):
                count = len(part)
                if count < self.producer_batch:
                    part = torch.cat(
                        (part, part[-1:].expand(self.producer_batch - count, -1, -1, -1, -1))
                    )
                parts.append(self._extract(part, delta)[:count])
            return torch.cat(parts)
        return self._extract(events, delta)

    def _extract(self, events: Tensor, delta: float) -> Tensor:
        """Producer invocation with one batch shape; padded outputs are discarded."""
        dt = torch.full((len(events), 2), delta, device=self.device, dtype=torch.float32)
        if self.student is None:
            return device_features(self.models, events, dt)
        output = self.models["A5"](events, dt, return_dense_features=True)
        raw = torch.zeros((len(events), 17), device=self.device)
        raw[:, :2] = events[:, -1, -2:].mean((-2, -1))
        raw[:, 2] = output.diagnostics["transport_flow_magnitude"][:, -1].float()
        raw[:, 3] = torch.minimum(
            output.log_height_ratio[:, -1].abs() / 0.002, output.sensor_support[:, -1] / 0.0001
        ).float()
        raw[:, 4] = output.ttc_log_variance.float()
        raw[:, 8] = (-torch.log1p(-0.1 / output.ttc_mean_seconds.double())).float()
        return self.student(raw).float()

    @torch.inference_mode()
    def heads_from_observations(self, observations: list[Observation], query: Query) -> Tensor:
        """Use actual source ages, never pretend approximated observations are exact."""
        raw = torch.stack([o.feature for o in observations])
        count = len(observations)
        times = raw.new_zeros((1, count, 4))
        anchors = np.asarray([o.anchor for o in observations])
        available = np.asarray([o.available for o in observations])
        times[0, :, 0] = torch.as_tensor((query.anchor - anchors) / 1e6, device=self.device)
        times[0, :, 1] = torch.as_tensor((query.available - available) / 1e6, device=self.device)
        times[0, :, 3] = torch.as_tensor((available - anchors) / 1e6, device=self.device)
        times[0, 1:, 2] = torch.as_tensor(np.diff(anchors) / 1e6, device=self.device)
        xs = (
            ((raw.double() - self.mean) / self.scale).float()[None],
            times,
            torch.ones((1, count), dtype=torch.bool, device=self.device),
            raw[-1:, 8:11],
        )
        if self.pad_heads:
            xs = pad_head_inputs(xs, self.state.policy.history)
        return torch.cat([phase_to_ttc(head(*xs)["point_phase"]) for head in self.heads.values()])

    @torch.inference_mode()
    def predict(self, query: Query, preparer: IncrementalPreparer) -> dict[str, Any]:  # noqa: ANN401
        """Return TTC and per-query cost/reuse receipts, including cold-start work."""
        self.synchronize()
        started = time.perf_counter()
        reset = self.state.begin(query.sequence, query.track, query.anchor)
        if reset:
            preparer.reset()
        h = self.state.policy.history
        anchors = [query.anchor - i * 50000 for i in range(h - 1, -1, -1)]
        anchors = [a for a, valid in zip(anchors, query.valid[-h:], strict=True) if valid]
        used: set[int] = set()
        selected: dict[int, Observation] = {}
        for anchor in anchors:
            item = self.state.find(
                anchor=anchor,
                now=query.anchor,
                windows=query.at(anchor),
                roi=query.roi,
                delta=query.delta,
                used=used,
            )
            if item is not None:
                selected[anchor] = item
                used.add(id(item))
        missing = [a for a in anchors if a not in selected]
        events = preparer.prepare(query, missing)
        prepared = time.perf_counter()
        features = self.extract(events, query.delta)
        self.synchronize()
        producers_done = time.perf_counter()
        new = [
            Observation(
                a, query.at(a), query.roi, query.delta, query.available, query.anchor, feature
            )
            for a, feature in zip(missing, features, strict=True)
        ]
        for item in new:
            selected[item.anchor] = item
        observations = [selected[a] for a in anchors]
        if any(a.anchor >= b.anchor for a, b in zip(observations, observations[1:], strict=False)):
            raise ValueError("reuse would create unordered or repeated temporal observations")
        prediction = self.heads_from_observations(observations, query).cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite experimental prediction")
        self.state.commit(new)
        finished = time.perf_counter()
        return {
            "ttc": float(np.median(prediction)),
            "seeds_ttc": prediction.tolist(),
            "history": h,
            "valid_observations": len(anchors),
            "reused": len(used),
            "computed": len(new),
            "reset": reset,
            "source_anchors": [o.anchor for o in observations],
            "max_time_approximation_us": max(
                a - o.anchor for a, o in zip(anchors, observations, strict=True)
            ),
            "min_source_roi_iou": min(roi_distance(o.roi, query.roi)[0] for o in observations),
            "cpu_ms": (prepared - started) * 1000,
            "producer_ms": (producers_done - prepared) * 1000,
            "head_and_commit_ms": (finished - producers_done) * 1000,
            "total_ms": (finished - started) * 1000,
            **preparer.last_diagnostics,
        }

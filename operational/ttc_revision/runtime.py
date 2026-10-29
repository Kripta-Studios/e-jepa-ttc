"""Explicitly versioned H8 execution with optional compact/device-resident paths."""

# ruff: noqa: ANN401 -- existing frozen model diagnostics are dynamic.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer.models import FrozenModels
from operational.simplex_t_shared_route.adapter import extract, inputs


@torch.inference_mode()
def device_features(models: dict[str, Any], events: Tensor, delta: Tensor) -> Tensor:
    """Compute PHASE17 on-device; admission must check finite-precision parity."""
    values = torch.zeros((len(events), 17), dtype=torch.float64, device=events.device)
    values[:, :2] = events[:, -1, -2:].mean((-2, -1)).double()
    a5 = None
    for name, offset, column in (("A5", 2, 8), ("C2F", 5, 9)):
        output = models[name](events, delta, return_dense_features=True)
        if name == "A5":
            a5 = output
        ttc = output.ttc_mean_seconds.float().double()
        values[:, offset] = output.diagnostics["transport_flow_magnitude"][:, -1].float()
        values[:, offset + 1] = torch.minimum(
            output.log_height_ratio[:, -1].abs() / 0.002, output.sensor_support[:, -1] / 0.0001
        ).float()
        values[:, offset + 2] = output.ttc_log_variance.float()
        values[:, column] = -torch.log1p(-0.1 / ttc)
    assert a5 is not None
    elapsed = delta[:, 1]
    support = a5.sensor_support
    pair = torch.cat(
        (
            a5.pair_tokens[:, -1].float(),
            torch.stack((elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()), -1),
            support[:, -1:],
            torch.minimum(support[:, -2], support[:, -1])[:, None],
        ),
        -1,
    )
    pair_ttc = models["PAIR"].predict_ttc(PairFeatureBatch(pair.float())).float().double()
    values[:, 10] = -torch.log1p(-0.1 / pair_ttc)
    values[:, 11] = values[:, 10] - values[:, 8]
    values[:, 12] = values[:, 10] - values[:, 9]
    values[:, 13] = values[:, 9] - values[:, 8]
    values[:, 14:17] = values[:, 11:14].abs()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite producer features")
    return values.float()


class H8Runtime:
    """Reuse frozen weights; changes of numerical execution are named explicitly."""

    def __init__(self, frozen: FrozenModels, *, batch: int = 16, on_device: bool = False) -> None:
        if batch not in (8, 16):
            raise ValueError("producer batch must be 8 or 16")
        self.frozen, self.batch, self.on_device = frozen, batch, on_device
        self.mean = torch.as_tensor(frozen.mean, device=frozen.device, dtype=torch.float64)
        self.scale = torch.as_tensor(frozen.scale, device=frozen.device, dtype=torch.float64)

    @torch.inference_mode()
    def features(self, own_events: np.ndarray) -> Tensor:
        """Return raw [8,17] features; no target access or optimizer."""
        events, _ = self.frozen._own_arrays(own_events, None)
        tensor = torch.from_numpy(events).to(self.frozen.device)
        if self.batch == 16:
            tensor = torch.cat((torch.zeros_like(tensor), tensor), 0)
        delta = torch.full((self.batch, 2), 0.1, dtype=torch.float32, device=tensor.device)
        if self.on_device:
            return device_features(self.frozen.models, tensor, delta)[-8:]
        raw = extract("H8_SEED7", self.frozen.models, tensor, delta)[-8:]
        return torch.from_numpy(raw).to(tensor.device)

    @torch.inference_mode()
    def predict_from_features(self, raw: Tensor, *, seeds: Sequence[int] = (7, 13, 23)) -> Tensor:
        """Evaluate selected heads once; output tensor stays on the device."""
        if raw.shape != (8, 17) or not seeds or any(s not in self.frozen.heads for s in seeds):
            raise ValueError("invalid features or seed selection")
        if self.on_device:
            features = ((raw.double() - self.mean) / self.scale).float()[None]
            times = raw.new_zeros((1, 8, 4))
            lags = (torch.arange(7, -1, -1, device=raw.device, dtype=torch.float64) / 20).float()
            times[0, :, 0], times[0, :, 3] = lags, lags
            times[0, 1:, 2] = 0.05
            xs = (
                features,
                times,
                torch.ones((1, 8), dtype=torch.bool, device=raw.device),
                raw[-1:, 8:11],
            )
        else:
            xs = tuple(
                x.to(raw.device)
                for x in inputs(
                    "H8_SEED7",
                    raw.cpu().numpy(),
                    np.ones(8, bool),
                    np.arange(350000, -1, -50000, dtype=np.int64),
                    0,
                    0,
                    self.frozen.mean,
                    self.frozen.scale,
                )
            )
        result = torch.cat(
            [phase_to_ttc(self.frozen.heads[seed](*xs)["point_phase"]) for seed in seeds]
        )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite head output")
        return result

    def predict(self, own_events: np.ndarray, *, seeds: Sequence[int] = (7, 13, 23)) -> np.ndarray:
        """Complete CPU input to CPU output path, including data transfers."""
        return self.predict_from_features(self.features(own_events), seeds=seeds).cpu().numpy()

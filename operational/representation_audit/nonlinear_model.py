"""A zero-initialized nonlinear height readout on frozen SP1 features."""

from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

from .spatial_model import SpatialReadout


class NonlinearSpatialReadout(SpatialReadout):
    """Preserve the original token route and add only a supervised height MLP."""

    def __init__(self, native: CausalScaleTTC, calibration: dict) -> None:
        super().__init__(native, calibration, "SP1")
        final = nn.Linear(64, 1)
        self.height_mlp = nn.Sequential(nn.Linear(512, 64), nn.SiLU(), final)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.height_mlp.requires_grad_(False)
        self.height_variance = float(calibration["variance_height_target"])
        self.temporal_variance = float(calibration["variance_temporal_target"])

    def height_delta(self, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = cache["height_features"]
        if raw.shape != (len(raw), 3, 512):
            raise ValueError("Expected the original three SP1 observations")
        global_values = torch.nn.functional.pad(raw.reshape(-1, 3, 64, 8).mean(-1), (0, 448))
        inputs = self.spatial_active * raw + (1 - self.spatial_active) * global_values
        z = (inputs - self.height_mean) / self.height_scale
        linear = self.height_intercept + z @ self.height_coefficient
        return (linear + self.height_mlp(z).squeeze(-1)).clamp(-2, 2)

    def corrected_inputs(self, cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(key.startswith(("target_", "dino_")) for key in cache):
            raise ValueError("Training targets are forbidden in inference inputs")
        delta = self.height_delta(cache)
        changed = {key: value for key, value in cache.items() if key != "height_features"}
        changed["analytic"] = cache["analytic"] + (delta[:, 1:] - delta[:, :-1])
        changed["visible_height"] = cache["visible_height"] * delta.exp()
        changed["log_height_correction"] = cache["log_height_correction"] + delta
        return changed

    def geometry_loss(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Use only P box heights and their changes, never TTC labels."""
        delta = self.height_delta(values)
        valid = values["target_valid"].bool()
        height = values["target_height_normalized"].clamp_min(1e-6)
        original = values["visible_height"].clamp_min(1e-6)
        y = height.log() - original.log()
        pairs = valid[:, -1] & valid[:, -2]
        if not valid.any() or not pairs.any():
            raise ValueError("Missing geometry training support")
        target = height[:, -1].log() - height[:, -2].log() - values["analytic"][:, -1]
        return (delta[valid] - y[valid]).square().mean() / self.height_variance + (
            (delta[:, -1] - delta[:, -2] - target)[pairs].square().mean() / self.temporal_variance
        )

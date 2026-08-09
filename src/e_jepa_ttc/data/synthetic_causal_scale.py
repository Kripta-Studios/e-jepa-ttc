"""Deterministic causal event fixtures for learning the v5 foreground-scale path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticCausalScaleConfig:
    """Controls for independent train/validation/test synthetic sequence groups."""

    samples: int = 512
    seed: int = 101
    canvas_size: int = 64
    endpoints: int = 3
    polarity_bins: int = 5
    delta_t_s: float = 0.1
    accumulation_s: float = 0.05
    micro_frames: int = 6
    background_events_per_endpoint: int = 8
    hot_pixel_probability: float = 0.1
    empty_probability: float = 0.1

    def __post_init__(self) -> None:
        if self.samples <= 0 or self.seed < 0:
            raise ValueError("samples must be positive and seed non-negative")
        if self.canvas_size < 32 or self.endpoints != 3:
            raise ValueError("synthetic causal scale requires a >=32 canvas and three endpoints")
        if self.polarity_bins != 5 or self.micro_frames != self.polarity_bins + 1:
            raise ValueError("the v5 event contract requires five bins and six micro frames")
        if self.delta_t_s <= 0.0 or not 0.0 < self.accumulation_s < self.delta_t_s:
            raise ValueError("accumulation must be positive and shorter than endpoint spacing")
        if self.background_events_per_endpoint < 0:
            raise ValueError("background event count must be non-negative")
        probabilities = (self.hot_pixel_probability, self.empty_probability)
        if any(not 0.0 <= value < 1.0 for value in probabilities):
            raise ValueError("synthetic probabilities must lie in [0,1)")

    @property
    def channels(self) -> int:
        """Five bins per polarity plus two aggregate support channels."""

        return 2 * self.polarity_bins + 2


class SyntheticCausalScaleSample(TypedDict):
    """One collatable synthetic event/foreground/TTC sample."""

    inputs: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_seconds: torch.Tensor
    target_log_ratio: torch.Tensor
    target_valid: torch.Tensor
    target_masks: torch.Tensor
    mask_valid: torch.Tensor
    direction: torch.Tensor
    shape: str
    sample_id: str


@dataclass(frozen=True)
class _Trajectory:
    current_height: float
    aspect: float
    current_ttc: float
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    ellipse: bool


class SyntheticCausalScaleDataset(Dataset[SyntheticCausalScaleSample]):
    """Generate event endpoints from moving rectangle/ellipse mask differences."""

    def __init__(self, config: SyntheticCausalScaleConfig | None = None) -> None:
        self.config = config or SyntheticCausalScaleConfig()
        size = self.config.canvas_size
        coordinates = (torch.arange(size, dtype=torch.float32) + 0.5) / float(size)
        self._y, self._x = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def __len__(self) -> int:
        return self.config.samples

    def _generator(self, index: int) -> torch.Generator:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed * 1_000_003 + index * 97_409)
        return generator

    @staticmethod
    def _uniform(generator: torch.Generator, low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand((), generator=generator).item())

    def _mask(
        self,
        *,
        center_x: float,
        center_y: float,
        height: float,
        width: float,
        ellipse: bool,
    ) -> torch.Tensor:
        half_h = max(height / (2.0 * self.config.canvas_size), 1.0e-4)
        half_w = max(width / (2.0 * self.config.canvas_size), 1.0e-4)
        x = (self._x - center_x) / half_w
        y = (self._y - center_y) / half_h
        if ellipse:
            return (x.square() + y.square() <= 1.0).to(torch.float32)
        return ((x.abs() <= 1.0) & (y.abs() <= 1.0)).to(torch.float32)

    def _state_mask(
        self,
        time_s: float,
        *,
        current_height: float,
        aspect: float,
        current_ttc: float,
        center_x: float,
        center_y: float,
        velocity_x: float,
        velocity_y: float,
        ellipse: bool,
    ) -> torch.Tensor:
        denominator = current_ttc - time_s
        if abs(denominator) < 1.0e-4:
            raise ValueError("synthetic trajectory crossed its projection singularity")
        scale = current_ttc / denominator
        if scale <= 0.0:
            raise ValueError("synthetic apparent scale became non-positive")
        return self._mask(
            center_x=center_x + velocity_x * time_s / self.config.canvas_size,
            center_y=center_y + velocity_y * time_s / self.config.canvas_size,
            height=current_height * scale,
            width=current_height * aspect * scale,
            ellipse=ellipse,
        )

    def _event_endpoint(
        self,
        endpoint_s: float,
        *,
        generator: torch.Generator,
        polarity_flip: bool,
        trajectory: _Trajectory,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        micro_times = torch.linspace(
            endpoint_s - self.config.accumulation_s,
            endpoint_s,
            self.config.micro_frames,
        )
        masks = [
            self._state_mask(
                float(time.item()),
                current_height=trajectory.current_height,
                aspect=trajectory.aspect,
                current_ttc=trajectory.current_ttc,
                center_x=trajectory.center_x,
                center_y=trajectory.center_y,
                velocity_x=trajectory.velocity_x,
                velocity_y=trajectory.velocity_y,
                ellipse=trajectory.ellipse,
            )
            for time in micro_times
        ]
        differences = torch.stack(
            [current - previous for previous, current in zip(masks, masks[1:], strict=False)]
        )
        positive = differences.clamp_min(0.0)
        negative = (-differences).clamp_min(0.0)
        if polarity_flip:
            positive, negative = negative, positive
        event = torch.cat(
            [
                positive,
                negative,
                positive.amax(dim=0, keepdim=True),
                negative.amax(dim=0, keepdim=True),
            ],
            dim=0,
        )
        for _ in range(self.config.background_events_per_endpoint):
            channel = int(torch.randint(0, 2 * self.config.polarity_bins, (), generator=generator))
            y = int(torch.randint(0, self.config.canvas_size, (), generator=generator))
            x = int(torch.randint(0, self.config.canvas_size, (), generator=generator))
            event[channel, y, x] = 1.0
        if float(torch.rand((), generator=generator)) < self.config.hot_pixel_probability:
            y = int(torch.randint(0, self.config.canvas_size, (), generator=generator))
            x = int(torch.randint(0, self.config.canvas_size, (), generator=generator))
            event[: 2 * self.config.polarity_bins, y, x] = 1.0
        return event, masks[-1]

    @staticmethod
    def _visible_pixel_height(mask: torch.Tensor) -> torch.Tensor:
        return mask.bool().any(dim=-1).sum().to(torch.float32).clamp_min(1.0)

    def __getitem__(self, index: int) -> SyntheticCausalScaleSample:
        generator = self._generator(index)
        empty = float(torch.rand((), generator=generator)) < self.config.empty_probability
        endpoint_times = torch.arange(
            -(self.config.endpoints - 1),
            1,
            dtype=torch.float32,
        ) * self.config.delta_t_s
        delta = torch.full((self.config.endpoints - 1,), self.config.delta_t_s)
        if empty:
            return {
                "inputs": torch.zeros(
                    self.config.endpoints,
                    self.config.channels,
                    self.config.canvas_size,
                    self.config.canvas_size,
                ),
                "delta_t_s": delta,
                "target_ttc_seconds": torch.tensor(0.0),
                "target_log_ratio": torch.tensor(0.0),
                "target_valid": torch.tensor(False),
                "target_masks": torch.zeros(
                    self.config.endpoints,
                    1,
                    self.config.canvas_size,
                    self.config.canvas_size,
                ),
                "mask_valid": torch.ones(self.config.endpoints, dtype=torch.bool),
                "direction": torch.tensor(0, dtype=torch.int64),
                "shape": "empty",
                "sample_id": f"synthetic-{self.config.seed}-{index}",
            }

        approaching = bool(torch.randint(0, 2, (), generator=generator).item())
        magnitude = self._uniform(generator, 0.7, 2.0)
        current_ttc = magnitude if approaching else -magnitude
        current_height = self._uniform(generator, 20.0, 30.0)
        aspect = self._uniform(generator, 0.75, 1.25)
        center_x = self._uniform(generator, 0.42, 0.58)
        center_y = self._uniform(generator, 0.42, 0.58)
        velocity_x = self._uniform(generator, -18.0, 18.0)
        velocity_y = self._uniform(generator, -12.0, 12.0)
        ellipse = bool(torch.randint(0, 2, (), generator=generator).item())
        polarity_flip = bool(torch.randint(0, 2, (), generator=generator).item())
        trajectory = _Trajectory(
            current_height=current_height,
            aspect=aspect,
            current_ttc=current_ttc,
            center_x=center_x,
            center_y=center_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            ellipse=ellipse,
        )
        endpoint_values = [
            self._event_endpoint(
                float(time.item()),
                generator=generator,
                polarity_flip=polarity_flip,
                trajectory=trajectory,
            )
            for time in endpoint_times
        ]
        inputs = torch.stack([value[0] for value in endpoint_values])
        masks = torch.stack([value[1] for value in endpoint_values]).unsqueeze(1)
        previous_height = self._visible_pixel_height(masks[-2, 0])
        current_height_pixels = self._visible_pixel_height(masks[-1, 0])
        log_ratio = current_height_pixels.log() - previous_height.log()
        valid = bool(log_ratio.abs() >= 2.0e-3)
        inverse_ttc = torch.expm1(log_ratio) / self.config.delta_t_s
        target_ttc = torch.reciprocal(inverse_ttc) if valid else torch.tensor(0.0)
        if valid and bool(torch.sign(target_ttc) != (1.0 if approaching else -1.0)):
            raise RuntimeError("rasterized synthetic direction differs from its trajectory")
        return {
            "inputs": inputs,
            "delta_t_s": delta,
            "target_ttc_seconds": target_ttc.to(torch.float32),
            "target_log_ratio": log_ratio.to(torch.float32),
            "target_valid": torch.tensor(valid),
            "target_masks": masks,
            "mask_valid": torch.ones(self.config.endpoints, dtype=torch.bool),
            "direction": torch.tensor(1 if approaching else -1, dtype=torch.int64),
            "shape": "ellipse" if ellipse else "rectangle",
            "sample_id": f"synthetic-{self.config.seed}-{index}",
        }


def synthetic_scale_config_identity(config: SyntheticCausalScaleConfig) -> str:
    """Return a stable human-readable identity for split provenance."""

    return (
        f"causal-scale-synthetic-v1:seed={config.seed}:samples={config.samples}:"
        f"canvas={config.canvas_size}:dt={config.delta_t_s:.6f}"
    )


__all__ = [
    "SyntheticCausalScaleConfig",
    "SyntheticCausalScaleDataset",
    "SyntheticCausalScaleSample",
    "synthetic_scale_config_identity",
]

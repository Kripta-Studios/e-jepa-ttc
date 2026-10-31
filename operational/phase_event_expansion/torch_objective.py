"""Optional cached Torch objective; independent of frozen solver/deployment code."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class SymmetricGaussian(nn.Module):
    """SciPy ndimage reflect padding includes the boundary pixel, unlike torch reflect."""

    vertical: torch.Tensor
    horizontal: torch.Tensor
    indices: torch.Tensor

    def __init__(
        self,
        size: int = 128,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        radius = 3  # scipy int(truncate4 * sigma.8 + .5)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * offsets**2 / 0.8**2)
        weights /= weights.sum()
        kernel = torch.tensor(weights, dtype=dtype, device=device)
        self.register_buffer("vertical", kernel.reshape(1, 1, -1, 1).repeat(4, 1, 1, 1))
        self.register_buffer("horizontal", kernel.reshape(1, 1, 1, -1).repeat(4, 1, 1, 1))
        indices = torch.arange(-radius, size + radius, device=device) % (2 * size)
        indices = torch.where(indices < size, indices, 2 * size - 1 - indices)
        self.register_buffer("indices", indices)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply y then x convolution, matching SciPy's separable-axis order."""
        image = functional.conv2d(image.index_select(2, self.indices), self.vertical, groups=4)
        return functional.conv2d(image.index_select(3, self.indices), self.horizontal, groups=4)


class CachedTorchObjective(nn.Module):
    """Preload one event window and evaluate endpoint NCC for batches of q,bx,by.

    This implements only the frozen objective, not optimization or acceptance.
    Four channels represent early+/early-/late+/late- on a fixed 128² canvas.
    Float64 is the default. Float32 is explicitly experimental, requiring parity.
    """

    points: torch.Tensor
    age: torch.Tensor
    channel_offset: torch.Tensor

    def __init__(
        self,
        points: np.ndarray,
        age_seconds: np.ndarray,
        polarity: np.ndarray,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        points = np.asarray(points, dtype=np.float64)
        age_seconds = np.asarray(age_seconds, dtype=np.float64)
        polarity = np.asarray(polarity)
        if not 1 <= len(points) <= 20_000 or points.shape != (len(points), 2):
            raise ValueError("Expected between 1 and 20000 two-coordinate events")
        if age_seconds.shape != (len(points),) or polarity.shape != (len(points),):
            raise ValueError("Mismatched event arrays")
        if not np.isfinite(points).all() or not np.isfinite(age_seconds).all():
            raise ValueError("Nonfinite event coordinates or age")
        if np.any(age_seconds < 0) or not np.isin(polarity, [-1, 1]).all():
            raise ValueError("Expected causal ages and signed polarity")
        if dtype not in (torch.float64, torch.float32):
            raise ValueError("Expected explicit float64 or float32 precision")
        early = age_seconds >= (age_seconds.min() + age_seconds.max()) / 2
        channels = (polarity < 0).astype(np.int64) + 2 * (~early).astype(np.int64)
        self.register_buffer("points", torch.tensor(points, dtype=dtype, device=device))
        self.register_buffer("age", torch.tensor(age_seconds, dtype=dtype, device=device))
        self.register_buffer("channel_offset", torch.tensor(channels * 128 * 128, device=device))
        self.gaussian = SymmetricGaussian(128, dtype=dtype, device=device)
        self.count = len(points)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Return [B] objective values; B must be 1..81 and observations stay resident."""
        if parameters.ndim != 2 or parameters.shape[1] != 3 or not 1 <= len(parameters) <= 81:
            raise ValueError("Expected [B,3] parameters with batch size between 1 and 81")
        q = parameters[:, 0, None]
        scale = 1 + q * self.age
        # Preserve operation order from frozen NumPy warp, including the final addition.
        warped = (
            self.points
            + (q[:, :, None] * self.points - parameters[:, None, 1:]) * self.age[:, None]
        )
        coordinate = (warped + 1) * (127 / 2)
        inside = torch.isfinite(coordinate).all(2) & (coordinate >= 0).all(2)
        inside = inside & (coordinate < 127).all(2)
        # Invalid points cannot contribute or index outside the fixed canvas.
        safe = torch.where(inside[:, :, None], coordinate, torch.zeros_like(coordinate))
        lower = torch.floor(safe).to(torch.int64)
        fraction = safe - lower
        image = parameters.new_zeros((len(parameters), 4 * 128 * 128))
        for dx in (0, 1):
            for dy in (0, 1):
                weight_x = fraction[:, :, 0] if dx else 1 - fraction[:, :, 0]
                weight_y = fraction[:, :, 1] if dy else 1 - fraction[:, :, 1]
                weights = weight_x * weight_y * inside
                indices = self.channel_offset + (lower[:, :, 1] + dy) * 128 + lower[:, :, 0] + dx
                image.scatter_add_(1, indices, weights)
        image = self.gaussian(image.reshape(len(parameters), 4, 128, 128))
        image = image - image.mean(dim=(2, 3), keepdim=True)
        first, last = image[:, :2], image[:, 2:]
        norm = torch.sqrt(first.square().sum((1, 2, 3)) * last.square().sum((1, 2, 3)))
        agreement = (first * last).sum((1, 2, 3)) / norm.clamp(min=torch.finfo(image.dtype).tiny)
        agreement = torch.where(norm > 0, agreement, torch.zeros_like(agreement))
        lost = (~inside).to(image.dtype).sum(1) / self.count
        invalid = (scale <= 0).any(1)
        return torch.where(invalid, 1000 + (-scale).clamp(min=0).sum(1), -agreement + 100 * lost)


class CudaGraphObjective:
    """Capture a fixed-shape resident objective once; replay with new parameters.

    Returned output aliases graph-owned storage and is overwritten by the next
    replay. Copy it when retaining predictions. Event caches never change here.
    """

    def __init__(self, model: CachedTorchObjective, batch_size: int) -> None:
        if model.points.device.type != "cuda":
            raise ValueError("CUDA graph requires a CUDA-resident observation cache")
        if not 1 <= batch_size <= 81:
            raise ValueError("Expected graph batch size between 1 and 81")
        self.model = model
        self.parameters = model.points.new_zeros((batch_size, 3))
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=model.points.device)
        stream.wait_stream(torch.cuda.current_stream(model.points.device))
        started = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                model(self.parameters)
        torch.cuda.current_stream(model.points.device).wait_stream(stream)
        torch.cuda.synchronize(model.points.device)
        self.warmup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=stream):
            self.output = model(self.parameters)
        torch.cuda.synchronize(model.points.device)
        self.capture_seconds = time.perf_counter() - started
        self.capture_count = 1
        self.replay_count = 0

    def __call__(self, parameters: torch.Tensor) -> torch.Tensor:
        """Copy only candidate parameters, replay the actual graph, and return its output."""
        if parameters.shape != self.parameters.shape or parameters.dtype != self.parameters.dtype:
            raise ValueError("Graph shape and precision must match capture")
        self.parameters.copy_(parameters)
        self.graph.replay()
        self.replay_count += 1
        return self.output

"""Explicit seed and environment capture for reproducible experiment ledgers."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch without hiding deterministic trade-offs."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    seed_torch_cpu(seed)
    if cuda_is_usable():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def environment_snapshot() -> dict[str, Any]:
    """Return serializable runtime information for an experiment artifact."""

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_is_usable(),
        "cuda_runtime_reported_available": cuda_runtime_reported_available(),
        "cuda_device_count": cuda_device_count(),
        "gpu_name": cuda_device_name(0),
    }


def cuda_runtime_reported_available() -> bool:
    """Return PyTorch's raw CUDA availability flag without propagating errors."""

    try:
        return bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError, OSError):
        return False


def cuda_device_count() -> int:
    """Return the number of visible CUDA devices that can be addressed."""

    try:
        if not cuda_runtime_reported_available():
            return 0
        return max(0, int(torch.cuda.device_count()))
    except (AssertionError, RuntimeError, OSError):
        return 0


def cuda_is_usable() -> bool:
    """Return whether at least one visible CUDA device is addressable."""

    return cuda_device_count() > 0


def cuda_device_name(device: int | torch.device = 0) -> str | None:
    """Return a CUDA device name, or ``None`` when it cannot be queried."""

    if isinstance(device, torch.device) and device.type != "cuda":
        return None
    count = cuda_device_count()
    if count == 0:
        return None
    index = device.index if isinstance(device, torch.device) else device
    index = 0 if index is None else int(index)
    if index < 0 or index >= count:
        return None
    try:
        return str(torch.cuda.get_device_name(index))
    except (AssertionError, RuntimeError, OSError, IndexError):
        return None


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` safely and validate explicit CUDA requests."""

    if requested == "auto":
        return torch.device("cuda", 0) if cuda_is_usable() else torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        count = cuda_device_count()
        index = 0 if device.index is None else int(device.index)
        if count == 0:
            raw_available = cuda_runtime_reported_available()
            msg = (
                "CUDA was explicitly requested, but no visible CUDA device is "
                f"addressable (torch.cuda.is_available()={raw_available}, "
                f"torch.cuda.device_count()={count}). Check CUDA_VISIBLE_DEVICES "
                "or use device='auto'/'cpu'."
            )
            raise RuntimeError(msg)
        if index < 0 or index >= count:
            raise RuntimeError(
                f"CUDA device index {index} is not visible; torch.cuda.device_count()={count}."
            )
    elif device.type != "cpu":
        raise ValueError(f"Unsupported training device type: {device.type!r}.")
    return torch.device("cuda", index) if device.type == "cuda" else device


def seed_torch_cpu(seed: int) -> None:
    """Seed the global CPU generator without touching CUDA initialization."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


__all__ = [
    "cuda_device_count",
    "cuda_device_name",
    "cuda_is_usable",
    "cuda_runtime_reported_available",
    "environment_snapshot",
    "resolve_device",
    "seed_torch_cpu",
    "seed_everything",
]

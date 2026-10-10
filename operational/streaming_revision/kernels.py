"""Bounded vectorized transport and encoder-only reduced precision adapters."""

from __future__ import annotations

import math
import types
from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from e_jepa_ttc.models.local_transport import LocalTransportMatch


def correlation(
    previous: Tensor,
    current: Tensor,
    *,
    radius: int,
    temperature: float,
    return_probability: bool = False,
) -> LocalTransportMatch:
    """Compute all offsets in row stripes; bound materialized patch memory to 16 MiB."""
    if previous.ndim != 4 or previous.shape != current.shape:
        raise ValueError("matching BCHW endpoints required")
    b, c, h, w = previous.shape
    if radius < 1 or min(h, w) <= 2 * radius or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid transport radius/temperature")
    left = F.normalize(previous.float(), dim=1, eps=1e-6)
    right = F.normalize(current.float(), dim=1, eps=1e-6)
    k = 2 * radius + 1
    padded = F.pad(right, (radius, radius, radius, radius))
    stripe = max(1, min(h, (16 * 1024**2) // (max(1, b * c * k * k * w) * 4)))
    parts = []
    for y in range(0, h, stripe):
        rows = min(stripe, h - y)
        patches = F.unfold(padded[:, :, y : y + rows + 2 * radius], kernel_size=k)
        patches = patches.reshape(b, c, k * k, rows, w)
        parts.append(torch.einsum("bchw,bckhw->bkhw", left[:, :, y : y + rows], patches))
    scores = torch.cat(parts, dim=2)
    offsets = torch.arange(-radius, radius + 1, device=left.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dx, dy = dx.flatten(), dy.flatten()
    xs = torch.arange(w, device=left.device)[None, None, :]
    ys = torch.arange(h, device=left.device)[None, :, None]
    valid = (
        (xs + dx[:, None, None] >= 0)
        & (xs + dx[:, None, None] < w)
        & (ys + dy[:, None, None] >= 0)
        & (ys + dy[:, None, None] < h)
    )
    valid = valid[None].expand(b, -1, -1, -1)
    masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    probability = torch.softmax(masked / temperature, 1) * valid
    probability = probability / probability.sum(1, keepdim=True).clamp_min(1e-8)
    top = masked.topk(2, dim=1).values
    entropy = -(probability.clamp_min(1e-12).log() * probability).sum(1)
    return LocalTransportMatch(
        dx=(probability * dx[None, :, None, None]).sum(1),
        dy=(probability * dy[None, :, None, None]).sum(1),
        confidence_margin=(top[:, 0] - top[:, 1]).clamp_min(0),
        entropy=(entropy / valid.sum(1).clamp_min(2).float().log()).clamp(0, 1),
        valid=valid.any(1),
        probability=probability if return_probability else None,
    )


def install_transport(model: nn.Module) -> None:
    """Bind a private function namespace on this instance; never monkeypatch globals."""
    method = cast(types.MethodType, model._forward_impl)
    original = cast(types.FunctionType, method.__func__)
    namespace = dict(original.__globals__, local_correlation_match=correlation)
    replacement = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    model.__dict__["_forward_impl"] = types.MethodType(replacement, model)


def owned_output(value: Any) -> Any:  # noqa: ANN401
    """Materialize graph output ownership recursively before the next replay step."""
    if isinstance(value, Tensor):
        return value.clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{field.name: owned_output(getattr(value, field.name)) for field in fields(value)},
        )
    if isinstance(value, dict):
        return {key: owned_output(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(owned_output(item) for item in value)
    if isinstance(value, list):
        return [owned_output(item) for item in value]
    return value


class GraphModule(nn.Module):
    """Compile complete producers/heads with independent output-buffer lifetimes."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = torch.compile(module, backend="cudagraphs", dynamic=False, fullgraph=False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Retain guards; graph breaks are permitted and their cost remains measured."""
        torch.compiler.cudagraph_mark_step_begin()
        return owned_output(self.module(*args, **kwargs))


def pad_head_inputs(
    xs: tuple[Tensor, Tensor, Tensor, Tensor], length: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Use a fixed history shape with explicitly invalid leading positions."""
    features, times, valid, experts = xs
    padding = length - features.shape[1]
    if padding < 0:
        raise ValueError("head input exceeds the configured history")
    return (
        F.pad(features, (0, 0, padding, 0)),
        F.pad(times, (0, 0, padding, 0)),
        F.pad(valid, (padding, 0), value=False),
        experts,
    )


class GarlForward(nn.Module):
    """Expose the native Garl inference entry point to the same graph wrapper."""

    def __init__(self, native: nn.Module) -> None:
        super().__init__()
        self.native = native

    def forward(self, sensor: Tensor) -> Any:  # noqa: ANN401
        """Preserve native height prediction and auxiliary outputs."""
        return cast(Callable, self.native.forward_test)(sensor)


class EncoderPrecision(nn.Module):
    """Autocast only the encoder; restore all tensor outputs to FP32 for geometry."""

    def __init__(self, encoder: nn.Module, precision: str, *, compile_graph: bool = False) -> None:
        super().__init__()
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("unsupported precision")
        self.precision = precision
        self.compiled = compile_graph
        self.encoder = (
            torch.compile(encoder, backend="cudagraphs", dynamic=False)
            if compile_graph
            else encoder
        )

    def forward(self, inputs: Tensor, **kwargs: Any) -> tuple:  # noqa: ANN401
        """Clone graph outputs before another invocation can overwrite graph buffers."""
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()
        if inputs.device.type == "cpu" and self.precision == "fp16":
            raise ValueError("FP16 encoder trial requires CUDA")
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with torch.autocast(inputs.device.type, dtype=dtype, enabled=self.precision != "fp32"):
            output = self.encoder(inputs, **kwargs)
        return tuple(x.float().clone() if isinstance(x, Tensor) else x for x in output)

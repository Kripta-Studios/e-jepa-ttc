"""Bounded CUDA-graph replay for the canonical H8 transport operations.

The context manager patches only the aliases imported by ``causal_scale_ttc``.
The canonical functions in ``local_transport`` remain unchanged and are used to
build every graph.  This module performs no CUDA work at import time.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock, get_ident
from typing import TypeAlias

import torch

from e_jepa_ttc.models import causal_scale_ttc
from e_jepa_ttc.models.local_transport import (
    LocalTransportMatch,
)
from e_jepa_ttc.models.local_transport import (
    local_correlation_match as _canonical_match,
)
from e_jepa_ttc.models.local_transport import (
    transport_physical_features as _canonical_features,
)

_TensorSignature: TypeAlias = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, str]
_CacheKey: TypeAlias = tuple[object, ...]


def _tensor_signature(value: torch.Tensor) -> _TensorSignature:
    """Return the layout information which must be stable across graph replay."""

    return tuple(value.shape), tuple(value.stride()), value.dtype, str(value.device)


def _flatten_match(match: LocalTransportMatch) -> tuple[torch.Tensor, ...]:
    """Flatten a match without losing whether its probability leaf is absent."""

    leaves = (match.dx, match.dy, match.confidence_margin, match.entropy, match.valid)
    if match.probability is None:
        return leaves
    return (*leaves, match.probability)


def _unflatten_match(leaves: tuple[torch.Tensor, ...]) -> LocalTransportMatch:
    """Reconstruct a match from five or six tensor leaves."""

    if len(leaves) not in (5, 6):
        raise ValueError("a transport match must contain five or six tensor leaves")
    return LocalTransportMatch(
        dx=leaves[0],
        dy=leaves[1],
        confidence_margin=leaves[2],
        entropy=leaves[3],
        valid=leaves[4],
        probability=leaves[5] if len(leaves) == 6 else None,
    )


def _clone_match(match: LocalTransportMatch) -> LocalTransportMatch:
    """Clone every leaf so a later replay cannot overwrite a live match."""

    return _unflatten_match(
        tuple(leaf.clone(memory_format=torch.preserve_format) for leaf in _flatten_match(match))
    )


def _validate_cuda_inputs(
    tensors: tuple[torch.Tensor, ...],
    *,
    bool_indices: frozenset[int] = frozenset(),
) -> torch.device:
    if not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
        raise RuntimeError("H8 transport graph replay requires torch.inference_mode()")
    if not tensors:
        raise ValueError("at least one tensor input is required")
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("H8 transport graph replay accepts CUDA tensors only")
    for index, tensor in enumerate(tensors):
        if tensor.device != device:
            raise ValueError("all transport tensors must be on the same CUDA device")
        expected_dtype = torch.bool if index in bool_indices else torch.float32
        if tensor.dtype != expected_dtype:
            raise ValueError(f"transport tensor {index} must have dtype {expected_dtype}")
        if tensor.requires_grad:
            raise ValueError("H8 transport graph replay does not accept autograd tensors")
    return device


def _static_like(value: torch.Tensor) -> torch.Tensor:
    static = torch.empty_strided(
        tuple(value.shape),
        tuple(value.stride()),
        dtype=value.dtype,
        device=value.device,
    )
    static.copy_(value)
    return static


@dataclass
class _CapturedCall:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        for target, source in zip(self.static_inputs, inputs, strict=True):
            target.copy_(source)
        self.graph.replay()
        return tuple(
            output.clone(memory_format=torch.preserve_format) for output in self.static_outputs
        )


class H8TransportGraph:
    """Install bounded CUDA-graph wrappers for H8 transport inference.

    Args:
        max_entries: Maximum combined match/feature graph signatures retained.
        warmup_calls: Canonical calls executed on a side stream before capture.
    """

    def __init__(self, *, max_entries: int = 8, warmup_calls: int = 3) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if warmup_calls < 1:
            raise ValueError("warmup_calls must be positive")
        self.max_entries = max_entries
        self.warmup_calls = warmup_calls
        self._cache: OrderedDict[_CacheKey, _CapturedCall] = OrderedDict()
        self._lock = RLock()
        self._installed = False
        self._closed = False
        self._captures = 0
        self._replays = 0
        self._evictions = 0
        self._failures = 0
        self._owner: tuple[int, str, int] | None = None

    def snapshot(self) -> dict[str, int | bool]:
        """Return lightweight replay/cache counters without synchronizing CUDA."""

        with self._lock:
            return {
                "cache_entries": len(self._cache),
                "captures": self._captures,
                "replays": self._replays,
                "evictions": self._evictions,
                "failures": self._failures,
                "installed": self._installed,
                "closed": self._closed,
            }

    def close(self) -> None:
        """Release all graph and static-buffer references."""

        with self._lock:
            if self._installed:
                raise RuntimeError("cannot close an installed H8 transport graph context")
            self._cache.clear()
            self._owner = None
            self._closed = True

    def _assert_owner(self, device: torch.device) -> None:
        """Confine static-buffer replay to one thread and one CUDA stream."""

        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        candidate = (get_ident(), str(device), stream_id)
        if self._owner is None:
            self._owner = candidate
        elif candidate != self._owner:
            raise RuntimeError(
                "H8 transport graph replay is confined to its first thread and CUDA stream"
            )

    @contextmanager
    def install(self) -> Iterator[H8TransportGraph]:
        """Patch the two imported aliases and restore them on every exit path."""

        with self._lock:
            if self._closed:
                raise RuntimeError("H8 transport graph context is closed")
            if self._installed:
                raise RuntimeError("H8 transport graph context is already installed")
            old_match = causal_scale_ttc.local_correlation_match
            old_features = causal_scale_ttc.transport_physical_features
            if old_match is not _canonical_match or old_features is not _canonical_features:
                raise RuntimeError("causal_scale_ttc transport aliases are already patched")
            causal_scale_ttc.local_correlation_match = self.local_correlation_match
            causal_scale_ttc.transport_physical_features = self.transport_physical_features
            self._installed = True
        try:
            yield self
        finally:
            with self._lock:
                causal_scale_ttc.local_correlation_match = old_match
                causal_scale_ttc.transport_physical_features = old_features
                self._installed = False

    def _capture(
        self,
        inputs: tuple[torch.Tensor, ...],
        invoke: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    ) -> _CapturedCall:
        device = inputs[0].device
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        static_inputs: tuple[torch.Tensor, ...]
        static_outputs: tuple[torch.Tensor, ...]
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(side):
                static_inputs = tuple(_static_like(value) for value in inputs)
                static_outputs = invoke(static_inputs)
                for _ in range(self.warmup_calls - 1):
                    static_outputs = invoke(static_inputs)
                with torch.cuda.graph(graph, stream=side):
                    static_outputs = invoke(static_inputs)
        except Exception:
            self._failures += 1
            raise
        finally:
            current.wait_stream(side)
        self._captures += 1
        return _CapturedCall(graph, static_inputs, static_outputs)

    def _run(
        self,
        key: _CacheKey,
        inputs: tuple[torch.Tensor, ...],
        invoke: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        with self._lock:
            if self._closed:
                raise RuntimeError("H8 transport graph context is closed")
            entry = self._cache.get(key)
            if entry is None:
                entry = self._capture(inputs, invoke)
                self._cache[key] = entry
                if len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
                    self._evictions += 1
            else:
                self._cache.move_to_end(key)
            outputs = entry.replay(inputs)
            self._replays += 1
            return outputs

    def local_correlation_match(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        *,
        radius: int,
        temperature: float,
        return_probability: bool = False,
    ) -> LocalTransportMatch:
        """Replay the canonical local match and return independently owned leaves."""

        inputs = (previous, current)
        device = _validate_cuda_inputs(inputs)
        self._assert_owner(device)
        key: _CacheKey = (
            "match",
            radius,
            float(temperature),
            return_probability,
            *(_tensor_signature(value) for value in inputs),
        )

        def invoke(static: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return _flatten_match(
                _canonical_match(
                    static[0],
                    static[1],
                    radius=radius,
                    temperature=temperature,
                    return_probability=return_probability,
                )
            )

        return _unflatten_match(self._run(key, inputs, invoke))

    def transport_physical_features(
        self,
        forward: LocalTransportMatch,
        reverse: LocalTransportMatch,
        *,
        foreground_weight: torch.Tensor | None,
        radius: int,
    ) -> torch.Tensor:
        """Replay canonical physical summaries and return an independent tensor."""

        forward_leaves = _flatten_match(forward)
        reverse_leaves = _flatten_match(reverse)
        if len(forward_leaves) != len(reverse_leaves):
            raise ValueError("forward and reverse matches must share probability structure")
        inputs = (*forward_leaves, *reverse_leaves)
        foreground_index: int | None = None
        if foreground_weight is not None:
            foreground_index = len(inputs)
            inputs = (*inputs, foreground_weight)
        bool_indices = frozenset((4, 4 + len(forward_leaves)))
        device = _validate_cuda_inputs(inputs, bool_indices=bool_indices)
        self._assert_owner(device)
        leaf_count = len(forward_leaves)
        key: _CacheKey = (
            "features",
            radius,
            leaf_count,
            foreground_index is not None,
            *(_tensor_signature(value) for value in inputs),
        )

        def invoke(static: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            static_forward = _unflatten_match(static[:leaf_count])
            static_reverse = _unflatten_match(static[leaf_count : 2 * leaf_count])
            foreground = static[foreground_index] if foreground_index is not None else None
            result = _canonical_features(
                static_forward,
                static_reverse,
                foreground_weight=foreground,
                radius=radius,
            )
            return (result,)

        return self._run(key, inputs, invoke)[0]


@contextmanager
def install_h8_transport_graph(
    *, max_entries: int = 8, warmup_calls: int = 3
) -> Iterator[H8TransportGraph]:
    """Convenience context which installs and then closes one graph cache."""

    runtime = H8TransportGraph(max_entries=max_entries, warmup_calls=warmup_calls)
    try:
        with runtime.install():
            yield runtime
    finally:
        runtime.close()


__all__ = ["H8TransportGraph", "install_h8_transport_graph"]

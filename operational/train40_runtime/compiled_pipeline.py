"""Experimental compiled transport inside otherwise unchanged producer graphs.

A dispatcher operator keeps TorchScript from entering Dynamo. Its implementation
executes the compiled complete-transport callable; CUDA Graph capture then records
those kernels. Encoders, producer tails, guards, stream ownership and output
lifetime remain provided by the existing producer runtime. This adapter is not
admitted for deployment until final TTC parity is measured independently.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import itertools
import sys
import textwrap
import types
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch
from torch import Tensor, nn

from e_jepa_ttc.models import local_transport as transport_oracle
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.train40_runtime.compiled_transport import (
    TransportPairTensorCore,
    attach_transport_pair_core,
)
from operational.train40_runtime.producers import AsyncExpertBlock, ProducerRuntime

OPERATOR_NAME = "train40_compiled_pipeline::transport_pair"
_HANDLES = itertools.count(1)
_NATIVE_LOG_CALLS = 0
_NATIVE_MATCH_CALLS = 0


@torch.library.custom_op("train40_compiled_pipeline::native_log", mutates_args=())
def _native_log(values: Tensor) -> Tensor:
    """Keep exactly the original ATen log arithmetic outside Inductor lowering."""
    global _NATIVE_LOG_CALLS
    if values.dtype != torch.float32:
        raise ValueError("Frozen entropy log expects FP32 inputs")
    _NATIVE_LOG_CALLS += 1
    return torch.log(values)


@_native_log.register_fake
def _native_log_fake(values: Tensor) -> Tensor:
    if values.dtype != torch.float32:
        raise ValueError("Frozen entropy log expects FP32 inputs")
    return torch.empty_like(values)


@torch.library.custom_op("train40_compiled_pipeline::native_match", mutates_args=())
def _native_match(
    previous: Tensor, current: Tensor, radius: int, temperature: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Keep the complete original matching arithmetic outside Inductor lowering."""
    global _NATIVE_MATCH_CALLS
    if previous.dtype != torch.float32 or current.dtype != torch.float32:
        raise ValueError("Frozen native matching expects FP32 inputs")
    _NATIVE_MATCH_CALLS += 1
    match = transport_oracle.local_correlation_match(
        previous, current, radius=radius, temperature=temperature
    )
    return match.dx, match.dy, match.confidence_margin, match.entropy, match.valid


@_native_match.register_fake
def _native_match_fake(
    previous: Tensor, current: Tensor, radius: int, temperature: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if previous.dtype != torch.float32 or current.dtype != torch.float32:
        raise ValueError("Frozen native matching expects FP32 inputs")
    if previous.ndim != 4 or previous.shape != current.shape:
        raise ValueError("transport endpoints must share shape [B,C,H,W]")
    shape = (previous.shape[0], previous.shape[2], previous.shape[3])
    return (
        previous.new_empty(shape),
        previous.new_empty(shape),
        previous.new_empty(shape),
        previous.new_empty(shape),
        previous.new_empty(shape, dtype=torch.bool),
    )


@lru_cache(maxsize=4)
def _rewritten_native_log_match(source: str) -> tuple[Any, dict[str, Any]]:
    """Replace exactly the two audited entropy logs in a private function view."""
    expected = {
        ast.dump(ast.parse(expression, mode="eval").body)
        for expression in ("probability.clamp_min(1.0e-12).log()", "valid_count.log()")
    }

    class Rewrite(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
            if ast.dump(node) in expected:
                self.count += 1
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_train40_native_log", ctx=ast.Load()),
                        args=[cast(ast.Attribute, node.func).value],
                        keywords=[],
                    ),
                    node,
                )
            return self.generic_visit(node)

    rewrite = Rewrite()
    parsed = rewrite.visit(ast.parse(source))
    if rewrite.count != 2:
        raise ValueError("Frozen entropy log statements changed; expected two exact replacements")
    ast.fix_missing_locations(parsed)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    name = f"_train40_native_log_match_{source_hash[:16]}"
    module = types.ModuleType(name)
    module.__dict__.update(transport_oracle.local_correlation_match.__globals__)
    module.__name__, module.__package__, module.__spec__ = name, "", None
    sys.modules[name] = module
    # Dynamo resolves function globals through __module__. A private registered
    # module prevents guards from looking for new names in scientific sources.
    scope = module.__dict__
    scope["_train40_native_log"] = _native_log
    exec(compile(parsed, __file__, "exec"), scope)  # noqa: S102
    return scope["local_correlation_match"], {
        "original_match_sha256": source_hash,
        "rewritten_match_ast_sha256": hashlib.sha256(ast.dump(parsed).encode()).hexdigest(),
        "native_log_replacements": rewrite.count,
    }


class NativeLogTransportPairTensorCore(TransportPairTensorCore):
    """Compile the transport block while preserving its two original ATen logs."""

    def __init__(self, original: TransportPairTensorCore) -> None:
        if original.reuse_reverse:
            raise ValueError("Native-log admission preserves original bidirectional matching")
        super().__init__(
            original.radius,
            original.temperature,
            deterministic_sampling=original.deterministic_sampling,
            reuse_reverse=False,
            reuse_samples=original.reuse_samples,
        )
        source = textwrap.dedent(inspect.getsource(transport_oracle.local_correlation_match))
        self.match, self.native_log_binding = _rewritten_native_log_match(source)

    def forward(
        self,
        previous: Tensor,
        current: Tensor,
        foreground_previous: Tensor,
        foreground_current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Keep all original correlations and summaries with native entropy logs."""
        forward = self.match(previous, current, radius=self.radius, temperature=self.temperature)
        reverse = self.match(current, previous, radius=self.radius, temperature=self.temperature)
        return (
            transport_oracle.transport_physical_features(
                forward,
                reverse,
                foreground_weight=foreground_previous,
                radius=self.radius,
                deterministic_sampling=self.deterministic_sampling,
                reuse_reverse_sample=self.reuse_samples,
            ),
            transport_oracle.transport_physical_features(
                reverse,
                forward,
                foreground_weight=foreground_current,
                radius=self.radius,
                deterministic_sampling=self.deterministic_sampling,
                reuse_reverse_sample=self.reuse_samples,
            ),
        )


class NativeMatchTransportPairTensorCore(TransportPairTensorCore):
    """Compile summaries/cycle while matching uses the untouched ATen oracle."""

    def __init__(self, original: TransportPairTensorCore) -> None:
        if original.reuse_reverse:
            raise ValueError("Native-match admission preserves original bidirectional matching")
        super().__init__(
            original.radius,
            original.temperature,
            deterministic_sampling=original.deterministic_sampling,
            reuse_reverse=False,
            reuse_samples=original.reuse_samples,
        )
        source = textwrap.dedent(inspect.getsource(transport_oracle.local_correlation_match))
        self.native_match_binding = {
            "original_match_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "boundary": "ORIGINAL_ATEN_LOCAL_CORRELATION_MATCH",
            "outputs": ["dx", "dy", "confidence_margin", "entropy", "valid"],
            "compiled_remainder": "ORIGINAL_TRANSPORT_PHYSICAL_FEATURES_AND_CYCLE",
        }

    def forward(
        self,
        previous: Tensor,
        current: Tensor,
        foreground_previous: Tensor,
        foreground_current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reuse no correlations; preserve each original directional match."""
        forward = transport_oracle.LocalTransportMatch(
            *_native_match(previous, current, self.radius, self.temperature)
        )
        reverse = transport_oracle.LocalTransportMatch(
            *_native_match(current, previous, self.radius, self.temperature)
        )
        return (
            transport_oracle.transport_physical_features(
                forward,
                reverse,
                foreground_weight=foreground_previous,
                radius=self.radius,
                deterministic_sampling=self.deterministic_sampling,
                reuse_reverse_sample=self.reuse_samples,
            ),
            transport_oracle.transport_physical_features(
                reverse,
                forward,
                foreground_weight=foreground_current,
                radius=self.radius,
                deterministic_sampling=self.deterministic_sampling,
                reuse_reverse_sample=self.reuse_samples,
            ),
        )


@dataclass(eq=False)
class _Executor:
    callable: Any
    backend: str
    native_log_binding: dict[str, Any] | None = None
    native_match_binding: dict[str, Any] | None = None
    calls: int = 0
    traced_calls: int = 0


# A runtime's model views own their executors. A traced graph alone is deliberately
# insufficient ownership; retaining the runtime (or wrapper) is mandatory.
_EXECUTORS: weakref.WeakValueDictionary[int, _Executor] = weakref.WeakValueDictionary()


@torch.library.custom_op(OPERATOR_NAME, mutates_args=())
def _transport_pair(
    previous: Tensor,
    current: Tensor,
    foreground_previous: Tensor,
    foreground_current: Tensor,
    handle: int,
) -> tuple[Tensor, Tensor]:
    executor = _EXECUTORS.get(handle)
    if executor is None:
        raise RuntimeError("Compiled transport owner expired; retain its runtime with its graph")
    executor.calls += 1
    executor.traced_calls += int(torch.jit.is_tracing())  # pyright: ignore[reportPrivateImportUsage]
    return cast(
        tuple[Tensor, Tensor],
        executor.callable(previous, current, foreground_previous, foreground_current),
    )


class OpaqueCompiledTransport(nn.Module):
    """Own one compiled callable and expose an opaque, traceable dispatcher call."""

    def __init__(
        self,
        core: TransportPairTensorCore,
        *,
        backend: str = "inductor",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not backend:
            raise ValueError("An explicit compiler backend is required")
        self.handle = next(_HANDLES)
        # Keep the compiled module outside registered children: TorchScript must
        # see only the dispatcher op, never recursively inspect OptimizedModule.
        executor = _Executor(
            torch.compile(
                core,
                backend=backend,
                fullgraph=True,
                dynamic=False,
                options=dict(options) if options is not None else None,
            ),
            backend,
            native_log_binding=(
                dict(core.native_log_binding)
                if isinstance(core, NativeLogTransportPairTensorCore)
                else None
            ),
            native_match_binding=(
                dict(core.native_match_binding)
                if isinstance(core, NativeMatchTransportPairTensorCore)
                else None
            ),
        )
        object.__setattr__(self, "_executor", executor)
        _EXECUTORS[self.handle] = executor

    def forward(
        self,
        previous: Tensor,
        current: Tensor,
        foreground_previous: Tensor,
        foreground_current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run compiled transport without entering Dynamo during JIT tracing."""
        return cast(
            tuple[Tensor, Tensor],
            _transport_pair(
                previous, current, foreground_previous, foreground_current, self.handle
            ),
        )

    def diagnostics(self) -> dict[str, Any]:
        """Count dispatcher execution; graph replay intentionally adds no calls."""
        executor = cast(_Executor, self._executor)
        return {
            "operator": OPERATOR_NAME,
            "handle": self.handle,
            "compile_backend": executor.backend,
            "dispatcher_calls": executor.calls,
            "nested_jit_tracing_calls": executor.traced_calls,
            "native_log_binding": executor.native_log_binding,
            "native_log_dispatcher_calls_process": (
                _NATIVE_LOG_CALLS if executor.native_log_binding is not None else None
            ),
            "native_match_binding": executor.native_match_binding,
            "native_match_dispatcher_calls_process": (
                _NATIVE_MATCH_CALLS if executor.native_match_binding is not None else None
            ),
        }


def attach_opaque_compiled_transport(
    model: CausalScaleTTC,
    *,
    backend: str = "inductor",
    options: Mapping[str, Any] | None = None,
    native_log: bool = False,
    native_match: bool = False,
) -> CausalScaleTTC:
    """Share frozen weights and replace only audited transport blocks privately."""
    if native_log and native_match:
        raise ValueError("Select only one native transport boundary: log or match")
    view = attach_transport_pair_core(model, reuse_reverse=False)
    for scale in ("fine", "coarse"):
        name = f"_train40_transport_{scale}"
        core = cast(TransportPairTensorCore, view._modules[name])
        if native_log:
            core = NativeLogTransportPairTensorCore(core).eval()
        elif native_match:
            core = NativeMatchTransportPairTensorCore(core).eval()
        view._modules[name] = OpaqueCompiledTransport(core, backend=backend, options=options)
    return view


class CompiledTransportProducerRuntime(ProducerRuntime):
    """Opt in to compiled transport with original encoders and graph guard API.

    Event endpoint reuse defaults off; RGB reuse defaults on. Keep RGB object
    chunk size one in the calling service to preserve its admitted encoder batch.
    Capture parity is exact against this runtime's compiled eager core, not the
    scientific oracle. Independent original-to-final-TTC admission is required.
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        device: str | torch.device,
        *,
        backend: str = "inductor",
        options: Mapping[str, Any] | None = None,
        reuse_event_encodings: bool = False,
        reuse_rgb_encodings: bool = True,
        native_log: bool = False,
        native_match: bool = False,
    ) -> None:
        if native_log and native_match:
            raise ValueError("Select only one native transport boundary: log or match")
        proposed = {
            name: attach_opaque_compiled_transport(
                model,
                backend=backend,
                options=options,
                native_log=native_log,
                native_match=native_match,
            )
            if isinstance(model, CausalScaleTTC) and model.config.transport_enabled
            else model
            for name, model in models.items()
        }
        super().__init__(
            proposed,
            device,
            graphs=True,
            block_graphs=True,
            reuse_event_encodings=reuse_event_encodings,
            reuse_rgb_encodings=reuse_rgb_encodings,
        )

    def expert_block_async(
        self,
        modality: str,
        sensor: Tensor,
        delta: Tensor,
        *,
        history_group_size: int | None = None,
    ) -> AsyncExpertBlock:
        """Retain compiled executors with each outstanding asynchronous ticket."""
        ticket = super().expert_block_async(
            modality, sensor, delta, history_group_size=history_group_size
        )
        return AsyncExpertBlock(
            ticket.raw,
            ticket.experts,
            ticket.guard,
            ticket.modality,
            (*ticket._lifetime, self),
        )

    def diagnostics(self) -> list[dict[str, object]]:
        """Bind each graph to opaque IR nodes and actual compiled dispatcher calls."""
        rows = super().diagnostics()
        for row, entry in zip(rows, self._entries.values(), strict=True):
            ir = str(entry.traced.inlined_graph)
            nodes = ir.count(OPERATOR_NAME)
            module_nodes = 0
            for module in entry.traced.modules():
                try:
                    module_nodes += str(module.graph).count(OPERATOR_NAME)
                except (AttributeError, RuntimeError):
                    # Unused legacy coarse cores do not have a traced forward.
                    continue
            if max(nodes, module_nodes) < 1:
                raise RuntimeError("Compiled producer graph lost its opaque transport operator")
            row.update(
                transport_mode="OPAQUE_COMPILED_COMPLETE_BLOCK",
                opaque_transport_ir_nodes=nodes,
                opaque_transport_module_ir_nodes=module_nodes,
                initial_parity_reference="COMPILED_CORE_ONLY_REQUIRES_ORIGINAL_FINAL_TTC_GATE",
            )
        return rows

    def transport_diagnostics(self) -> dict[str, list[dict[str, Any]]]:
        """Report compiler callbacks separately from graph replay counters."""
        return {
            name: [
                module.diagnostics()
                for module in model.modules()
                if isinstance(module, OpaqueCompiledTransport)
            ]
            for name, model in self.models.items()
            if isinstance(model, CausalScaleTTC) and model.config.transport_enabled
        }

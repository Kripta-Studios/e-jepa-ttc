"""Experimental compiled transport inside otherwise unchanged producer graphs.

A dispatcher operator keeps TorchScript from entering Dynamo. Its implementation
executes the compiled complete-transport callable; CUDA Graph capture then records
those kernels. Encoders, producer tails, guards, stream ownership and output
lifetime remain provided by the existing producer runtime. This adapter is not
admitted for deployment until final TTC parity is measured independently.
"""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.train40_runtime.compiled_transport import (
    TransportPairTensorCore,
    attach_transport_pair_core,
)
from operational.train40_runtime.producers import ProducerRuntime

OPERATOR_NAME = "train40_compiled_pipeline::transport_pair"
_HANDLES = itertools.count(1)


@dataclass(eq=False)
class _Executor:
    callable: Any
    backend: str
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
        }


def attach_opaque_compiled_transport(
    model: CausalScaleTTC,
    *,
    backend: str = "inductor",
    options: Mapping[str, Any] | None = None,
) -> CausalScaleTTC:
    """Share frozen weights and replace only audited transport blocks privately."""
    view = attach_transport_pair_core(model, reuse_reverse=False)
    for scale in ("fine", "coarse"):
        name = f"_train40_transport_{scale}"
        core = cast(TransportPairTensorCore, view._modules[name])
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
    ) -> None:
        proposed = {
            name: attach_opaque_compiled_transport(model, backend=backend, options=options)
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

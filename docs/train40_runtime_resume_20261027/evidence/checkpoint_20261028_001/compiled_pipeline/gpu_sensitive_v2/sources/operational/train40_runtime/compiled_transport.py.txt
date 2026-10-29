"""Opt-in tensor cores for complete bidirectional frozen transport.

The core includes correlation, both matches, cycle sampling and all global /
foreground summaries. It preserves the original reduction expressions. Inductor
may nevertheless reorder floating-point work, so compilation is experimental
until final TTC parity is measured with the unchanged tolerances.

The model adapter rewrites exactly two audited statement sequences in a private
copy of the original forward function. It changes no source file or parameter,
and fails closed if the original statement structure changes. No input guard is
removed. Compiled cores cannot be nested inside the existing JIT trace route.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import textwrap
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any, cast

import torch
from torch import Tensor, nn

from e_jepa_ttc.models import local_transport as oracle
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.train40_runtime.transport import correlation_pair


class TransportPairTensorCore(nn.Module):
    """Return the two original 18-coordinate transport summaries for one scale."""

    def __init__(
        self,
        radius: int,
        temperature: float,
        *,
        deterministic_sampling: bool,
        reuse_reverse: bool = False,
        reuse_samples: bool = True,
    ) -> None:
        super().__init__()
        if radius < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Transport needs positive radius and finite positive temperature")
        self.radius = radius
        self.temperature = temperature
        self.deterministic_sampling = deterministic_sampling
        self.reuse_reverse = reuse_reverse
        self.reuse_samples = reuse_samples

    def forward(
        self,
        previous: Tensor,
        current: Tensor,
        foreground_previous: Tensor,
        foreground_current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the complete bidirectional block without a cross-query cache."""
        if self.reuse_reverse:
            forward, reverse = correlation_pair(
                previous, current, radius=self.radius, temperature=self.temperature
            )
        else:
            forward = oracle.local_correlation_match(
                previous, current, radius=self.radius, temperature=self.temperature
            )
            reverse = oracle.local_correlation_match(
                current, previous, radius=self.radius, temperature=self.temperature
            )
        return (
            oracle.transport_physical_features(
                forward,
                reverse,
                foreground_weight=foreground_previous,
                radius=self.radius,
                deterministic_sampling=self.deterministic_sampling,
                reuse_reverse_sample=self.reuse_samples,
            ),
            oracle.transport_physical_features(
                reverse,
                forward,
                foreground_weight=foreground_current,
                radius=self.radius,
                deterministic_sampling=self.deterministic_sampling,
                reuse_reverse_sample=self.reuse_samples,
            ),
        )


@dataclass(frozen=True)
class TransportCoreBinding:
    """Auditable identity of the replaced statements and selected execution mode."""

    original_forward_sha256: str
    rewritten_ast_sha256: str
    replacements: int
    reuse_reverse: bool
    compile_backend: str | None


def _expected_block(scale: str) -> list[ast.stmt]:
    if scale == "fine":
        previous, current = "dense_previous", "dense_current"
        forward, reverse = "forward_match", "reverse_match"
        foreground_previous, foreground_current = "foreground_previous", "foreground_current"
    else:
        previous, current = "coarse_previous", "coarse_current"
        forward, reverse = "coarse_match", "coarse_reverse_match"
        foreground_previous = "coarse_foreground_previous"
        foreground_current = "coarse_foreground_current"
    source = f"""
{forward} = local_correlation_match(
    {previous}, {current}, radius={scale}_radius,
    temperature=self.config.transport_temperature,
)
{reverse} = local_correlation_match(
    {current}, {previous}, radius={scale}_radius,
    temperature=self.config.transport_temperature,
)
{scale}_forward = transport_physical_features(
    {forward}, {reverse}, foreground_weight={foreground_previous},
    radius={scale}_radius,
    deterministic_sampling=self.config.deterministic_transport_sampling,
    reuse_reverse_sample=self.runtime_reuse_reverse_sample,
)
{scale}_reverse = transport_physical_features(
    {reverse}, {forward}, foreground_weight={foreground_current},
    radius={scale}_radius,
    deterministic_sampling=self.config.deterministic_transport_sampling,
    reuse_reverse_sample=self.runtime_reuse_reverse_sample,
)
"""
    return ast.parse(source).body


class _TransportBlockRewriter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.replacements = 0

    def _rewrite(self, nodes: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        index = 0
        while index < len(nodes):
            found = False
            for scale in ("fine", "coarse"):
                expected = _expected_block(scale)
                actual = nodes[index : index + len(expected)]
                if [ast.dump(node) for node in actual] != [ast.dump(node) for node in expected]:
                    continue
                if scale == "fine":
                    args = "dense_previous, dense_current, foreground_previous, foreground_current"
                else:
                    args = (
                        "coarse_previous, coarse_current, "
                        "coarse_foreground_previous, coarse_foreground_current"
                    )
                replacement = ast.parse(
                    f"{scale}_forward, {scale}_reverse = self._train40_transport_{scale}({args})"
                ).body[0]
                result.append(ast.copy_location(replacement, nodes[index]))
                index += len(expected)
                self.replacements += 1
                found = True
                break
            if not found:
                result.append(nodes[index])
                index += 1
        return result

    def visit_If(self, node: ast.If) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        node.body = self._rewrite(node.body)
        node.orelse = self._rewrite(node.orelse)
        return node


def _rewritten_forward() -> tuple[Callable[..., Any], str, str, int]:
    original = CausalScaleTTC._forward_impl
    source = textwrap.dedent(inspect.getsource(original))
    parsed = ast.parse(source)
    rewriter = _TransportBlockRewriter()
    transformed = rewriter.visit(parsed)
    if rewriter.replacements != 2:
        raise RuntimeError("Frozen transport statements changed; expected two exact replacements")
    ast.fix_missing_locations(transformed)
    identity = ast.dump(transformed)
    scope = dict(original.__globals__)
    exec(compile(transformed, __file__, "exec"), scope)  # noqa: S102
    return (
        scope[original.__name__],
        hashlib.sha256(source.encode()).hexdigest(),
        hashlib.sha256(identity.encode()).hexdigest(),
        rewriter.replacements,
    )


def attach_transport_pair_core(
    model: CausalScaleTTC,
    *,
    reuse_reverse: bool = False,
    compile_backend: str | None = None,
    compile_options: Mapping[str, Any] | None = None,
) -> CausalScaleTTC:
    """Return a frozen view with complete transport cores and unchanged weights.

    ``compile_backend=None`` remains ordinary eager PyTorch and can be traced by
    the existing producer graph runner. An explicit backend compiles only the
    complete transport blocks; this does not compile the encoder or final TTC.
    The caller must initialize any isolated compiler dependencies beforehand.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Transport cores accept only frozen eval models")
    if not model.config.transport_enabled:
        raise ValueError("The model does not contain local transport")
    if hasattr(model, "_train40_transport_options") or hasattr(model, "_train40_transport_binding"):
        raise ValueError("A transport adapter is already attached")
    implementation = cast(FunctionType, model._forward_impl.__func__)
    if implementation.__code__ is not CausalScaleTTC._forward_impl.__code__:
        raise ValueError("Transport cores require the original forward implementation")
    if compile_options is not None and compile_backend is None:
        raise ValueError("Compilation options require an explicit backend")
    forward, source_hash, ast_hash, count = _rewritten_forward()
    view = copy(model)
    view._modules = model._modules.copy()
    fine_radius = (
        model.config.transport_fine_radius
        if model.config.transport_mode == "adaptive_pyramid"
        else model.config.transport_radius
    )
    for name, radius in (("fine", fine_radius), ("coarse", model.config.transport_coarse_radius)):
        core: nn.Module = TransportPairTensorCore(
            radius,
            model.config.transport_temperature,
            deterministic_sampling=model.config.deterministic_transport_sampling,
            reuse_reverse=reuse_reverse,
        ).eval()
        if compile_backend is not None:
            core = cast(
                nn.Module,
                torch.compile(
                    core,
                    backend=compile_backend,
                    fullgraph=True,
                    dynamic=False,
                    options=dict(compile_options) if compile_options is not None else None,
                ),
            )
        view.add_module(f"_train40_transport_{name}", core)
    object.__setattr__(view, "_forward_impl", MethodType(forward, view))
    object.__setattr__(
        view,
        "_train40_transport_binding",
        TransportCoreBinding(source_hash, ast_hash, count, reuse_reverse, compile_backend),
    )
    return view

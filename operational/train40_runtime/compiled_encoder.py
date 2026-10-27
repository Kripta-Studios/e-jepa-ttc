"""Private FP32 encoder compilation experiments for the frozen TRAIN40 models.

This module never enables TF32, changes the scientific source, or relaxes the
final TTC parity gate. Compiled encoders cannot be nested in the existing
TorchScript trace route. Their integration needs an explicitly traced tail or
another audited graph boundary that retains all original input guards.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, cast

import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

ENCODER_OUTPUT_NAMES = ("foreground", "token", "dense")


def encoder_compile_options(conv_backends: str = "ATEN,TRITON") -> dict[str, Any]:
    """Return explicit FP32-preserving compiler intent, not an accuracy proof."""
    backends = conv_backends.split(",")
    if (
        not backends
        or len(set(backends)) != len(backends)
        or any(value not in {"ATEN", "TRITON"} for value in backends)
    ):
        raise ValueError("conv_backends must select unique ATEN/TRITON backends")
    return {
        "max_autotune": True,
        "max_autotune_conv_backends": conv_backends,
        "use_fast_math": False,
        "force_same_precision": True,
        "triton.cudagraphs": False,
    }


class EncoderTensorCore(nn.Module):
    """Expose foreground, token and dense tensors through a fixed output ABI."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        foreground, token, dense = self.encoder(values, return_dense_features=True)
        if dense is None:
            raise ValueError("Encoder did not return its requested dense feature map")
        return foreground.contiguous(), token.contiguous(), dense.contiguous()


class CompiledEndpointEncoder(nn.Module):
    """Copy a frozen encoder and optionally compile its fixed three-tensor core.

    ``compile_backend=None`` is a CPU-testable reference wrapper, not compilation.
    The production experiment uses only ``inductor``; compilation remains lazy
    until the first input and its cost must be recorded separately from latency.
    Weight/buffer storage never aliases the supplied encoder.
    """

    _compiled_callable: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]]

    def __init__(
        self,
        encoder: nn.Module,
        *,
        compile_backend: str | None = "inductor",
        conv_backends: str = "ATEN,TRITON",
    ) -> None:
        super().__init__()
        if compile_backend not in (None, "inductor"):
            raise ValueError("Only explicit Inductor compilation or a reference wrapper is allowed")
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("Compiled encoder requires a frozen eval source")
        if any(parameter.dtype != torch.float32 for parameter in encoder.parameters()):
            raise ValueError("Compiled encoder preserves FP32 parameter values")
        copied = deepcopy(encoder)
        for key, original in encoder.state_dict().items():
            if not torch.equal(original, copied.state_dict()[key]):
                raise ValueError(f"Encoder copy changed original state: {key}")
        self.core = EncoderTensorCore(copied).eval().requires_grad_(False)
        self.compile_backend = compile_backend
        self.options = encoder_compile_options(conv_backends)
        self.eval()
        compiled = (
            torch.compile(
                self.core,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options=self.options,
            )
            if compile_backend is not None
            else self.core
        )
        # Avoid registering the OptimizedModule as a second path to the same
        # parameters. The owning nn.Module tree remains self.core only.
        object.__setattr__(self, "_compiled_callable", compiled)

    def flat(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the three-output core with explicit precision and tracing guards."""
        if torch.jit.is_tracing():  # pyright: ignore[reportPrivateImportUsage]
            raise RuntimeError("Compiled encoders cannot be nested inside torch.jit.trace")
        if values.dtype != torch.float32 or values.ndim != 4:
            raise ValueError("Encoder input must be an NCHW FP32 tensor")
        if values.device.type == "cuda" and (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ):
            raise ValueError("Encoder experiment requires matmul and cuDNN TF32 disabled")
        return cast(tuple[Tensor, Tensor, Tensor], self._compiled_callable(values))

    def forward(
        self, values: Tensor, *, return_dense_features: bool = False
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Preserve the endpoint encoder's keyword and optional-dense interface."""
        foreground, token, dense = self.flat(values)
        return foreground, token, dense if return_dense_features else None


def clone_producer_models_for_compilation(
    models: Mapping[str, nn.Module], *, conv_backends: str = "ATEN,TRITON"
) -> dict[str, nn.Module]:
    """Return private producer copies with compiled encoders, sharing other heads.

    This helper does not integrate with ProducerRuntime's TorchScript tracing.
    It prepares a separately admitted experimental model view; callers must choose
    a compatible graph boundary and retain the original producer guards.
    """
    result = dict(models)
    for name, model in models.items():
        if not isinstance(model, CausalScaleTTC):
            continue
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Compiled producers require frozen eval source models")
        copied = deepcopy(model)
        copied._modules["encoder"] = CompiledEndpointEncoder(
            copied.encoder, conv_backends=conv_backends
        )
        if copied.transport_encoder is not None:
            copied._modules["transport_encoder"] = CompiledEndpointEncoder(
                copied.transport_encoder, conv_backends=conv_backends
            )
        result[name] = copied
    return result

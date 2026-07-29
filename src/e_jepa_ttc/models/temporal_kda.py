"""Reference PyTorch Kimi Delta Attention for short causal event histories."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class _CausalDepthwiseShortConv(nn.Module):
    """Small causal convolution used by the KDA q/k/v projections."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if dim <= 0 or kernel_size <= 0:
            raise ValueError("dim and kernel_size must be positive.")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            groups=dim,
            bias=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("values must have shape [B,T,P,D].")
        batch, steps, patches, dim = values.shape
        sequence = values.permute(0, 2, 3, 1).reshape(batch * patches, dim, steps)
        sequence = functional.pad(sequence, (self.kernel_size - 1, 0))
        convolved = self.conv(sequence)
        return convolved.reshape(batch, patches, dim, steps).permute(0, 3, 1, 2)


def kimi_delta_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Eq. 1 of Kimi K3 with an explicit causal recurrence.

    Shapes are ``[B,T,P,H,K]`` for query/key/retention,
    ``[B,T,P,H,V]`` for value and ``[B,T,P,H]`` for beta.
    """

    if query.shape != key.shape or query.shape != retention.shape:
        raise ValueError("query, key and retention must have identical shapes.")
    if value.shape[:-1] != query.shape[:-1] or beta.shape != query.shape[:-1]:
        raise ValueError("value and beta axes must match query axes.")
    batch, steps, patches, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    state = torch.zeros(
        batch,
        patches,
        heads,
        key_dim,
        value_dim,
        device=query.device,
        dtype=torch.float32,
    )
    outputs: list[torch.Tensor] = []
    for step in range(steps):
        q_t = query[:, step].float()
        k_t = key[:, step].float()
        v_t = value[:, step].float()
        alpha_t = retention[:, step].float()
        beta_t = beta[:, step].float()
        decayed = alpha_t[..., None] * state
        previous_read = torch.einsum("bphk,bphkv->bphv", k_t, decayed)
        innovation = v_t - previous_read
        state = decayed + beta_t[..., None, None] * torch.einsum(
            "bphk,bphv->bphkv",
            k_t,
            innovation,
        )
        outputs.append(torch.einsum("bphk,bphkv->bphv", q_t, state))
    return torch.stack(outputs, dim=1).to(value.dtype)


class KimiDeltaAttention(nn.Module):
    """Mathematically faithful recurrent KDA without a custom CUDA kernel.

    The short EvTTC history makes the explicit recurrence preferable to adding
    FlashKDA, whose published kernel requirements do not match this host.
    """

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        short_conv_kernel: int = 3,
        decay_rank: int = 16,
        minimum_log_decay: float = -5.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads.")
        if decay_rank <= 0 or minimum_log_decay >= 0:
            raise ValueError("decay_rank must be positive and minimum_log_decay negative.")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.minimum_log_decay = float(minimum_log_decay)
        # Keep the learnable RMS scale explicitly. ``nn.RMSNorm`` currently
        # falls back to a slower path under CUDA autocast when BF16 activations
        # meet FP32 parameters.  The explicit formulation preserves the same
        # operation while casting only the scale to the activation dtype.
        self.input_norm_weight = nn.Parameter(torch.ones(dim))
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.query_conv = _CausalDepthwiseShortConv(dim, short_conv_kernel)
        self.key_conv = _CausalDepthwiseShortConv(dim, short_conv_kernel)
        self.value_conv = _CausalDepthwiseShortConv(dim, short_conv_kernel)
        self.beta_projection = nn.Linear(dim, heads)
        self.decay_down = nn.Linear(dim, decay_rank, bias=False)
        self.decay_up = nn.Linear(decay_rank, dim, bias=False)
        self.decay_bias = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.log_decay_scale = nn.Parameter(torch.zeros(heads, 1))
        self.output_gate = nn.Linear(dim, dim)
        self.head_scale = nn.Parameter(torch.ones(heads, self.head_dim))
        self.output_projection = nn.Linear(dim, dim, bias=False)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.unflatten(-1, (self.heads, self.head_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Causally mix aligned ``[B,T,P,D]`` patch or object sequences."""

        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError("tokens must have shape [B,T,P,dim].")
        normalized = tokens * torch.rsqrt(
            tokens.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(tokens.dtype)
        normalized = normalized * self.input_norm_weight.to(tokens.dtype)
        query = functional.silu(self.query_conv(self.query_projection(normalized)))
        key = functional.silu(self.key_conv(self.key_projection(normalized)))
        value = functional.silu(self.value_conv(self.value_projection(normalized)))
        query = functional.normalize(self._heads(query), dim=-1)
        key = functional.normalize(self._heads(key), dim=-1)
        value_heads = self._heads(value)
        beta = torch.sigmoid(self.beta_projection(normalized))
        decay_logits = self._heads(self.decay_up(self.decay_down(normalized)))
        decay_logits = decay_logits + self.decay_bias
        log_decay = self.minimum_log_decay * torch.sigmoid(
            self.log_decay_scale.exp() * decay_logits
        )
        retention = log_decay.exp()
        recurrent = kimi_delta_recurrence(
            query,
            key,
            value_heads,
            retention,
            beta,
        )
        recurrent = recurrent * torch.rsqrt(
            recurrent.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(recurrent.dtype)
        recurrent = recurrent * self.head_scale.to(recurrent.dtype)
        gate = torch.sigmoid(self._heads(self.output_gate(normalized)))
        output = self.output_projection((gate * recurrent).flatten(-2))
        return tokens + output


class TemporalKDAStack(nn.Module):
    """A small stack of KDA layers for the conditional temporal ablation."""

    def __init__(self, dim: int, *, heads: int = 4, depth: int = 3) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.layers = nn.ModuleList(KimiDeltaAttention(dim, heads=heads) for _ in range(depth))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


# Backwards-compatible name retained for configs and historical manifests.
TemporalDeltaMemory = KimiDeltaAttention

__all__ = [
    "KimiDeltaAttention",
    "TemporalDeltaMemory",
    "TemporalKDAStack",
    "kimi_delta_recurrence",
]

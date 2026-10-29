"""Reference PyTorch Kimi Delta Attention for short causal event histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class KimiDeltaState:
    """Streaming state for one KDA layer.

    ``recurrent`` is deliberately kept in FP32.  The convolution histories use
    the activation dtype because they are only causal input buffers; retaining
    them in FP32 would increase streaming memory without improving the delta
    recurrence's numerical invariant.
    """

    recurrent: torch.Tensor
    query_conv: torch.Tensor
    key_conv: torch.Tensor
    value_conv: torch.Tensor


@dataclass(frozen=True)
class KDALayoutMetadata:
    """Explicit semantic axes required by every KDA invocation."""

    batch_size: int
    temporal_steps: int
    patch_count: int
    embedding_dim: int

    def validate(self, tokens: torch.Tensor) -> None:
        """Reject ambiguous or stale layout metadata."""

        expected = (
            self.batch_size,
            self.temporal_steps,
            self.patch_count,
            self.embedding_dim,
        )
        if min(expected) <= 0:
            raise ValueError("KDA B,T,P,D metadata values must be positive.")
        if tokens.ndim != 4 or tuple(tokens.shape) != expected:
            raise ValueError(
                "KDA tokens must match explicit B,T,P,D metadata: "
                f"expected {expected}, got {tuple(tokens.shape)}."
            )


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

    def forward_with_state(
        self,
        values: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 4:
            raise ValueError("values must have shape [B,T,P,D].")
        batch, steps, patches, dim = values.shape
        if valid_patch_mask is not None and valid_patch_mask.shape != values.shape[:3]:
            raise ValueError("valid_patch_mask must have shape [B,T,P].")
        sequence = values.permute(0, 2, 3, 1).reshape(batch * patches, dim, steps)
        history_size = self.kernel_size - 1
        if state is None:
            history = sequence.new_zeros(batch, patches, dim, history_size)
        else:
            expected = (batch, patches, dim, history_size)
            if tuple(state.shape) != expected:
                raise ValueError(
                    f"Convolution state must have shape {expected}, got {tuple(state.shape)}."
                )
            history = state.to(device=values.device, dtype=values.dtype)
        history_flat = history.reshape(batch * patches, dim, history_size)
        if valid_patch_mask is not None:
            outputs: list[torch.Tensor] = []
            running = history
            for step in range(steps):
                current = values[:, step].unsqueeze(-1)
                candidate_input = torch.cat((running, current), dim=-1)
                convolved = self.conv(candidate_input.reshape(batch * patches, dim, -1))
                convolved = convolved.reshape(batch, patches, dim)
                candidate_history = (
                    candidate_input[..., -history_size:]
                    if history_size
                    else candidate_input[..., :0]
                )
                valid = valid_patch_mask[:, step, :, None]
                outputs.append(convolved.masked_fill(~valid, 0.0))
                running = torch.where(valid.unsqueeze(-1), candidate_history, running)
            return torch.stack(outputs, dim=1), running
        padded = torch.cat((history_flat, sequence), dim=-1)
        convolved = self.conv(padded)
        next_history = padded[..., -history_size:] if history_size else padded[..., :0]
        return (
            convolved.reshape(batch, patches, dim, steps).permute(0, 3, 1, 2),
            next_history.reshape(batch, patches, dim, history_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the causal convolution with a zero initial history."""

        output, _state = self.forward_with_state(values)
        return output


def _validate_recurrence_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if query.shape != key.shape or query.shape != retention.shape:
        raise ValueError("query, key and retention must have identical shapes.")
    if value.shape[:-1] != query.shape[:-1] or beta.shape != query.shape[:-1]:
        raise ValueError("value and beta axes must match query axes.")
    batch, steps, patches, heads, key_dim = query.shape
    return batch, steps, patches, heads, key_dim


def _validate_recurrence_state(
    state: torch.Tensor | None,
    *,
    batch: int,
    patches: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    device: torch.device,
) -> torch.Tensor:
    expected = (batch, patches, heads, key_dim, value_dim)
    if state is None:
        return torch.zeros(expected, device=device, dtype=torch.float32)
    if tuple(state.shape) != expected:
        raise ValueError(f"Recurrence state must have shape {expected}, got {tuple(state.shape)}.")
    return state.to(device=device, dtype=torch.float32)


def kimi_delta_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    return_state: bool = False,
    valid_patch_mask: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Eq. 1 of Kimi K3 with an explicit causal recurrence.

    Shapes are ``[B,T,P,H,K]`` for query/key/retention,
    ``[B,T,P,H,V]`` for value and ``[B,T,P,H]`` for beta.
    """

    batch, steps, patches, heads, key_dim = _validate_recurrence_inputs(
        query, key, value, retention, beta
    )
    value_dim = value.shape[-1]
    if valid_patch_mask is not None and valid_patch_mask.shape != query.shape[:3]:
        raise ValueError("valid_patch_mask must have shape [B,T,P].")
    state = _validate_recurrence_state(
        initial_state,
        batch=batch,
        patches=patches,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        device=query.device,
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
        candidate_state = decayed + beta_t[..., None, None] * torch.einsum(
            "bphk,bphv->bphkv",
            k_t,
            innovation,
        )
        if valid_patch_mask is None:
            state = candidate_state
            valid_output = None
        else:
            valid_state = valid_patch_mask[:, step, :, None, None, None]
            valid_output = valid_patch_mask[:, step, :, None, None]
            state = torch.where(valid_state, candidate_state, state)
        step_output = torch.einsum("bphk,bphkv->bphv", q_t, state)
        if valid_output is not None:
            step_output = step_output.masked_fill(~valid_output, 0.0)
        outputs.append(step_output)
    result = torch.stack(outputs, dim=1).to(value.dtype)
    if return_state:
        return result, state
    return result


def kimi_delta_recurrence_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retention: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the same recurrence in causal chunks for streaming inference.

    The returned state is detached from the graph and stored in FP32.  A
    caller starts a new sequence by passing ``initial_state=None``; no module
    global is used, so timestamps and sequence boundaries cannot leak state.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    _validate_recurrence_inputs(query, key, value, retention, beta)
    outputs: list[torch.Tensor] = []
    state: torch.Tensor | None = initial_state
    for start in range(0, query.shape[1], chunk_size):
        stop = min(start + chunk_size, query.shape[1])
        chunk_result = kimi_delta_recurrence(
            query[:, start:stop],
            key[:, start:stop],
            value[:, start:stop],
            retention[:, start:stop],
            beta[:, start:stop],
            initial_state=state,
            return_state=True,
        )
        if not isinstance(chunk_result, tuple):
            raise RuntimeError("The recurrence did not return its requested state.")
        chunk_output, state = chunk_result
        outputs.append(chunk_output)
    if not outputs:
        raise ValueError("The recurrence input must contain at least one time step.")
    if state is None:
        raise RuntimeError("The recurrence state was not initialized.")
    return torch.cat(outputs, dim=1), state.detach()


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

    def _forward_with_state(
        self,
        tokens: torch.Tensor,
        metadata: KDALayoutMetadata,
        state: KimiDeltaState | None = None,
        *,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KimiDeltaState]:
        """Causally mix aligned ``[B,T,P,D]`` patch or object sequences."""

        metadata.validate(tokens)
        if metadata.embedding_dim != self.dim:
            raise ValueError(f"KDA metadata D must equal module dim {self.dim}.")
        if valid_patch_mask is None:
            valid_patch_mask = torch.ones(tokens.shape[:3], dtype=torch.bool, device=tokens.device)
        elif valid_patch_mask.shape != tokens.shape[:3] or valid_patch_mask.dtype != torch.bool:
            raise ValueError("valid_patch_mask must be bool with shape [B,T,P].")
        tokens = tokens.masked_fill(~valid_patch_mask.unsqueeze(-1), 0.0)
        normalized = tokens * torch.rsqrt(
            tokens.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(tokens.dtype)
        normalized = normalized * self.input_norm_weight.to(tokens.dtype)
        query, query_state = self.query_conv.forward_with_state(
            self.query_projection(normalized),
            state.query_conv if state else None,
            valid_patch_mask=valid_patch_mask,
        )
        key, key_state = self.key_conv.forward_with_state(
            self.key_projection(normalized),
            state.key_conv if state else None,
            valid_patch_mask=valid_patch_mask,
        )
        value, value_state = self.value_conv.forward_with_state(
            self.value_projection(normalized),
            state.value_conv if state else None,
            valid_patch_mask=valid_patch_mask,
        )
        query = functional.silu(query)
        key = functional.silu(key)
        value = functional.silu(value)
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
        recurrent, recurrent_state = kimi_delta_recurrence(
            query,
            key,
            value_heads,
            retention,
            beta,
            initial_state=state.recurrent if state else None,
            return_state=True,
            valid_patch_mask=valid_patch_mask,
        )
        recurrent = recurrent * torch.rsqrt(
            recurrent.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(recurrent.dtype)
        recurrent = recurrent * self.head_scale.to(recurrent.dtype)
        gate = torch.sigmoid(self._heads(self.output_gate(normalized)))
        output = self.output_projection((gate * recurrent).flatten(-2))
        output = (tokens + output).masked_fill(~valid_patch_mask.unsqueeze(-1), 0.0)
        return output, KimiDeltaState(
            recurrent=recurrent_state.detach(),
            query_conv=query_state.detach(),
            key_conv=key_state.detach(),
            value_conv=value_state.detach(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        metadata: KDALayoutMetadata,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causally mix tokens with a fresh zero state."""

        output, _state = self._forward_with_state(
            tokens, metadata, valid_patch_mask=valid_patch_mask
        )
        return output

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        state: KimiDeltaState | None = None,
        *,
        metadata: KDALayoutMetadata,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KimiDeltaState]:
        """Process a causal chunk and return explicit state for the next chunk."""

        return self._forward_with_state(tokens, metadata, state, valid_patch_mask=valid_patch_mask)


class TemporalDeltaMemory(KimiDeltaAttention):
    """Legacy wrapper that infers layout metadata for the pre-v4 API.

    The canonical KDA and all high-resolution paths still require explicit
    :class:`KDALayoutMetadata`.  Older OGE callers exposed ``TemporalDeltaMemory``
    as a plain ``[B,T,P,D]`` module, so this compatibility name derives only the
    unambiguous shape and never flattens or reorders the patch axis.
    """

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        metadata: KDALayoutMetadata | None = None,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if metadata is None:
            if tokens.ndim != 4:
                raise ValueError("TemporalDeltaMemory tokens must have shape [B,T,P,D].")
            metadata = KDALayoutMetadata(*tokens.shape)
        return super().forward(
            tokens,
            metadata=metadata,
            valid_patch_mask=valid_patch_mask,
        )

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        state: KimiDeltaState | None = None,
        *,
        metadata: KDALayoutMetadata | None = None,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KimiDeltaState]:
        if metadata is None:
            if tokens.ndim != 4:
                raise ValueError("TemporalDeltaMemory tokens must have shape [B,T,P,D].")
            metadata = KDALayoutMetadata(*tokens.shape)
        return super().forward_chunk(
            tokens,
            state,
            metadata=metadata,
            valid_patch_mask=valid_patch_mask,
        )


class TemporalKDAStack(nn.Module):
    """A small stack of KDA layers for the conditional temporal ablation."""

    def __init__(self, dim: int, *, heads: int = 4, depth: int = 3) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.layers = nn.ModuleList(KimiDeltaAttention(dim, heads=heads) for _ in range(depth))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        metadata: KDALayoutMetadata,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            layer_module = cast(KimiDeltaAttention, layer)
            tokens = layer_module(tokens, metadata=metadata, valid_patch_mask=valid_patch_mask)
        return tokens

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        state: tuple[KimiDeltaState, ...] | None = None,
        *,
        metadata: KDALayoutMetadata,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[KimiDeltaState, ...]]:
        """Process a chunk through every layer with explicit resettable state."""

        if state is not None and len(state) != len(self.layers):
            raise ValueError("TemporalKDAStack state must contain one state per layer.")
        next_states: list[KimiDeltaState] = []
        value = tokens
        for index, layer in enumerate(self.layers):
            layer_state = state[index] if state is not None else None
            layer_module = cast(KimiDeltaAttention, layer)
            value, layer_next_state = layer_module.forward_chunk(
                value,
                layer_state,
                metadata=metadata,
                valid_patch_mask=valid_patch_mask,
            )
            next_states.append(layer_next_state)
        return value, tuple(next_states)


__all__ = [
    "KDALayoutMetadata",
    "KimiDeltaAttention",
    "KimiDeltaState",
    "TemporalDeltaMemory",
    "TemporalKDAStack",
    "kimi_delta_recurrence",
    "kimi_delta_recurrence_chunked",
]

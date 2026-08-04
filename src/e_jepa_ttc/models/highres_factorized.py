"""Factorized high-resolution tubelet encoder with explicit padding contracts.

The module keeps spatial patches in their spatial axes until window attention has
finished. Temporal mixers receive ``[B, T, P, D]`` and are therefore unable to
interpret raster order as a causal axis. High-resolution inputs are padded to a
patch grid; they are never cropped or upsampled from the 160x90 cache.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.temporal_kda import KDALayoutMetadata, TemporalKDAStack


class TheoreticalOOMError(RuntimeError):
    """Raised before a forbidden global-attention tensor could be allocated."""


@dataclass(frozen=True)
class PatchGeometry:
    """Geometry of an explicitly padded spatial patch grid."""

    source_height: int
    source_width: int
    patch_size: int
    grid_height: int
    grid_width: int
    padded_height: int
    padded_width: int

    @property
    def patch_count(self) -> int:
        """Return the number of spatial patches, including masked border patches."""

        return self.grid_height * self.grid_width

    @property
    def padded_pixels(self) -> int:
        """Return the number of pixels added by right/bottom padding."""

        return self.padded_height * self.padded_width - self.source_height * self.source_width


@dataclass
class HighResFeatures:
    """Dense features and validity metadata emitted by the encoder."""

    tokens: torch.Tensor
    valid_patch_mask: torch.Tensor
    geometry: PatchGeometry
    diagnostics: dict[str, torch.Tensor]
    encoded_grid_height: int = 0
    encoded_grid_width: int = 0
    post_merge_patch_coordinates: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 2), dtype=torch.float32)
    )

    @property
    def patch_coordinates(self) -> torch.Tensor:
        """Return normalized ``[x, y]`` coordinates aligned with the emitted patch axis."""

        return self.post_merge_patch_coordinates


@dataclass
class EJEPATubeletLHROutput:
    """Prediction and dense-token output for the high-resolution model."""

    ttc_mean_seconds: torch.Tensor
    collision_logits: torch.Tensor
    embedding: torch.Tensor
    tokens: torch.Tensor
    valid_patch_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def make_patch_geometry(height: int, width: int, patch_size: int) -> PatchGeometry:
    """Compute ceil-based patch geometry without dropping image borders."""

    if min(height, width, patch_size) <= 0:
        raise ValueError("height, width and patch_size must be positive.")
    grid_height = math.ceil(height / patch_size)
    grid_width = math.ceil(width / patch_size)
    return PatchGeometry(
        source_height=height,
        source_width=width,
        patch_size=patch_size,
        grid_height=grid_height,
        grid_width=grid_width,
        padded_height=grid_height * patch_size,
        padded_width=grid_width * patch_size,
    )


def pad_to_patch_grid(
    inputs: torch.Tensor,
    patch_size: int,
    *,
    valid_temporal_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, PatchGeometry]:
    """Pad ``[B,T,C,H,W]`` on the right/bottom and return a patch mask."""

    if inputs.ndim != 5:
        raise ValueError("inputs must have shape [B,T,C,H,W].")
    if valid_temporal_mask is None:
        valid_temporal_mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
    elif valid_temporal_mask.shape != inputs.shape[:2] or valid_temporal_mask.dtype != torch.bool:
        raise ValueError("valid_temporal_mask must be bool with shape [B,T].")
    inputs = inputs.masked_fill(~valid_temporal_mask[:, :, None, None, None], 0.0)
    geometry = make_patch_geometry(inputs.shape[-2], inputs.shape[-1], patch_size)
    padding_right = geometry.padded_width - geometry.source_width
    padding_bottom = geometry.padded_height - geometry.source_height
    padded = functional.pad(inputs, (0, padding_right, 0, padding_bottom))
    valid_pixels = torch.ones(
        inputs.shape[0],
        inputs.shape[1],
        geometry.source_height,
        geometry.source_width,
        dtype=torch.bool,
        device=inputs.device,
    )
    valid_pixels = functional.pad(valid_pixels, (0, padding_right, 0, padding_bottom), value=False)
    valid_patches = valid_pixels.reshape(
        inputs.shape[0],
        inputs.shape[1],
        geometry.grid_height,
        patch_size,
        geometry.grid_width,
        patch_size,
    ).any(dim=(3, 5))
    valid_patches &= valid_temporal_mask[:, :, None, None]
    return padded, valid_patches, geometry


def theoretical_attention_bytes(
    batch: int,
    steps: int,
    patches: int,
    heads: int,
    *,
    bytes_per_value: int = 2,
) -> int:
    """Estimate the dominant global-attention score tensor size."""

    tokens = steps * patches
    return batch * heads * tokens * tokens * bytes_per_value


def theoretical_oom_guard(
    *,
    batch: int,
    steps: int,
    patches: int,
    heads: int,
    memory_budget_gb: float,
    global_attention: bool,
) -> None:
    """Reject unsafe global attention before any score tensor is allocated."""

    if not theoretical_oom_guard_required(
        batch=batch,
        steps=steps,
        patches=patches,
        heads=heads,
        memory_budget_gb=memory_budget_gb,
        global_attention=global_attention,
    ):
        return
    tokens = steps * patches
    estimate = theoretical_attention_bytes(batch, steps, patches, heads)
    budget = int(memory_budget_gb * (1024**3))
    raise TheoreticalOOMError(
        "Global attention is forbidden for this high-resolution layout: "
        f"{tokens} tokens, estimated score storage {estimate} bytes, "
        f"budget {budget} bytes. Use windowed spatial attention and a temporal-only mixer."
    )


def theoretical_oom_guard_required(
    *,
    batch: int,
    steps: int,
    patches: int,
    heads: int,
    memory_budget_gb: float,
    global_attention: bool,
) -> bool:
    """Return whether global-attention preflight must reject the layout."""

    if not global_attention:
        return False
    tokens = steps * patches
    estimate = theoretical_attention_bytes(batch, steps, patches, heads)
    budget = int(memory_budget_gb * (1024**3))
    return tokens >= 4800 or estimate > budget // 2


class WindowSpatialAttention(nn.Module):
    """Bidirectional attention inside fixed or shifted spatial windows."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError("shift_size must lie in [0, window_size).")
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.projection = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Mix spatial positions without mixing time or padded patches."""

        if tokens.ndim != 5 or valid_mask.shape != tokens.shape[:4]:
            raise ValueError("tokens must be [B,T,H,W,D] and mask must be [B,T,H,W].")
        batch, steps, height, width, dim = tokens.shape
        window = self.window_size
        offset = self.shift_size
        padded_height = math.ceil((height + offset) / window) * window
        padded_width = math.ceil((width + offset) / window) * window
        padded = tokens.new_zeros(batch, steps, padded_height, padded_width, dim)
        padded[:, :, offset : offset + height, offset : offset + width] = tokens
        padded_mask = torch.zeros(
            batch, steps, padded_height, padded_width, dtype=torch.bool, device=tokens.device
        )
        padded_mask[:, :, offset : offset + height, offset : offset + width] = valid_mask
        windows_h = padded_height // window
        windows_w = padded_width // window
        grouped = padded.reshape(
            batch,
            steps,
            windows_h,
            window,
            windows_w,
            window,
            dim,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        grouped_mask = padded_mask.reshape(
            batch,
            steps,
            windows_h,
            window,
            windows_w,
            window,
        ).permute(0, 1, 2, 4, 3, 5)
        flat = grouped.reshape(-1, window * window, dim)
        flat_mask = grouped_mask.reshape(-1, window * window)
        normalized = self.norm(flat)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        attended = flat + self.projection(attended)
        attended = attended.masked_fill(~flat_mask.unsqueeze(-1), 0.0)
        restored = attended.reshape(
            batch,
            steps,
            windows_h,
            windows_w,
            window,
            window,
            dim,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        restored = restored.reshape(batch, steps, padded_height, padded_width, dim)
        return restored[:, :, offset : offset + height, offset : offset + width]


def space_to_depth_2x2(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge 2x2 spatial patches while retaining a validity mask."""

    if tokens.ndim != 5 or valid_mask.shape != tokens.shape[:4]:
        raise ValueError("tokens must be [B,T,H,W,D] and mask must be [B,T,H,W].")
    batch, steps, height, width, dim = tokens.shape
    padded_height = height + height % 2
    padded_width = width + width % 2
    padded = tokens.new_zeros(batch, steps, padded_height, padded_width, dim)
    # Invalid children must not leak their values through the 4D projection.
    # The mask is an explicit contract, but zeroing here is also required when
    # a caller reuses a tensor whose invalid entries still contain activations.
    padded[:, :, :height, :width] = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    padded_mask = torch.zeros(
        batch, steps, padded_height, padded_width, dtype=torch.bool, device=tokens.device
    )
    padded_mask[:, :, :height, :width] = valid_mask
    merged = padded.reshape(
        batch,
        steps,
        padded_height // 2,
        2,
        padded_width // 2,
        2,
        dim,
    ).permute(0, 1, 2, 4, 3, 5, 6)
    merged_mask = padded_mask.reshape(
        batch,
        steps,
        padded_height // 2,
        2,
        padded_width // 2,
        2,
    ).permute(0, 1, 2, 4, 3, 5)
    return (
        merged.reshape(batch, steps, padded_height // 2, padded_width // 2, dim * 4),
        merged_mask.any(dim=(4, 5)),
    )


def normalized_patch_coordinates(
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return raster-ordered post-merge patch centres in normalized ``[x, y]`` space.

    The coordinates deliberately use the final encoded grid rather than the source
    ``PatchGeometry`` grid.  This makes the output stable for odd source grids where
    a 2x2 space-to-depth merge creates a partially populated final row or column.
    """

    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("grid_height and grid_width must be positive.")
    rows = (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) / grid_height
    columns = (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) / grid_width
    y, x = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(grid_height * grid_width, 2)


class TemporalOnlyAttention(nn.Module):
    """Causal attention independently per spatial patch."""

    def __init__(self, dim: int, *, heads: int = 4, depth: int = 2) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 3,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply a temporal causal mask without mixing different patches."""

        if tokens.ndim != 4 or valid_mask.shape != tokens.shape[:3]:
            raise ValueError("tokens must be [B,T,P,D] and mask must be [B,T,P].")
        batch, steps, patches, dim = tokens.shape
        flat = tokens.permute(0, 2, 1, 3).reshape(batch * patches, steps, dim)
        flat_valid = valid_mask.permute(0, 2, 1).reshape(batch * patches, steps)
        active = flat_valid.any(dim=1)
        output = torch.zeros_like(flat)
        if bool(active.any()):
            causal = torch.triu(
                torch.ones(steps, steps, dtype=torch.bool, device=tokens.device), diagonal=1
            )
            value = flat[active]
            value_mask = flat_valid[active]
            for layer in self.layers:
                value = layer(value, src_mask=causal, src_key_padding_mask=~value_mask)
            output[active] = value
        return output.reshape(batch, patches, steps, dim).permute(0, 2, 1, 3)


@dataclass(frozen=True)
class EJEPATubeletLHRConfig:
    """Configuration shared by S3, S4 and S5."""

    in_channels: int = 21
    embed_dim: int = 192
    patch_size: int = 16
    spatial_window: int = 8
    heads: int = 6
    spatial_depth: int = 1
    temporal_depth: int = 2
    temporal_mixer: str = "block_causal"
    merge_2x2: bool = False
    global_attention: bool = False
    memory_budget_gb: float = 12.0
    pooling: str = "query"
    query_count: int = 8


# This is intentionally narrower than ``state_dict()``.  Query pooling and task
# readouts are downstream-only state and must never be mistaken for pretraining
# transfer state.
BACKBONE_STATE_PREFIXES: tuple[str, ...] = (
    "patch_embed.",
    "spatial.",
    "merge.",
    "temporal.",
    "final_norm.",
)
BACKBONE_STRUCTURAL_CONFIG_FIELDS: tuple[str, ...] = (
    "in_channels",
    "embed_dim",
    "patch_size",
    "spatial_window",
    "heads",
    "spatial_depth",
    "temporal_depth",
    "temporal_mixer",
    "merge_2x2",
)


def backbone_structural_config(config: EJEPATubeletLHRConfig) -> dict[str, Any]:
    """Return the exact architecture fields that govern transferable backbone state."""

    source = asdict(config)
    return {key: source[key] for key in BACKBONE_STRUCTURAL_CONFIG_FIELDS}


def _backbone_key_allowed(key: str) -> bool:
    return key.startswith(BACKBONE_STATE_PREFIXES)


def exact_backbone_state_dict(model: EJEPATubeletLHR) -> dict[str, torch.Tensor]:
    """Extract only the state permitted to transfer into the downstream backbone."""

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if _backbone_key_allowed(key)
    }


def _normalize_backbone_config(
    config: EJEPATubeletLHRConfig | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(config, EJEPATubeletLHRConfig):
        return backbone_structural_config(config)
    received_keys = {str(key) for key in config}
    expected_keys = set(BACKBONE_STRUCTURAL_CONFIG_FIELDS)
    missing = sorted(expected_keys - received_keys)
    extra = sorted(received_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"Backbone structural config must match exactly; missing={missing}, extra={extra}."
        )
    return {key: config[key] for key in BACKBONE_STRUCTURAL_CONFIG_FIELDS}


def validate_exact_backbone_transfer(
    model: EJEPATubeletLHR,
    source_state: Mapping[str, Any],
    source_config: EJEPATubeletLHRConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a checkpoint exactly matches the downstream backbone.

    No task state is filtered opportunistically: the supplied state dictionary must
    contain *only* the exact backbone keys with their expected tensor shapes.  The
    returned report is suitable for checkpoint provenance.
    """

    expected_config = backbone_structural_config(model.config)
    received_config = _normalize_backbone_config(source_config)
    config_mismatches = {
        key: {"expected": expected_config[key], "received": received_config[key]}
        for key in BACKBONE_STRUCTURAL_CONFIG_FIELDS
        if expected_config[key] != received_config[key]
    }
    if config_mismatches:
        raise ValueError(
            "Backbone structural config mismatch: "
            + ", ".join(
                f"{key}={value['received']!r} (expected {value['expected']!r})"
                for key, value in sorted(config_mismatches.items())
            )
        )

    if any(not isinstance(key, str) for key in source_state):
        raise ValueError("Exact backbone transfer rejected: state-dict keys must be strings.")
    expected_state = exact_backbone_state_dict(model)
    received_keys = set(source_state)
    expected_keys = set(expected_state)
    extra = sorted(received_keys - expected_keys)
    missing = sorted(expected_keys - received_keys)
    forbidden = sorted(key for key in received_keys if not _backbone_key_allowed(key))
    shape_mismatches: list[str] = []
    non_tensors: list[str] = []
    for key in sorted(expected_keys & received_keys):
        value = source_state[key]
        if not isinstance(value, torch.Tensor):
            non_tensors.append(key)
        elif value.shape != expected_state[key].shape:
            shape_mismatches.append(
                f"{key}: checkpoint {tuple(value.shape)} != "
                f"model {tuple(expected_state[key].shape)}"
            )
    if extra or missing or forbidden or shape_mismatches or non_tensors:
        details: list[str] = []
        if missing:
            details.append("missing=" + repr(missing))
        if extra:
            details.append("extra=" + repr(extra))
        if forbidden:
            details.append("forbidden_non_backbone=" + repr(forbidden))
        if shape_mismatches:
            details.append("shape_mismatches=" + repr(shape_mismatches))
        if non_tensors:
            details.append("non_tensor=" + repr(non_tensors))
        raise ValueError("Exact backbone transfer rejected: " + "; ".join(details))
    return {
        "structural_config": expected_config,
        "transferred_keys": sorted(expected_keys),
        "key_count": len(expected_keys),
    }


class EJEPATubeletLHR(nn.Module):
    """Dense causal tubelet model with factorized spatial/temporal mixing."""

    def __init__(self, config: EJEPATubeletLHRConfig | None = None) -> None:
        super().__init__()
        self.config = config or EJEPATubeletLHRConfig()
        if self.config.temporal_mixer not in {"block_causal", "kda"}:
            raise ValueError("temporal_mixer must be 'block_causal' or 'kda'.")
        if self.config.embed_dim % self.config.heads:
            raise ValueError("embed_dim must be divisible by heads.")
        if self.config.pooling not in {"query", "mean"}:
            raise ValueError("pooling must be 'query' or 'mean'.")
        if self.config.query_count <= 0:
            raise ValueError("query_count must be positive.")
        self.patch_embed = nn.Conv2d(
            self.config.in_channels,
            self.config.embed_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            bias=False,
        )
        self.spatial = nn.ModuleList(
            WindowSpatialAttention(
                self.config.embed_dim,
                heads=self.config.heads,
                window_size=self.config.spatial_window,
                shift_size=(
                    self.config.spatial_window // 2
                    if index % 2 and self.config.spatial_window > 1
                    else 0
                ),
            )
            for index in range(self.config.spatial_depth)
        )
        self.merge = (
            nn.Linear(self.config.embed_dim * 4, self.config.embed_dim)
            if self.config.merge_2x2
            else None
        )
        if self.config.temporal_mixer == "block_causal":
            self.temporal: nn.Module = TemporalOnlyAttention(
                self.config.embed_dim,
                heads=self.config.heads,
                depth=self.config.temporal_depth,
            )
        else:
            self.temporal = TemporalKDAStack(
                self.config.embed_dim,
                heads=self.config.heads,
                depth=self.config.temporal_depth,
            )
        self.final_norm = nn.LayerNorm(self.config.embed_dim)
        self.ttc_head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.embed_dim // 2, 1),
        )
        self.collision_head = nn.Linear(self.config.embed_dim, 4)
        if self.config.pooling == "query":
            self.query_tokens: nn.Parameter | None = nn.Parameter(
                torch.empty(self.config.query_count, self.config.embed_dim)
            )
            nn.init.normal_(self.query_tokens, mean=0.0, std=self.config.embed_dim**-0.5)
            self.query_attention: nn.MultiheadAttention | None = nn.MultiheadAttention(
                self.config.embed_dim,
                self.config.heads,
                batch_first=True,
            )
        else:
            self.register_parameter("query_tokens", None)
            self.query_attention = None

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the exact state allowed to transfer from SSL into this backbone."""

        return exact_backbone_state_dict(self)

    def backbone_structural_config(self) -> dict[str, Any]:
        """Return the architecture identity required for exact SSL transfer."""

        return backbone_structural_config(self.config)

    def load_exact_backbone_state_dict(
        self,
        source_state: Mapping[str, Any],
        source_config: EJEPATubeletLHRConfig | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load a fully validated backbone-only state without touching task modules."""

        report = validate_exact_backbone_transfer(self, source_state, source_config)
        result = self.load_state_dict(
            {key: source_state[key] for key in report["transferred_keys"]}, strict=False
        )
        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected exact-backbone keys after validation: {result.unexpected_keys}"
            )
        return {
            **report,
            "missing_non_backbone_keys": sorted(result.missing_keys),
        }

    def _patch_tokens(
        self,
        inputs: torch.Tensor,
        valid_temporal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, PatchGeometry]:
        padded, valid, geometry = pad_to_patch_grid(
            inputs,
            self.config.patch_size,
            valid_temporal_mask=valid_temporal_mask,
        )
        batch, steps, channels, height, width = padded.shape
        projected = self.patch_embed(padded.reshape(batch * steps, channels, height, width))
        tokens = projected.permute(0, 2, 3, 1).reshape(
            batch, steps, geometry.grid_height, geometry.grid_width, self.config.embed_dim
        )
        for layer in self.spatial:
            tokens = layer(tokens, valid)
        if self.merge is not None:
            tokens, valid = space_to_depth_2x2(tokens, valid)
            tokens = self.merge(tokens)
        return tokens, valid, geometry

    def forward_features(
        self,
        inputs: torch.Tensor,
        *,
        valid_temporal_mask: torch.Tensor | None = None,
    ) -> HighResFeatures:
        """Return dense features while preserving temporal and patch axes."""

        if inputs.ndim != 5 or inputs.shape[2] != self.config.in_channels:
            raise ValueError(f"inputs must have shape [B,T,{self.config.in_channels},H,W].")
        batch, steps = inputs.shape[:2]
        # Check the potentially forbidden global layout before patch embedding
        # or any attention-sized tensor is materialized.  The post-merge count
        # is used because merge is the only permitted reduction before a
        # hypothetical global refresh.
        initial_geometry = make_patch_geometry(
            inputs.shape[-2], inputs.shape[-1], self.config.patch_size
        )
        initial_patches = initial_geometry.patch_count
        if self.config.merge_2x2:
            initial_patches = math.ceil(initial_geometry.grid_height / 2) * math.ceil(
                initial_geometry.grid_width / 2
            )
        guard_required = theoretical_oom_guard_required(
            batch=batch,
            steps=steps,
            patches=initial_patches,
            heads=self.config.heads,
            memory_budget_gb=self.config.memory_budget_gb,
            global_attention=self.config.global_attention,
        )
        theoretical_oom_guard(
            batch=batch,
            steps=steps,
            patches=initial_patches,
            heads=self.config.heads,
            memory_budget_gb=self.config.memory_budget_gb,
            global_attention=self.config.global_attention,
        )
        if self.config.global_attention:
            raise TheoreticalOOMError(
                "Global high-resolution attention is intentionally not implemented; "
                "use windowed spatial attention and a temporal-only mixer."
            )
        tokens, valid, geometry = self._patch_tokens(inputs, valid_temporal_mask)
        patches = tokens.shape[2] * tokens.shape[3]
        flat = tokens.reshape(batch, steps, patches, self.config.embed_dim)
        flat_valid = valid.reshape(batch, steps, patches)
        if self.config.temporal_mixer == "kda":
            flat = self.temporal(
                flat,
                metadata=KDALayoutMetadata(
                    batch_size=batch,
                    temporal_steps=steps,
                    patch_count=patches,
                    embedding_dim=self.config.embed_dim,
                ),
                valid_patch_mask=flat_valid,
            )
        else:
            flat = self.temporal(flat, flat_valid)
        flat = self.final_norm(flat)
        flat = flat.masked_fill(~flat_valid.unsqueeze(-1), 0.0)
        diagnostics = {
            "tokens_before_merge": torch.tensor(
                geometry.patch_count * steps, device=inputs.device, dtype=torch.int64
            ),
            "tokens_after_merge": torch.tensor(
                flat.shape[1] * flat.shape[2], device=inputs.device, dtype=torch.int64
            ),
            "valid_patch_count": flat_valid.sum().to(dtype=torch.int64),
            "padded_patch_count": (~flat_valid).sum().to(dtype=torch.int64),
            "spatial_attention_pairs": torch.tensor(
                math.ceil(geometry.grid_height / self.config.spatial_window)
                * math.ceil(geometry.grid_width / self.config.spatial_window)
                * self.config.spatial_window**4
                * steps,
                device=inputs.device,
                dtype=torch.int64,
            ),
            "temporal_attention_pairs": torch.tensor(
                flat.shape[2] * steps * steps,
                device=inputs.device,
                dtype=torch.int64,
            ),
            "theoretical_oom_guard_required": torch.tensor(
                guard_required, device=inputs.device, dtype=torch.bool
            ),
            "theoretical_oom_guard_triggered": torch.tensor(
                False, device=inputs.device, dtype=torch.bool
            ),
        }
        encoded_grid_height = tokens.shape[2]
        encoded_grid_width = tokens.shape[3]
        coordinates = normalized_patch_coordinates(
            encoded_grid_height,
            encoded_grid_width,
            device=flat.device,
            dtype=flat.dtype,
        )
        diagnostics["encoded_grid_height"] = torch.tensor(
            encoded_grid_height, device=inputs.device, dtype=torch.int64
        )
        diagnostics["encoded_grid_width"] = torch.tensor(
            encoded_grid_width, device=inputs.device, dtype=torch.int64
        )
        return HighResFeatures(
            tokens=flat,
            valid_patch_mask=flat_valid,
            geometry=geometry,
            diagnostics=diagnostics,
            encoded_grid_height=encoded_grid_height,
            encoded_grid_width=encoded_grid_width,
            post_merge_patch_coordinates=coordinates,
        )

    def _pool_tokens(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool dense tokens with fixed learned queries or an explicit control."""

        weights = valid_mask.to(tokens.dtype).unsqueeze(-1)
        if self.config.pooling == "mean":
            return (tokens * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)

        batch, steps, patches, dim = tokens.shape
        flat_tokens = tokens.reshape(batch, steps * patches, dim)
        flat_valid = valid_mask.reshape(batch, steps * patches)
        pooled = tokens.new_zeros(batch, dim)
        active = flat_valid.any(dim=1)
        if bool(active.any()):
            if self.query_tokens is None or self.query_attention is None:
                raise RuntimeError("Query pooling modules are not initialized.")
            queries = self.query_tokens.unsqueeze(0).expand(int(active.sum()), -1, -1)
            attended, _ = self.query_attention(
                queries,
                flat_tokens[active],
                flat_tokens[active],
                key_padding_mask=~flat_valid[active],
                need_weights=False,
            )
            pooled[active] = attended.mean(dim=1).to(pooled.dtype)
        return pooled

    def pool_features(self, features: HighResFeatures) -> torch.Tensor:
        """Pool all valid temporal/spatial tokens into one downstream embedding."""

        return self._pool_tokens(features.tokens, features.valid_patch_mask)

    def pool_temporal_steps(self, features: HighResFeatures) -> torch.Tensor:
        """Pool every temporal endpoint independently with the configured readout.

        Object-centric LHR must preserve the distinction between the two
        observations. Pooling the complete sequence before predicting heights
        would make the two height outputs exchangeable and reintroduce the
        constant-ratio attractor.
        """

        if features.tokens.ndim != 4 or features.valid_patch_mask.shape != features.tokens.shape[:3]:
            raise ValueError("HighResFeatures must contain [B,T,P,D] tokens and [B,T,P] mask.")
        pooled = [
            self._pool_tokens(
                features.tokens[:, step : step + 1],
                features.valid_patch_mask[:, step : step + 1],
            )
            for step in range(features.tokens.shape[1])
        ]
        return torch.stack(pooled, dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        valid_temporal_mask: torch.Tensor | None = None,
    ) -> EJEPATubeletLHROutput:
        """Predict TTC and collision thresholds from dense causal tokens."""

        features = self.forward_features(inputs, valid_temporal_mask=valid_temporal_mask)
        pooled = self.pool_features(features)
        # eAP/Garl-TTC is a signed protocol: receding objects have negative TTC.
        # Exponentiating this head made the high-resolution candidate incapable
        # of representing an entire official evaluation bucket.
        signed_ttc = self.ttc_head(pooled).squeeze(-1)
        return EJEPATubeletLHROutput(
            ttc_mean_seconds=signed_ttc.clamp(-60.0, 60.0),
            collision_logits=self.collision_head(pooled),
            embedding=pooled,
            tokens=features.tokens,
            valid_patch_mask=features.valid_patch_mask,
            diagnostics=features.diagnostics,
        )


__all__ = [
    "BACKBONE_STATE_PREFIXES",
    "BACKBONE_STRUCTURAL_CONFIG_FIELDS",
    "EJEPATubeletLHR",
    "EJEPATubeletLHRConfig",
    "EJEPATubeletLHROutput",
    "HighResFeatures",
    "PatchGeometry",
    "TheoreticalOOMError",
    "WindowSpatialAttention",
    "backbone_structural_config",
    "exact_backbone_state_dict",
    "make_patch_geometry",
    "normalized_patch_coordinates",
    "pad_to_patch_grid",
    "space_to_depth_2x2",
    "theoretical_attention_bytes",
    "theoretical_oom_guard",
    "theoretical_oom_guard_required",
    "validate_exact_backbone_transfer",
]

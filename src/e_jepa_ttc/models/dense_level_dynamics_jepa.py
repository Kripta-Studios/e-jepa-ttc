"""Dense, label-free Level--Dynamics JEPA built on the exact Tubelet LHR backbone.

The module deliberately keeps the downstream encoder's dense ``[B, T, P, D]``
contract.  Its target representation is an EMA-only deep copy, while the predictor
mixes time independently for each post-merge spatial patch.  Nothing in this module
accepts TTC, boxes, categories, RGB, or future target tokens as predictor input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    HighResFeatures,
    PatchGeometry,
)


@dataclass(frozen=True)
class MemoryProfileException:
    """A versioned measured exception to one or more resident resource limits.

    The exception is intentionally fail-closed.  It must name the exact request
    hash, list every exceeded limit, and report a measured peak below the 10.5 GiB
    allocation ceiling.  It is not a general opt-out switch.
    """

    version: str
    resource_request_sha256: str
    measured_peak_allocated_vram_gb: float
    permitted_limit_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("memory-profile version must be non-empty.")
        if len(self.resource_request_sha256) != 64:
            raise ValueError("memory-profile resource_request_sha256 must be a SHA-256 hex string.")
        if not all(character in "0123456789abcdef" for character in self.resource_request_sha256):
            raise ValueError(
                "memory-profile resource_request_sha256 must use lowercase hexadecimal."
            )
        if not 0.0 < self.measured_peak_allocated_vram_gb < 10.5:
            raise ValueError(
                "memory-profile measured_peak_allocated_vram_gb must lie strictly below 10.5."
            )
        if not self.permitted_limit_names:
            raise ValueError("memory-profile must list the exceeded resource limits it permits.")


@dataclass(frozen=True)
class DenseLevelDynamicsConfig:
    """Fixed-capacity configuration for the first Dense Level--Dynamics pilot."""

    encoder: EJEPATubeletLHRConfig = field(default_factory=EJEPATubeletLHRConfig)
    projection_dim: int = 96
    predictor_dim: int = 96
    predictor_layers: int = 2
    predictor_heads: int = 4
    predictor_mlp_ratio: int = 2
    patch_query_chunk_size: int = 60
    max_batch_size: int = 2
    max_temporal_steps: int = 5
    max_patches: int = 240
    max_horizons: int = 3
    ema_start: float = 0.99
    ema_end: float = 0.9999
    ema_total_updates: int = 1_000
    memory_profile: MemoryProfileException | None = None

    def __post_init__(self) -> None:
        if self.projection_dim <= 0 or self.predictor_dim <= 0:
            raise ValueError("projection_dim and predictor_dim must be positive.")
        if self.projection_dim != self.predictor_dim:
            raise ValueError(
                "projection_dim and predictor_dim must match in the aligned-patch predictor."
            )
        if self.predictor_dim % self.predictor_heads:
            raise ValueError("predictor_dim must be divisible by predictor_heads.")
        if self.predictor_layers <= 0 or self.predictor_heads <= 0 or self.predictor_mlp_ratio <= 0:
            raise ValueError("predictor layers, heads and MLP ratio must be positive.")
        if self.patch_query_chunk_size <= 0:
            raise ValueError("patch_query_chunk_size must be positive.")
        if (
            min(
                self.max_batch_size,
                self.max_temporal_steps,
                self.max_patches,
                self.max_horizons,
                self.ema_total_updates,
            )
            <= 0
        ):
            raise ValueError("resident limits and ema_total_updates must be positive.")
        if not 0.0 < self.ema_start <= self.ema_end < 1.0:
            raise ValueError("EMA momenta must satisfy 0 < start <= end < 1.")
        self._validate_static_limits()

    def _validate_static_limits(self) -> None:
        limits = {
            "projection_dim": (self.projection_dim, 96),
            "predictor_dim": (self.predictor_dim, 96),
            "predictor_layers": (self.predictor_layers, 2),
            "predictor_heads": (self.predictor_heads, 4),
            "predictor_mlp_ratio": (self.predictor_mlp_ratio, 2),
            "patch_query_chunk_size": (self.patch_query_chunk_size, 60),
            "max_batch_size": (self.max_batch_size, 2),
            "max_temporal_steps": (self.max_temporal_steps, 5),
            "max_patches": (self.max_patches, 240),
            "max_horizons": (self.max_horizons, 3),
        }
        exceeded = tuple(name for name, (value, maximum) in limits.items() if value > maximum)
        if exceeded:
            self._require_memory_profile(exceeded)

    @staticmethod
    def resource_request_sha256_from_mapping(value: Mapping[str, Any]) -> str:
        """Hash a prospective config mapping before constructing a profile exception.

        This lets a measured-profile author bind an otherwise invalid larger request
        without first constructing a config that would correctly fail preflight.
        The input must contain the same fields emitted by ``dataclasses.asdict``.
        """

        payload_value = dict(value)
        payload_value.pop("memory_profile", None)
        payload = json.dumps(payload_value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resource_request_sha256(self) -> str:
        """Return the hash a measured memory profile must bind to exactly."""

        return self.resource_request_sha256_from_mapping(asdict(self))

    def resident_shape_request_sha256(
        self,
        *,
        batch: int,
        steps: int,
        patches: int,
        horizons: int,
    ) -> str:
        """Hash an explicit runtime residency request for a measured profile exception."""

        value = asdict(self)
        value.pop("memory_profile", None)
        value["runtime_shape"] = {
            "batch": batch,
            "steps": steps,
            "patches": patches,
            "horizons": horizons,
        }
        return self.resource_request_sha256_from_mapping(value)

    def _require_memory_profile(
        self,
        exceeded: tuple[str, ...],
        *,
        request_sha256: str | None = None,
    ) -> None:
        profile = self.memory_profile
        if profile is None:
            raise ValueError(
                "Dense Level-Dynamics resident limit exceeded: "
                + ", ".join(exceeded)
                + ". A versioned measured memory profile is required."
            )
        expected_hash = self.resource_request_sha256() if request_sha256 is None else request_sha256
        if profile.resource_request_sha256 != expected_hash:
            raise ValueError("memory-profile hash does not bind to this exact resource request.")
        unapproved = sorted(set(exceeded) - set(profile.permitted_limit_names))
        if unapproved:
            raise ValueError(
                "memory-profile does not explicitly permit exceeded limits: "
                + ", ".join(unapproved)
            )

    def validate_resident_shape(
        self, *, batch: int, steps: int, patches: int, horizons: int
    ) -> None:
        """Reject a runtime shape above the frozen microbatch residency contract."""

        requested = {
            "runtime_batch": (batch, self.max_batch_size),
            "runtime_temporal_steps": (steps, self.max_temporal_steps),
            "runtime_patches": (patches, self.max_patches),
            "runtime_horizons": (horizons, self.max_horizons),
        }
        exceeded = tuple(name for name, (value, maximum) in requested.items() if value > maximum)
        if exceeded:
            # Runtime bounds are never widened implicitly by the static default.  A
            # profile can permit them only when it was explicitly signed for this
            # configuration and names the runtime limits as well.
            self._require_memory_profile(
                exceeded,
                request_sha256=self.resident_shape_request_sha256(
                    batch=batch,
                    steps=steps,
                    patches=patches,
                    horizons=horizons,
                ),
            )


@dataclass
class DenseRepresentationOutput:
    """Online or EMA dense representation before predictor processing."""

    level_tokens: torch.Tensor
    dynamics_tokens: torch.Tensor
    valid_patch_mask: torch.Tensor
    geometry: PatchGeometry
    encoded_grid_height: int
    encoded_grid_width: int
    patch_coordinates: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class PredictorOutput:
    """Factorized predictions for each requested future horizon."""

    level_tokens: torch.Tensor
    dynamics_tokens: torch.Tensor
    residual_tokens: torch.Tensor


@dataclass
class DenseLevelDynamicsOutput:
    """Complete forward result used by all four preregistered objective arms."""

    level_tokens: torch.Tensor
    dynamics_tokens: torch.Tensor
    valid_patch_mask: torch.Tensor
    geometry: PatchGeometry
    patch_coordinates: torch.Tensor
    predicted_level_tokens: torch.Tensor
    predicted_dynamics_tokens: torch.Tensor
    predicted_residual_tokens: torch.Tensor
    target_level_tokens: torch.Tensor
    target_dynamics_tokens: torch.Tensor
    target_reference_dynamics_tokens: torch.Tensor
    target_reference_valid_patch_mask: torch.Tensor
    valid_target_patch_mask: torch.Tensor
    horizon_delta_t_s: torch.Tensor
    diagnostics: dict[str, torch.Tensor]

    @property
    def future_target_level_tokens(self) -> torch.Tensor:
        """Alias for future target level patches used by dense level supervision."""

        return self.target_level_tokens

    @property
    def future_target_dynamics_tokens(self) -> torch.Tensor:
        """Alias for future target dynamics patches used by NCE supervision."""

        return self.target_dynamics_tokens


class ProjectionHead(nn.Module):
    """Small independent projection from shared encoder tokens into one latent role."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project the final dimension without changing batch, time or patch axes."""

        return self.projection(self.norm(tokens))


class DenseLevelDynamicsRepresentation(nn.Module):
    """The shared LHR encoder plus explicit level and dynamics projection heads."""

    def __init__(
        self,
        config: DenseLevelDynamicsConfig,
        *,
        encoder: EJEPATubeletLHR | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else EJEPATubeletLHR(config.encoder)
        if self.encoder.config != config.encoder:
            raise ValueError(
                "The supplied LHR encoder config must exactly match DenseLevelDynamicsConfig."
            )
        self.level_head = ProjectionHead(self.encoder.config.embed_dim, config.projection_dim)
        self.dynamics_head = ProjectionHead(self.encoder.config.embed_dim, config.projection_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        valid_temporal_mask: torch.Tensor | None = None,
    ) -> DenseRepresentationOutput:
        """Encode raw event volumes into dense level and dynamics token grids."""

        features: HighResFeatures = self.encoder.forward_features(
            inputs,
            valid_temporal_mask=valid_temporal_mask,
        )
        valid = features.valid_patch_mask
        level = self.level_head(features.tokens).masked_fill(~valid.unsqueeze(-1), 0.0)
        dynamics = self.dynamics_head(features.tokens).masked_fill(~valid.unsqueeze(-1), 0.0)
        return DenseRepresentationOutput(
            level_tokens=level,
            dynamics_tokens=dynamics,
            valid_patch_mask=valid,
            geometry=features.geometry,
            encoded_grid_height=features.encoded_grid_height,
            encoded_grid_width=features.encoded_grid_width,
            patch_coordinates=features.post_merge_patch_coordinates,
            diagnostics=dict(features.diagnostics),
        )


class FourierDeltaTimeEmbedding(nn.Module):
    """Learn a floating-point Delta-t conditioning from fixed Fourier features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(128.0), steps=8))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.network = nn.Sequential(
            nn.Linear(16, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, delta_t_s: torch.Tensor) -> torch.Tensor:
        """Return learned conditioning for a finite positive floating Delta-t tensor."""

        if delta_t_s.ndim != 1:
            raise ValueError("delta_t_s must have shape [B].")
        if not torch.isfinite(delta_t_s).all() or bool((delta_t_s <= 0).any()):
            raise ValueError("horizon_delta_t_s must contain finite positive floating durations.")
        phase = delta_t_s[:, None] * self.frequencies[None, :].to(delta_t_s)
        return self.network(torch.cat((phase.sin(), phase.cos()), dim=-1))


class _FactorizedPatchBranch(nn.Module):
    """Temporal-only transformer branch evaluated one spatial patch at a time."""

    def __init__(self, dim: int, *, layers: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.delta_t = FourierDeltaTimeEmbedding(dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * mlp_ratio,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_valid_patch_mask: torch.Tensor,
        patch_coordinates: torch.Tensor,
        horizon_delta_t_s: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        """Predict ``[B,H,P,D]`` by serial horizon/chunk processing without spatial attention."""

        if context_tokens.ndim != 4 or context_valid_patch_mask.shape != context_tokens.shape[:3]:
            raise ValueError("context tokens/mask must be [B,T,P,D] and [B,T,P].")
        batch, steps, patches, dim = context_tokens.shape
        if patch_coordinates.shape != (patches, 2):
            raise ValueError("patch_coordinates must have shape [P,2] aligned to context tokens.")
        if horizon_delta_t_s.ndim != 2 or horizon_delta_t_s.shape[0] != batch:
            raise ValueError("horizon_delta_t_s must have shape [B,H].")
        horizons = horizon_delta_t_s.shape[1]
        output = context_tokens.new_zeros(batch, horizons, patches, dim)
        # The query is appended after the causal context sequence.  Its position
        # can attend to every context token, whereas no context token can attend to
        # the query or a later context token.
        causal = torch.triu(
            torch.ones(steps + 1, steps + 1, dtype=torch.bool, device=context_tokens.device),
            diagonal=1,
        )
        positions = self.position(
            patch_coordinates.to(device=context_tokens.device, dtype=context_tokens.dtype)
        )
        for horizon_index in range(horizons):
            delta = self.delta_t(horizon_delta_t_s[:, horizon_index].to(context_tokens.dtype))
            for start in range(0, patches, chunk_size):
                stop = min(start + chunk_size, patches)
                count = stop - start
                context = (
                    context_tokens[:, :, start:stop]
                    .permute(0, 2, 1, 3)
                    .reshape(batch * count, steps, dim)
                )
                context_valid = (
                    context_valid_patch_mask[:, :, start:stop]
                    .permute(0, 2, 1)
                    .reshape(batch * count, steps)
                )
                query = (positions[start:stop].unsqueeze(0) + delta.unsqueeze(1)).reshape(
                    batch * count, 1, dim
                )
                value = torch.cat((context, query), dim=1)
                padding = torch.cat(
                    (
                        ~context_valid,
                        torch.zeros(
                            batch * count, 1, dtype=torch.bool, device=context_tokens.device
                        ),
                    ),
                    dim=1,
                )
                for layer in self.layers:
                    value = layer(value, src_mask=causal, src_key_padding_mask=padding)
                output[:, horizon_index, start:stop] = self.final_norm(value[:, -1]).reshape(
                    batch, count, dim
                )
        return output


class FactorizedAlignedPatchPredictor(nn.Module):
    """Predict level, dynamics and residual patches with no target-content input."""

    def __init__(self, config: DenseLevelDynamicsConfig) -> None:
        super().__init__()
        kwargs = {
            "layers": config.predictor_layers,
            "heads": config.predictor_heads,
            "mlp_ratio": config.predictor_mlp_ratio,
        }
        self.level_branch = _FactorizedPatchBranch(config.predictor_dim, **kwargs)
        self.dynamics_branch = _FactorizedPatchBranch(config.predictor_dim, **kwargs)
        # This head predicts the residual directly.  It never subtracts an online
        # reference representation, preserving the target-only residual contract.
        # Keeping it as a readout rather than a third transformer preserves the
        # fixed two-layer factorized predictor capacity for every objective arm.
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(config.predictor_dim),
            nn.Linear(config.predictor_dim, config.predictor_dim),
        )
        self.patch_query_chunk_size = config.patch_query_chunk_size

    def forward(
        self,
        level_tokens: torch.Tensor,
        dynamics_tokens: torch.Tensor,
        valid_patch_mask: torch.Tensor,
        patch_coordinates: torch.Tensor,
        horizon_delta_t_s: torch.Tensor,
    ) -> PredictorOutput:
        """Produce factorized forecasts; future targets are intentionally absent."""

        dynamics = self.dynamics_branch(
            dynamics_tokens,
            valid_patch_mask,
            patch_coordinates,
            horizon_delta_t_s,
            chunk_size=self.patch_query_chunk_size,
        )
        return PredictorOutput(
            level_tokens=self.level_branch(
                level_tokens,
                valid_patch_mask,
                patch_coordinates,
                horizon_delta_t_s,
                chunk_size=self.patch_query_chunk_size,
            ),
            dynamics_tokens=dynamics,
            residual_tokens=self.residual_projection(dynamics),
        )


def _last_valid_time(
    tokens: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the last valid token per ``[B,P]`` without assuming dense temporal validity."""

    if tokens.ndim != 4 or valid_mask.shape != tokens.shape[:3]:
        raise ValueError("tokens/mask must have shapes [B,T,P,D] and [B,T,P].")
    batch, steps, patches, dim = tokens.shape
    indices = torch.arange(steps, device=tokens.device).reshape(1, steps, 1)
    last = torch.where(valid_mask, indices, torch.full_like(indices, -1)).amax(dim=1)
    selected_valid = last >= 0
    gathered = tokens.gather(
        1,
        last.clamp_min(0).reshape(batch, 1, patches, 1).expand(batch, 1, patches, dim),
    ).squeeze(1)
    return gathered.masked_fill(~selected_valid.unsqueeze(-1), 0.0), selected_valid


class DenseLevelDynamicsJEPA(nn.Module):
    """EMA-target Dense Level--Dynamics JEPA with an exact downstream LHR encoder."""

    def __init__(
        self,
        config: DenseLevelDynamicsConfig | None = None,
        *,
        encoder: EJEPATubeletLHR | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DenseLevelDynamicsConfig()
        self.online_representation = DenseLevelDynamicsRepresentation(self.config, encoder=encoder)
        self.target_representation = copy.deepcopy(self.online_representation)
        self.target_representation.requires_grad_(False)
        self.target_representation.eval()
        self.predictor = FactorizedAlignedPatchPredictor(self.config)
        self._assert_target_exact()

    def _assert_target_exact(self) -> None:
        online = self.online_representation.state_dict()
        target = self.target_representation.state_dict()
        if online.keys() != target.keys():
            raise RuntimeError(
                "EMA target state keys differ from the online representation at initialization."
            )
        if any(not torch.equal(online[key], target[key]) for key in online):
            raise RuntimeError("EMA target must be an exact deep copy at initialization.")

    @property
    def online_encoder(self) -> EJEPATubeletLHR:
        """Expose the exact downstream online LHR backbone without projection heads."""

        return self.online_representation.encoder

    @property
    def target_encoder(self) -> EJEPATubeletLHR:
        """Expose the frozen EMA LHR backbone for inspection-only compatibility."""

        return self.target_representation.encoder

    @property
    def online_level_head(self) -> ProjectionHead:
        """Return the trainable online level projection head."""

        return self.online_representation.level_head

    @property
    def online_dynamics_head(self) -> ProjectionHead:
        """Return the trainable online dynamics projection head."""

        return self.online_representation.dynamics_head

    def train(self, mode: bool = True) -> DenseLevelDynamicsJEPA:
        """Keep the EMA target eval-only even while online modules enter train mode."""

        super().train(mode)
        self.target_representation.eval()
        return self

    def ema_momentum(self, update_index: int, *, total_updates: int | None = None) -> float:
        """Return cosine-scheduled target momentum from 0.99 toward 0.9999."""

        total = self.config.ema_total_updates if total_updates is None else total_updates
        if total <= 0 or update_index < 0:
            raise ValueError("EMA update_index must be non-negative and total_updates positive.")
        fraction = min(update_index, max(total - 1, 0)) / max(total - 1, 1)
        return self.config.ema_start + (self.config.ema_end - self.config.ema_start) * (
            0.5 - 0.5 * math.cos(math.pi * fraction)
        )

    @torch.no_grad()
    def update_target_ema(self, update_index: int, *, total_updates: int | None = None) -> float:
        """Update the frozen target representation using exact name-aligned EMA arithmetic."""

        momentum = self.ema_momentum(update_index, total_updates=total_updates)
        online_state = self.online_representation.state_dict()
        target_state = self.target_representation.state_dict()
        if online_state.keys() != target_state.keys():
            raise RuntimeError("Online and target representation state keys diverged.")
        for key, target_value in target_state.items():
            source_value = online_state[key]
            if torch.is_floating_point(target_value):
                target_value.mul_(momentum).add_(source_value, alpha=1.0 - momentum)
            else:
                target_value.copy_(source_value)
        self.target_representation.requires_grad_(False)
        self.target_representation.eval()
        return momentum

    def downstream_backbone_payload(self) -> dict[str, Any]:
        """Return the only SSL state permitted to transfer into downstream inference."""

        encoder = self.online_representation.encoder
        return {
            "online_encoder_state_dict": encoder.backbone_state_dict(),
            "online_encoder_config": encoder.backbone_structural_config(),
        }

    def _preflight_inputs(
        self,
        context_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
        horizon_delta_t_s: torch.Tensor,
    ) -> None:
        if context_inputs.ndim != 5:
            raise ValueError("context_inputs must have shape [B,T,C,H,W].")
        if future_inputs.ndim != 6:
            raise ValueError("future_inputs must have shape [B,H,T,C,H,W].")
        batch, steps, channels, height, width = context_inputs.shape
        future_batch, horizons, future_steps, future_channels, future_height, future_width = (
            future_inputs.shape
        )
        if (future_batch, future_steps, future_channels, future_height, future_width) != (
            batch,
            steps,
            channels,
            height,
            width,
        ):
            raise ValueError(
                "future_inputs must preserve context batch, T, C and spatial dimensions."
            )
        if channels != self.config.encoder.in_channels:
            raise ValueError(
                "context/future inputs must contain "
                f"{self.config.encoder.in_channels} event channels."
            )
        if (
            horizon_delta_t_s.shape != (batch, horizons)
            or not horizon_delta_t_s.is_floating_point()
        ):
            raise ValueError("horizon_delta_t_s must be a floating tensor with shape [B,H].")
        grid_height = math.ceil(height / self.config.encoder.patch_size)
        grid_width = math.ceil(width / self.config.encoder.patch_size)
        if self.config.encoder.merge_2x2:
            grid_height = math.ceil(grid_height / 2)
            grid_width = math.ceil(grid_width / 2)
        self.config.validate_resident_shape(
            batch=batch,
            steps=steps,
            patches=grid_height * grid_width,
            horizons=horizons,
        )

    @staticmethod
    def _validate_temporal_masks(
        context_valid: torch.Tensor | None,
        future_valid: torch.Tensor | None,
        *,
        batch: int,
        horizons: int,
        steps: int,
    ) -> None:
        if context_valid is not None and (
            context_valid.dtype != torch.bool or context_valid.shape != (batch, steps)
        ):
            raise ValueError("context_valid_temporal_mask must be bool with shape [B,T].")
        if future_valid is not None and (
            future_valid.dtype != torch.bool or future_valid.shape != (batch, horizons, steps)
        ):
            raise ValueError("future_valid_temporal_mask must be bool with shape [B,H,T].")

    def forward(
        self,
        context_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
        horizon_delta_t_s: torch.Tensor,
        *,
        context_valid_temporal_mask: torch.Tensor | None = None,
        future_valid_temporal_mask: torch.Tensor | None = None,
    ) -> DenseLevelDynamicsOutput:
        """Encode context/future event windows and predict dense aligned future patches.

        ``future_inputs`` are consumed by the frozen target representation only.  The
        predictor receives online context tokens, post-merge patch coordinates and
        floating Delta-t values; it never receives target content.
        """

        self._preflight_inputs(context_inputs, future_inputs, horizon_delta_t_s)
        batch, steps = context_inputs.shape[:2]
        horizons = future_inputs.shape[1]
        self._validate_temporal_masks(
            context_valid_temporal_mask,
            future_valid_temporal_mask,
            batch=batch,
            horizons=horizons,
            steps=steps,
        )
        online = self.online_representation(
            context_inputs,
            valid_temporal_mask=context_valid_temporal_mask,
        )
        with torch.no_grad():
            target_reference = self.target_representation(
                context_inputs,
                valid_temporal_mask=context_valid_temporal_mask,
            )
            reference_dynamics, reference_valid = _last_valid_time(
                target_reference.dynamics_tokens,
                target_reference.valid_patch_mask,
            )
            target_levels: list[torch.Tensor] = []
            target_dynamics: list[torch.Tensor] = []
            target_valids: list[torch.Tensor] = []
            for horizon_index in range(horizons):
                target = self.target_representation(
                    future_inputs[:, horizon_index],
                    valid_temporal_mask=(
                        future_valid_temporal_mask[:, horizon_index]
                        if future_valid_temporal_mask is not None
                        else None
                    ),
                )
                if (
                    target.encoded_grid_height != online.encoded_grid_height
                    or target.encoded_grid_width != online.encoded_grid_width
                    or not torch.equal(target.patch_coordinates, online.patch_coordinates)
                ):
                    raise ValueError(
                        "Future target geometry must exactly match post-merge context patches."
                    )
                level, valid = _last_valid_time(target.level_tokens, target.valid_patch_mask)
                dynamics, dynamics_valid = _last_valid_time(
                    target.dynamics_tokens,
                    target.valid_patch_mask,
                )
                if not torch.equal(valid, dynamics_valid):
                    raise RuntimeError(
                        "Target level/dynamics patch validity diverged unexpectedly."
                    )
                target_levels.append(level.detach())
                target_dynamics.append(dynamics.detach())
                target_valids.append(valid.detach())
        target_level_tokens = torch.stack(target_levels, dim=1)
        target_dynamics_tokens = torch.stack(target_dynamics, dim=1)
        valid_target_patch_mask = torch.stack(target_valids, dim=1)
        predictions = self.predictor(
            online.level_tokens,
            online.dynamics_tokens,
            online.valid_patch_mask,
            online.patch_coordinates,
            horizon_delta_t_s,
        )
        diagnostics = dict(online.diagnostics)
        diagnostics.update(
            {
                "target_valid_patch_count": valid_target_patch_mask.sum().detach().to(torch.int64),
                "target_reference_valid_patch_count": reference_valid.sum()
                .detach()
                .to(torch.int64),
                "horizon_count": torch.tensor(
                    horizons, device=context_inputs.device, dtype=torch.int64
                ),
            }
        )
        return DenseLevelDynamicsOutput(
            level_tokens=online.level_tokens,
            dynamics_tokens=online.dynamics_tokens,
            valid_patch_mask=online.valid_patch_mask,
            geometry=online.geometry,
            patch_coordinates=online.patch_coordinates,
            predicted_level_tokens=predictions.level_tokens,
            predicted_dynamics_tokens=predictions.dynamics_tokens,
            predicted_residual_tokens=predictions.residual_tokens,
            target_level_tokens=target_level_tokens,
            target_dynamics_tokens=target_dynamics_tokens,
            target_reference_dynamics_tokens=reference_dynamics.detach(),
            target_reference_valid_patch_mask=reference_valid.detach(),
            valid_target_patch_mask=valid_target_patch_mask,
            horizon_delta_t_s=horizon_delta_t_s,
            diagnostics=diagnostics,
        )


__all__ = [
    "DenseLevelDynamicsConfig",
    "DenseLevelDynamicsJEPA",
    "DenseLevelDynamicsOutput",
    "DenseLevelDynamicsRepresentation",
    "DenseRepresentationOutput",
    "FactorizedAlignedPatchPredictor",
    "FourierDeltaTimeEmbedding",
    "MemoryProfileException",
    "PredictorOutput",
    "ProjectionHead",
]

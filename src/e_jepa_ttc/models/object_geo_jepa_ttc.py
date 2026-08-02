"""Gate-configurable OGE-JEPA-TTC architecture for EvTTC experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.geometry.ego_motion_compensation import CameraYawDerotator
from e_jepa_ttc.models.attention_residual_router import TaskSpecificAttentionResiduals
from e_jepa_ttc.models.block_causal_transformer import BlockCausalTransformer
from e_jepa_ttc.models.dense_patch_ttc import (
    BaseEventTubeletBackbone,
    DensePatchEventBackbone,
)
from e_jepa_ttc.models.geometry_mixture import GeometryMixture
from e_jepa_ttc.models.highres_refiner import HighResolutionMaskRefiner
from e_jepa_ttc.models.hybrid_spatiotemporal_mixer import HybridSpatiotemporalMixer
from e_jepa_ttc.models.residual_ttc import BoundedInverseTTCResidual
from e_jepa_ttc.models.risk_selector import RiskSelector
from e_jepa_ttc.models.spatial_patch_mixer import SpatialPatchMixer
from e_jepa_ttc.models.target_query import TargetBackgroundQuery
from e_jepa_ttc.models.temporal_kda import KDALayoutMetadata, TemporalKDAStack
from e_jepa_ttc.models.uncertainty_head import TTCUncertaintyHead


@dataclass(frozen=True)
class OGEConfig:
    """Single-factor switches matching the staged architecture matrix."""

    in_channels: int
    event_channels: int | None = None
    backbone: str = "compact_dense"
    base_encoder_checkpoint: str | None = None
    allow_random_base_initialization: bool = False
    dim: int = 128
    backbone_depth: int = 4
    heads: int = 4
    temporal_depth: int = 3
    head_mode: str = "global"
    bbox_roi_grid_size: int = 3
    bbox_roi_expansion: float = 1.0
    use_attention_residuals: bool = False
    temporal_mixer: str = "block_causal"
    use_target_query: bool = False
    use_highres_refiner: bool = False
    geometry_mode: str = "none"
    use_bounded_residual: bool = False
    use_uncertainty: bool = False
    use_yaw_derotation: bool = False
    action_dim: int = 8
    bbox_source: str = "ground_truth"
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    residual_limit: float = 0.30
    sequence_embedding_forbidden: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        event_channels = self.resolved_event_channels
        if event_channels < 4 or event_channels % 2 or event_channels > self.in_channels:
            raise ValueError("event_channels must be even, >=4 and no greater than in_channels.")
        if self.backbone not in {"compact_dense", "base_event_tubelet"}:
            raise ValueError("backbone must be compact_dense or base_event_tubelet.")
        if (
            self.backbone == "base_event_tubelet"
            and self.base_encoder_checkpoint is None
            and not self.allow_random_base_initialization
        ):
            raise ValueError("BASE EventTubelet backbone requires its audited SSL checkpoint.")
        if self.base_encoder_checkpoint is not None and self.allow_random_base_initialization:
            raise ValueError(
                "Select either an audited BASE checkpoint or explicit random initialization."
            )
        if self.backbone == "base_event_tubelet" and (
            self.in_channels != 21 or self.dim != 192 or self.backbone_depth != 6
        ):
            raise ValueError("Audited BASE requires 21 channels, dim=192 and depth=6.")
        if self.head_mode not in {"global", "dense", "bbox_roi"}:
            raise ValueError("head_mode must be global, dense or bbox_roi.")
        if self.bbox_roi_grid_size <= 0 or self.bbox_roi_expansion <= 0.0:
            raise ValueError("BBox ROI grid size and expansion must be positive.")
        if self.temporal_mixer not in {
            "block_causal",
            "object_kda",
            "aligned_patch_kda",
        }:
            raise ValueError("Invalid temporal_mixer.")
        if self.geometry_mode not in {"none", "deterministic", "router", "top2"}:
            raise ValueError("Invalid geometry_mode.")
        if self.bbox_source not in {"ground_truth", "predicted"}:
            raise ValueError("bbox_source must be ground_truth or predicted.")
        if self.bbox_source == "predicted" and not self.use_target_query:
            raise ValueError("Predicted geometry requires use_target_query=True.")
        if self.use_highres_refiner and not self.use_target_query:
            raise ValueError("High-resolution refinement requires use_target_query=True.")

    @property
    def resolved_event_channels(self) -> int:
        """Return channels that contain voxels, excluding BASE auxiliary maps."""

        if self.event_channels is not None:
            return self.event_channels
        # The audited BASE tensor is 10 polarity-separated event bins followed
        # by 11 auxiliary channels.  Compact tests/custom encoders normally
        # contain events only.
        return 10 if self.in_channels == 21 else self.in_channels


@dataclass
class OGEOutput:
    """TTC distribution, masks, experts and diagnostic signals."""

    ttc_seconds: torch.Tensor
    inverse_ttc_mean: torch.Tensor
    inverse_ttc_log_variance: torch.Tensor
    risk_logits: torch.Tensor
    selected_object_index: torch.Tensor
    mask_logits: torch.Tensor | None
    refined_mask_logits: torch.Tensor | None
    predicted_boxes: torch.Tensor | None
    geometry_estimates: torch.Tensor | None
    geometry_confidence: torch.Tensor | None
    geometry_weights: torch.Tensor | None
    residual: torch.Tensor
    object_token: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    # Raw backbone tokens are exposed only for optional training-time latent
    # regularization. Inference heads and serialized weights remain unchanged.
    backbone_dense_tokens: torch.Tensor | None = None


class ObjectGeometryJEPATTC(nn.Module):
    """Dense Patch Policy model with optional localization and physical solver."""

    def __init__(self, config: OGEConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = (
            BaseEventTubeletBackbone(
                config.base_encoder_checkpoint,
                allow_random_initialization=config.allow_random_base_initialization,
            )
            if config.backbone == "base_event_tubelet"
            else DensePatchEventBackbone(
                config.in_channels,
                dim=config.dim,
                depth=config.backbone_depth,
            )
        )
        # Construct the common prediction head before any arm-specific module.
        # With a shared seed this makes its initial parameters byte-identical
        # across A0/A1/A2/K1 even though the optional mixers differ in size.
        self.direct_log_ttc_head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(config.dim // 2, 1),
        )
        # The matched global control receives the same causal frame history as
        # every dense arm.  Its only architectural difference is that spatial
        # patches are mean-pooled before temporal mixing.  The previous proxy
        # encoded only the last frame, which confounded Patch Policy with
        # additional temporal context.
        self.global_temporal = (
            BlockCausalTransformer(
                config.dim,
                heads=config.heads,
                depth=config.temporal_depth,
            )
            if config.head_mode == "global"
            else None
        )
        self.mixer = (
            None
            if config.head_mode == "global" or config.temporal_mixer == "object_kda"
            else HybridSpatiotemporalMixer(
                config.dim,
                mode=config.temporal_mixer,
                heads=config.heads,
                depth=config.temporal_depth,
            )
        )
        self.object_kda_spatial = (
            SpatialPatchMixer(config.dim, heads=config.heads)
            if config.temporal_mixer == "object_kda"
            else None
        )
        self.object_kda = (
            TemporalKDAStack(
                config.dim,
                heads=config.heads,
                depth=config.temporal_depth,
            )
            if config.temporal_mixer == "object_kda"
            else None
        )
        # Optional modules are constructed after the common direct head and
        # reference temporal/spatial mixers so enabling AttnRes cannot shift
        # the initialization of modules shared with A1.
        self.attention_residuals = (
            TaskSpecificAttentionResiduals(config.dim, config.backbone_depth)
            if config.use_attention_residuals
            else None
        )
        self.target_query = TargetBackgroundQuery(config.dim) if config.use_target_query else None
        self.highres_refiner = (
            HighResolutionMaskRefiner(config.in_channels) if config.use_highres_refiner else None
        )
        learned_router = config.geometry_mode in {"router", "top2"}
        self.geometry = (
            GeometryMixture(
                config.dim,
                learned_router=learned_router,
                inference_top_k=2 if config.geometry_mode == "top2" else None,
            )
            if config.geometry_mode != "none"
            else None
        )
        self.residual_head = (
            BoundedInverseTTCResidual(config.dim, residual_limit=config.residual_limit)
            if config.use_bounded_residual
            else None
        )
        self.uncertainty_head = TTCUncertaintyHead(config.dim) if config.use_uncertainty else None
        self.risk_selector = RiskSelector(config.dim, config.risk_thresholds_s)
        self.yaw_derotator = (
            CameraYawDerotator(config.action_dim) if config.use_yaw_derotation else None
        )

    def _query_history(
        self,
        tokens: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.target_query is not None
        object_tokens: list[torch.Tensor] = []
        mask_logits: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []
        for step in range(tokens.shape[1]):
            query = self.target_query(tokens[:, step], spatial_shape)
            object_tokens.append(query.object_token)
            mask_logits.append(query.mask_logits)
            boxes.append(query.box_xyxy)
        return (
            torch.stack(object_tokens, dim=1),
            torch.stack(mask_logits, dim=1),
            torch.stack(boxes, dim=1),
        )

    def _dense_pool_history(self, tokens: torch.Tensor) -> torch.Tensor:
        # Mean pooling is intentionally parameter-free.  In the matched
        # A0/A1 comparison the location of this same reduction (before or
        # after temporal patch mixing) is the factor under test.
        return tokens.mean(dim=2)

    def _bbox_roi_pool_history(
        self,
        tokens: torch.Tensor,
        boxes: torch.Tensor,
        object_mask: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Pool dense tokens inside each causal ground-truth object box.

        ``grid_sample`` keeps small boxes useful even when no coarse patch
        centre lies strictly inside the ROI. Multiple valid objects are
        averaged; frames without a valid object fall back to global pooling.
        The boxes are normalized ``xyxy`` coordinates supplied by the
        benchmark and are never predicted from future information.
        """

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,P,D].")
        if boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [B,T,O,4].")
        if object_mask.shape != boxes.shape[:-1]:
            raise ValueError("object_mask must match boxes [B,T,O].")
        batch, steps, patches, dim = tokens.shape
        grid_height, grid_width = spatial_shape
        if patches != grid_height * grid_width:
            raise ValueError("Token count does not match the backbone spatial grid.")
        if boxes.shape[:2] != (batch, steps):
            raise ValueError("Box history must match token batch and time dimensions.")

        box_values = boxes.to(dtype=tokens.dtype).clamp(0.0, 1.0)
        center = 0.5 * (box_values[..., :2] + box_values[..., 2:])
        half_size = (
            0.5
            * (box_values[..., 2:] - box_values[..., :2]).clamp_min(1e-6)
            * self.config.bbox_roi_expansion
        )
        lower = (center - half_size).clamp(0.0, 1.0)
        upper = (center + half_size).clamp(0.0, 1.0)

        samples = self.config.bbox_roi_grid_size
        fractions = (
            torch.arange(samples, device=tokens.device, dtype=tokens.dtype) + 0.5
        ) / samples
        x = lower[..., 0, None] + (upper - lower)[..., 0, None] * fractions
        y = lower[..., 1, None] + (upper - lower)[..., 1, None] * fractions
        sample_y, sample_x = torch.meshgrid(
            torch.arange(samples, device=tokens.device),
            torch.arange(samples, device=tokens.device),
            indexing="ij",
        )
        grid = torch.stack(
            (
                x[..., sample_x] * 2.0 - 1.0,
                y[..., sample_y] * 2.0 - 1.0,
            ),
            dim=-1,
        )

        objects = boxes.shape[2]
        feature_maps = (
            tokens.reshape(batch * steps, grid_height, grid_width, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        feature_maps = (
            feature_maps[:, None]
            .expand(-1, objects, -1, -1, -1)
            .reshape(batch * steps * objects, dim, grid_height, grid_width)
        )
        sampled = functional.grid_sample(
            feature_maps,
            grid.reshape(batch * steps * objects, samples, samples, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        object_tokens = sampled.mean(dim=(-1, -2)).reshape(batch, steps, objects, dim)
        valid = object_mask.to(dtype=tokens.dtype)
        pooled = (object_tokens * valid[..., None]).sum(dim=2) / valid.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1.0)
        fallback = self._dense_pool_history(tokens)
        has_object = object_mask.bool().any(dim=2, keepdim=True)
        return torch.where(has_object, pooled, fallback)

    def forward(
        self,
        context_events: torch.Tensor,
        *,
        context_times_s: torch.Tensor,
        context_boxes: torch.Tensor | None = None,
        context_object_mask: torch.Tensor | None = None,
        context_ego_actions: torch.Tensor | None = None,
        context_ego_action_mask: torch.Tensor | None = None,
        context_intrinsics_normalized: torch.Tensor | None = None,
    ) -> OGEOutput:
        """Predict from causal context; no future tensor is accepted by this API."""

        features = self.backbone(context_events)
        diagnostics: dict[str, torch.Tensor] = {}
        task_tokens: dict[str, torch.Tensor]
        if self.attention_residuals is not None:
            routed = self.attention_residuals(features.layer_tokens)
            task_tokens = routed.task_tokens
            for task, weights in routed.task_weights.items():
                diagnostics[f"attnres_{task}_max_weight"] = weights.amax(dim=-1).mean()
        else:
            task_tokens = {
                task: features.dense_tokens for task in ("mask", "motion", "geometry", "risk")
            }

        base_tokens = task_tokens["geometry"]
        mask_logits: torch.Tensor | None = None
        predicted_boxes: torch.Tensor | None = None
        object_history: torch.Tensor | None = None
        if self.config.temporal_mixer == "object_kda":
            assert self.object_kda_spatial is not None and self.object_kda is not None
            spatial = self.object_kda_spatial(task_tokens["mask"])
            if self.target_query is not None:
                object_history, mask_logits, predicted_boxes = self._query_history(
                    spatial,
                    features.spatial_shape,
                )
            else:
                object_history = self._dense_pool_history(spatial)
            k1_tokens = object_history[:, :, None]
            object_history = self.object_kda(
                k1_tokens,
                metadata=KDALayoutMetadata(
                    batch_size=k1_tokens.shape[0],
                    temporal_steps=k1_tokens.shape[1],
                    patch_count=1,
                    embedding_dim=k1_tokens.shape[3],
                ),
            )[:, :, 0]
            mixed_tokens = spatial
        elif self.mixer is not None:
            mixed_tokens = self.mixer(base_tokens)
        else:
            mixed_tokens = base_tokens

        if self.target_query is not None and object_history is None:
            object_history, mask_logits, predicted_boxes = self._query_history(
                mixed_tokens,
                features.spatial_shape,
            )
        if object_history is not None:
            object_token = object_history[:, -1]
        elif self.config.head_mode == "global":
            assert self.global_temporal is not None
            global_history = self.global_temporal(features.global_token[:, :, None, :])
            object_token = global_history[:, -1, 0]
        elif self.config.head_mode == "bbox_roi":
            if context_boxes is None or context_object_mask is None:
                raise ValueError("BBox ROI pooling requires boxes and object mask.")
            object_history = self._bbox_roi_pool_history(
                mixed_tokens,
                context_boxes,
                context_object_mask.bool(),
                features.spatial_shape,
            )
            object_token = object_history[:, -1]
            diagnostics["bbox_roi_valid_fraction"] = (
                context_object_mask.bool().any(dim=2).float().mean()
            )
            roi_area = (context_boxes[..., 2] - context_boxes[..., 0]).clamp_min(0.0) * (
                context_boxes[..., 3] - context_boxes[..., 1]
            ).clamp_min(0.0)
            roi_valid = context_object_mask.to(dtype=roi_area.dtype)
            diagnostics["bbox_roi_mean_area"] = (
                roi_area * roi_valid
            ).sum() / roi_valid.sum().clamp_min(1.0)
        else:
            object_token = mixed_tokens[:, -1].mean(dim=1)
        direct_log_ttc = self.direct_log_ttc_head(object_token).squeeze(-1)
        direct_ttc = direct_log_ttc.exp().clamp(0.1, 12.0)
        direct_inverse = direct_ttc.reciprocal()
        geometry_estimates: torch.Tensor | None = None
        geometry_confidence: torch.Tensor | None = None
        geometry_weights: torch.Tensor | None = None
        router_balance = direct_inverse.new_zeros(())
        router_entropy = direct_inverse.new_zeros(direct_inverse.shape)
        ego_angle = direct_inverse.new_zeros((direct_inverse.shape[0], context_events.shape[1]))
        if self.geometry is not None:
            if self.config.bbox_source == "predicted":
                if predicted_boxes is None:
                    raise RuntimeError("Predicted boxes unavailable.")
                boxes = predicted_boxes[:, :, None]
                object_mask = torch.ones(
                    boxes.shape[:-1],
                    device=boxes.device,
                    dtype=torch.bool,
                )
            else:
                if context_boxes is None or context_object_mask is None:
                    raise ValueError("Ground-truth geometry requires boxes and object mask.")
                boxes = context_boxes
                object_mask = context_object_mask.bool()
            if self.yaw_derotator is not None:
                if context_ego_actions is None or context_ego_action_mask is None:
                    raise ValueError(
                        "Ego compensation requires causal actions and their valid mask."
                    )
                boxes, ego_angle = self.yaw_derotator(
                    boxes,
                    context_ego_actions,
                    context_ego_action_mask.bool(),
                    context_times_s,
                    intrinsics_normalized=context_intrinsics_normalized,
                )
            masks = torch.sigmoid(mask_logits) if mask_logits is not None else None
            mixture = self.geometry(
                boxes_xyxy=boxes,
                object_mask=object_mask,
                event_frames=context_events[:, :, : self.config.resolved_event_channels],
                object_token=object_token[:, None],
                times_s=context_times_s,
                soft_masks=masks,
            )
            inverse_ttc = mixture.inverse_ttc[:, 0]
            inverse_ttc = inverse_ttc.clamp_min(1e-4)
            geometry_estimates = mixture.estimates[:, 0]
            geometry_confidence = mixture.confidence[:, 0]
            geometry_weights = mixture.weights[:, 0]
            router_balance = mixture.router_balance_loss
            router_entropy = mixture.router_entropy[:, 0]
        else:
            inverse_ttc = direct_inverse

        residual = torch.zeros_like(inverse_ttc)
        if self.residual_head is not None:
            inverse_ttc, residual = self.residual_head(inverse_ttc, object_token)
        log_variance = (
            self.uncertainty_head(object_token)
            if self.uncertainty_head is not None
            else torch.zeros_like(inverse_ttc)
        )
        risk_logits, selected = self.risk_selector(
            object_token[:, None],
            inverse_ttc[:, None],
            torch.ones_like(inverse_ttc[:, None], dtype=torch.bool),
        )
        refined: torch.Tensor | None = None
        if self.highres_refiner is not None:
            if mask_logits is None:
                raise RuntimeError("Mask logits unavailable for the refiner.")
            refined = self.highres_refiner(context_events[:, -1], mask_logits[:, -1])
        diagnostics.update(
            {
                "router_balance_loss": router_balance,
                "router_entropy": router_entropy.mean(),
                "residual_fraction": (residual.abs() / inverse_ttc.detach().clamp_min(1e-4)).mean(),
                "embedding_std": object_token.std(dim=0, unbiased=False).mean(),
                "ego_yaw_compensation_abs_rad": ego_angle.abs().mean(),
                "ego_direct_ttc_correction_abs": direct_inverse.new_zeros(()),
            }
        )
        return OGEOutput(
            ttc_seconds=inverse_ttc.clamp_min(1e-4).reciprocal().clamp_max(12.0),
            inverse_ttc_mean=inverse_ttc,
            inverse_ttc_log_variance=log_variance,
            risk_logits=risk_logits,
            selected_object_index=selected,
            mask_logits=mask_logits,
            refined_mask_logits=refined,
            predicted_boxes=predicted_boxes,
            geometry_estimates=geometry_estimates,
            geometry_confidence=geometry_confidence,
            geometry_weights=geometry_weights,
            residual=residual,
            object_token=object_token,
            diagnostics=diagnostics,
            backbone_dense_tokens=features.dense_tokens,
        )


__all__ = ["ObjectGeometryJEPATTC", "OGEConfig", "OGEOutput"]

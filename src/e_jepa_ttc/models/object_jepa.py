"""Object-centric recurrent Event-JEPA for label-efficient TTC estimation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ObjectJEPAConfig:
    """Compact configuration for the object-centric Event-JEPA model."""

    in_channels: int
    action_dim: int = 8
    embedding_dim: int = 192
    feature_dim: int = 128
    roi_size: int = 4
    predictor_depth: int = 3
    predictor_heads: int = 6
    dropout: float = 0.05
    pre_cropped_events: bool = False
    use_recurrence: bool = True
    use_geometry: bool = True
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.action_dim <= 0:
            msg = "in_channels and action_dim must be positive."
            raise ValueError(msg)
        if self.embedding_dim <= 0 or self.feature_dim <= 0 or self.roi_size <= 0:
            msg = "embedding_dim, feature_dim and roi_size must be positive."
            raise ValueError(msg)
        if self.embedding_dim % self.predictor_heads != 0:
            msg = "embedding_dim must be divisible by predictor_heads."
            raise ValueError(msg)
        if self.feature_dim % 16 != 0:
            msg = "feature_dim must be divisible by 16 for stable GroupNorm groups."
            raise ValueError(msg)
        if not self.risk_thresholds_s or any(value <= 0 for value in self.risk_thresholds_s):
            msg = "risk_thresholds_s must contain positive thresholds."
            raise ValueError(msg)


@dataclass
class ObjectEncoderOutput:
    """Object and scene latents produced by the recurrent encoder."""

    object_tokens: torch.Tensor
    scene_token: torch.Tensor
    current_box_features: torch.Tensor
    object_mask: torch.Tensor
    recurrent_states: torch.Tensor


@dataclass
class ObjectJEPAOutput:
    """Predicted/teacher latents plus geometric and TTC distributions."""

    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    predicted_geometry: torch.Tensor
    geometry_log_variance: torch.Tensor
    future_mask: torch.Tensor
    inverse_ttc_mean: torch.Tensor
    inverse_ttc_log_variance: torch.Tensor
    risk_logits: torch.Tensor
    current_object_mask: torch.Tensor
    action_conditioning_mask: torch.Tensor
    horizons_s: torch.Tensor


@dataclass
class ObjectTTCOutput:
    """Probabilistic current-TTC and threshold-risk predictions per object."""

    inverse_ttc_mean: torch.Tensor
    inverse_ttc_log_variance: torch.Tensor
    risk_logits: torch.Tensor
    object_mask: torch.Tensor


def normalized_box_features(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``xyxy`` boxes into stable geometric features."""

    if boxes_xyxy.shape[-1] != 4:
        msg = "boxes_xyxy must have a final dimension of four."
        raise ValueError(msg)
    x_min, y_min, x_max, y_max = boxes_xyxy.unbind(dim=-1)
    width = (x_max - x_min).clamp_min(1e-4)
    height = (y_max - y_min).clamp_min(1e-4)
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    return torch.stack(
        (
            center_x,
            center_y,
            width.log(),
            height.log(),
            (width * height).log(),
            (width / height).log(),
        ),
        dim=-1,
    )


def roi_sample(
    feature_map: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    output_size: int,
) -> torch.Tensor:
    """Differentiably sample normalized boxes without a torchvision dependency."""

    if feature_map.ndim != 4 or boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        msg = "Expected feature_map [B,C,H,W] and boxes_xyxy [B,O,4]."
        raise ValueError(msg)
    if feature_map.shape[0] != boxes_xyxy.shape[0] or output_size <= 0:
        msg = "ROI batch sizes must match and output_size must be positive."
        raise ValueError(msg)
    batch, object_count = boxes_xyxy.shape[:2]
    if object_count == 0:
        return feature_map.new_empty(
            (batch, 0, feature_map.shape[1], output_size, output_size)
        )
    boxes = boxes_xyxy.clamp(0.0, 1.0)
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    line = torch.linspace(0.0, 1.0, output_size, device=boxes.device, dtype=boxes.dtype)
    grid_y, grid_x = torch.meshgrid(line, line, indexing="ij")
    sample_x = x_min[..., None, None] + (x_max - x_min)[..., None, None] * grid_x
    sample_y = y_min[..., None, None] + (y_max - y_min)[..., None, None] * grid_y
    grid = torch.stack((sample_x * 2.0 - 1.0, sample_y * 2.0 - 1.0), dim=-1)
    expanded_features = feature_map[:, None].expand(
        batch,
        object_count,
        *feature_map.shape[1:],
    )
    sampled = functional.grid_sample(
        expanded_features.reshape(batch * object_count, *feature_map.shape[1:]),
        grid.reshape(batch * object_count, output_size, output_size, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(
        batch,
        object_count,
        feature_map.shape[1],
        output_size,
        output_size,
    )


class ObjectCentricRecurrentEncoder(nn.Module):
    """Encode tracked event ROIs and update one recurrent state per object."""

    def __init__(self, config: ObjectJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dim = config.embedding_dim
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, config.feature_dim // 2, 5, stride=2, padding=2),
            nn.GroupNorm(8, config.feature_dim // 2),
            nn.GELU(),
            nn.Conv2d(config.feature_dim // 2, config.feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, config.feature_dim),
            nn.GELU(),
            nn.Conv2d(config.feature_dim, config.feature_dim, 3, padding=1),
            nn.GroupNorm(8, config.feature_dim),
            nn.GELU(),
        )
        self.roi_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.embedding_dim),
            nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_dim + 1, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.recurrent_cell = nn.GRUCell(config.embedding_dim * 3, config.embedding_dim)
        self.output_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self,
        event_frames: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        object_mask: torch.Tensor,
        *,
        sampling_boxes_xyxy: torch.Tensor | None = None,
        ego_actions: torch.Tensor | None = None,
        ego_action_mask: torch.Tensor | None = None,
    ) -> ObjectEncoderOutput:
        """Encode causal frames with fixed tracked-object slots."""

        if event_frames.ndim != 5:
            msg = "event_frames must have shape [B,T,C,H,W]."
            raise ValueError(msg)
        if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[-1] != 4:
            msg = "boxes_xyxy must have shape [B,T,O,4]."
            raise ValueError(msg)
        batch, steps, channels = event_frames.shape[:3]
        if channels != self.config.in_channels:
            msg = f"Expected {self.config.in_channels} event channels, got {channels}."
            raise ValueError(msg)
        if boxes_xyxy.shape[:2] != (batch, steps) or object_mask.shape != boxes_xyxy.shape[:3]:
            msg = "Event, box and object-mask temporal shapes must match."
            raise ValueError(msg)
        sampling_boxes = boxes_xyxy if sampling_boxes_xyxy is None else sampling_boxes_xyxy
        if sampling_boxes.shape != boxes_xyxy.shape:
            msg = "sampling_boxes_xyxy must match boxes_xyxy when provided."
            raise ValueError(msg)
        object_count = boxes_xyxy.shape[2]
        if object_count <= 0:
            msg = "The tracked-object slot axis must be non-empty."
            raise ValueError(msg)
        actions, action_mask = self._validated_actions(
            event_frames,
            ego_actions=ego_actions,
            ego_action_mask=ego_action_mask,
        )
        hidden = event_frames.new_zeros((batch, object_count, self.config.embedding_dim))
        recurrent_states: list[torch.Tensor] = []
        for step in range(steps):
            features = self.spatial_encoder(event_frames[:, step])
            if self.config.pre_cropped_events:
                if object_count != 1:
                    msg = "pre_cropped_events currently supports exactly one object ROI per sample."
                    raise ValueError(msg)
                roi = functional.adaptive_avg_pool2d(
                    features,
                    (self.config.roi_size, self.config.roi_size),
                )[:, None]
            else:
                roi = roi_sample(
                    features,
                    sampling_boxes[:, step],
                    output_size=self.config.roi_size,
                )
            roi_embedding = self.roi_projection(roi.mean(dim=(-1, -2)))
            if self.config.use_geometry:
                geometry = self.geometry_projection(
                    normalized_box_features(boxes_xyxy[:, step])
                )
            else:
                geometry = torch.zeros_like(roi_embedding)
            action_input = torch.cat(
                (actions[:, step], action_mask[:, step, None].to(actions.dtype)),
                dim=-1,
            )
            action_embedding = self.action_projection(action_input)[:, None, :].expand(
                batch,
                object_count,
                self.config.embedding_dim,
            )
            previous_hidden = hidden if self.config.use_recurrence else torch.zeros_like(hidden)
            update = self.recurrent_cell(
                torch.cat((roi_embedding, geometry, action_embedding), dim=-1).reshape(
                    batch * object_count,
                    -1,
                ),
                previous_hidden.reshape(batch * object_count, -1),
            ).reshape(batch, object_count, -1)
            valid = object_mask[:, step, :, None].bool()
            hidden = torch.where(valid, update, hidden)
            recurrent_states.append(self.output_norm(hidden))
        current_mask = object_mask[:, -1].bool()
        object_tokens = self.output_norm(hidden) * current_mask[..., None]
        denominator = current_mask.sum(dim=1, keepdim=True).clamp_min(1).to(object_tokens.dtype)
        scene_token = object_tokens.sum(dim=1) / denominator
        return ObjectEncoderOutput(
            object_tokens=object_tokens,
            scene_token=scene_token,
            current_box_features=normalized_box_features(boxes_xyxy[:, -1]),
            object_mask=current_mask,
            recurrent_states=torch.stack(recurrent_states, dim=1),
        )

    def _validated_actions(
        self,
        reference: torch.Tensor,
        *,
        ego_actions: torch.Tensor | None,
        ego_action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = reference.shape[:2]
        if ego_actions is None:
            actions = reference.new_zeros((batch, steps, self.config.action_dim))
        else:
            if ego_actions.shape != (batch, steps, self.config.action_dim):
                msg = "ego_actions has an incompatible shape."
                raise ValueError(msg)
            actions = ego_actions.to(device=reference.device, dtype=reference.dtype)
        if ego_action_mask is None:
            mask = torch.zeros((batch, steps), dtype=torch.bool, device=reference.device)
        else:
            if ego_action_mask.shape != (batch, steps):
                msg = "ego_action_mask has an incompatible shape."
                raise ValueError(msg)
            mask = ego_action_mask.to(device=reference.device, dtype=torch.bool)
        actions = torch.where(mask[..., None], actions, torch.zeros_like(actions))
        return actions, mask


class MultiHorizonObjectPredictor(nn.Module):
    """Predict object latents and geometric dynamics for arbitrary horizons."""

    def __init__(self, config: ObjectJEPAConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        self.box_projection = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.horizon_projection = nn.Sequential(
            nn.Linear(16, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_dim + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.predictor_heads,
            dim_feedforward=dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.predictor_depth,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(dim)
        self.latent_head = nn.Linear(dim, dim)
        self.geometry_head = nn.Linear(dim, 6)
        self.geometry_log_variance_head = nn.Linear(dim, 6)

    def forward(
        self,
        object_tokens: torch.Tensor,
        box_features: torch.Tensor,
        object_mask: torch.Tensor,
        horizons_s: torch.Tensor,
        *,
        future_ego_actions: torch.Tensor | None = None,
        future_ego_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return latent, geometry, log-variance and action-validity tensors."""

        if object_tokens.ndim != 3 or box_features.shape[:2] != object_tokens.shape[:2]:
            msg = "Object tokens and box features must have shapes [B,O,D] and [B,O,6]."
            raise ValueError(msg)
        if horizons_s.ndim != 1 or horizons_s.numel() == 0 or torch.any(horizons_s <= 0):
            msg = "horizons_s must be a non-empty positive one-dimensional tensor."
            raise ValueError(msg)
        batch, object_count, dim = object_tokens.shape
        horizon_count = int(horizons_s.shape[0])
        horizon_features = _fourier_horizon_features(
            horizons_s.to(device=object_tokens.device, dtype=object_tokens.dtype)
        )
        horizon_embedding = self.horizon_projection(horizon_features)
        actions, action_mask = self._future_actions(
            object_tokens,
            horizon_count=horizon_count,
            future_ego_actions=future_ego_actions,
            future_ego_action_mask=future_ego_action_mask,
        )
        action_input = torch.cat((actions, action_mask[..., None].to(actions.dtype)), dim=-1)
        action_embedding = self.action_projection(action_input)
        box_embedding = self.box_projection(box_features)
        if not self.config.use_geometry:
            box_embedding = torch.zeros_like(box_embedding)
        tokens = (
            object_tokens[:, None, :, :]
            + box_embedding[:, None, :, :]
            + horizon_embedding[None, :, None, :]
            + action_embedding[:, :, None, :]
        )
        tokens = tokens.reshape(batch, horizon_count * object_count, dim)
        padding_mask = (~object_mask[:, None, :].bool()).expand(
            batch,
            horizon_count,
            object_count,
        )
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask.reshape(batch, horizon_count * object_count),
        )
        encoded = self.final_norm(encoded).reshape(batch, horizon_count, object_count, dim)
        predicted_latents = self.latent_head(encoded)
        geometry = self.geometry_head(encoded)
        geometry_log_variance = self.geometry_log_variance_head(encoded).clamp(-8.0, 5.0)
        return predicted_latents, geometry, geometry_log_variance, action_mask

    def _future_actions(
        self,
        reference: torch.Tensor,
        *,
        horizon_count: int,
        future_ego_actions: torch.Tensor | None,
        future_ego_action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = reference.shape[0]
        expected = (batch, horizon_count, self.config.action_dim)
        if future_ego_actions is None:
            actions = reference.new_zeros(expected)
        else:
            if future_ego_actions.shape != expected:
                msg = f"future_ego_actions must have shape {expected}."
                raise ValueError(msg)
            actions = future_ego_actions.to(device=reference.device, dtype=reference.dtype)
        if future_ego_action_mask is None:
            mask = torch.zeros(
                (batch, horizon_count),
                dtype=torch.bool,
                device=reference.device,
            )
        else:
            if future_ego_action_mask.shape != (batch, horizon_count):
                msg = "future_ego_action_mask has an incompatible shape."
                raise ValueError(msg)
            mask = future_ego_action_mask.to(device=reference.device, dtype=torch.bool)
        return torch.where(mask[..., None], actions, torch.zeros_like(actions)), mask


class ObjectCentricEventJEPA(nn.Module):
    """EMA-teacher Event-JEPA with recurrent object memory and action conditioning."""

    def __init__(self, config: ObjectJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = ObjectCentricRecurrentEncoder(config)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = MultiHorizonObjectPredictor(config)
        dim = config.embedding_dim
        self.inverse_ttc_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, len(config.risk_thresholds_s)),
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Update the target encoder by exponential moving average."""

        if not 0.0 <= momentum < 1.0:
            msg = "EMA momentum must lie in [0, 1)."
            raise ValueError(msg)
        for target, context in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(context.detach(), alpha=1.0 - momentum)
        for target, context in zip(
            self.target_encoder.buffers(),
            self.context_encoder.buffers(),
            strict=True,
        ):
            target.copy_(context)

    def train(self, mode: bool = True) -> ObjectCentricEventJEPA:
        """Keep the EMA teacher in evaluation mode."""

        super().train(mode)
        self.target_encoder.eval()
        return self

    def predict_ttc(
        self,
        context_events: torch.Tensor,
        context_boxes: torch.Tensor,
        context_object_mask: torch.Tensor,
        *,
        context_sampling_boxes: torch.Tensor | None = None,
        context_ego_actions: torch.Tensor | None = None,
        context_ego_action_mask: torch.Tensor | None = None,
    ) -> ObjectTTCOutput:
        """Predict current TTC without evaluating future JEPA targets."""

        context = self.context_encoder(
            context_events,
            context_boxes,
            context_object_mask,
            sampling_boxes_xyxy=context_sampling_boxes,
            ego_actions=context_ego_actions,
            ego_action_mask=context_ego_action_mask,
        )
        distribution = self.inverse_ttc_head(context.object_tokens)
        return ObjectTTCOutput(
            inverse_ttc_mean=distribution[..., 0],
            inverse_ttc_log_variance=distribution[..., 1].clamp(-8.0, 5.0),
            risk_logits=self.risk_head(context.object_tokens),
            object_mask=context.object_mask,
        )

    def forward(
        self,
        context_events: torch.Tensor,
        context_boxes: torch.Tensor,
        context_object_mask: torch.Tensor,
        future_events: torch.Tensor,
        future_boxes: torch.Tensor,
        future_object_mask: torch.Tensor,
        horizons_s: torch.Tensor,
        *,
        context_sampling_boxes: torch.Tensor | None = None,
        future_sampling_boxes: torch.Tensor | None = None,
        context_ego_actions: torch.Tensor | None = None,
        context_ego_action_mask: torch.Tensor | None = None,
        future_ego_actions: torch.Tensor | None = None,
        future_ego_action_mask: torch.Tensor | None = None,
    ) -> ObjectJEPAOutput:
        """Predict future object dynamics; future actions never enter the teacher."""

        context = self.context_encoder(
            context_events,
            context_boxes,
            context_object_mask,
            sampling_boxes_xyxy=context_sampling_boxes,
            ego_actions=context_ego_actions,
            ego_action_mask=context_ego_action_mask,
        )
        if torch.any(~context.object_mask.any(dim=1)):
            msg = "Every context sample must end with at least one valid tracked object."
            raise ValueError(msg)
        if future_events.ndim != 5 or future_boxes.ndim != 4:
            msg = "Future events and boxes must have shapes [B,H,C,Y,X] and [B,H,O,4]."
            raise ValueError(msg)
        batch, horizon_count = future_events.shape[:2]
        object_count = future_boxes.shape[2]
        if horizons_s.shape != (horizon_count,):
            msg = "The number of future frames must equal the number of horizons."
            raise ValueError(msg)
        if future_object_mask.shape != (batch, horizon_count, object_count):
            msg = "future_object_mask has an incompatible shape."
            raise ValueError(msg)
        if future_sampling_boxes is not None and future_sampling_boxes.shape != future_boxes.shape:
            msg = "future_sampling_boxes must match future_boxes when provided."
            raise ValueError(msg)
        if object_count != context.object_tokens.shape[1]:
            msg = "Context and future tracked-object slot counts must match."
            raise ValueError(msg)
        with torch.no_grad():
            target = self.target_encoder(
                future_events.reshape(batch * horizon_count, 1, *future_events.shape[2:]),
                future_boxes.reshape(batch * horizon_count, 1, object_count, 4),
                future_object_mask.reshape(batch * horizon_count, 1, object_count),
                sampling_boxes_xyxy=(
                    future_sampling_boxes.reshape(
                        batch * horizon_count,
                        1,
                        object_count,
                        4,
                    )
                    if future_sampling_boxes is not None
                    else None
                ),
                ego_actions=None,
                ego_action_mask=None,
            )
            target_latents = target.object_tokens.reshape(
                batch,
                horizon_count,
                object_count,
                -1,
            ).detach()
        predicted, geometry, log_variance, action_mask = self.predictor(
            context.object_tokens,
            context.current_box_features,
            context.object_mask,
            horizons_s,
            future_ego_actions=future_ego_actions,
            future_ego_action_mask=future_ego_action_mask,
        )
        ttc_distribution = self.inverse_ttc_head(context.object_tokens)
        future_mask = future_object_mask.bool() & context.object_mask[:, None, :]
        return ObjectJEPAOutput(
            predicted_latents=predicted,
            target_latents=target_latents,
            predicted_geometry=geometry,
            geometry_log_variance=log_variance,
            future_mask=future_mask,
            inverse_ttc_mean=ttc_distribution[..., 0],
            inverse_ttc_log_variance=ttc_distribution[..., 1].clamp(-8.0, 5.0),
            risk_logits=self.risk_head(context.object_tokens),
            current_object_mask=context.object_mask,
            action_conditioning_mask=action_mask,
            horizons_s=horizons_s.detach().clone(),
        )


def _fourier_horizon_features(horizons_s: torch.Tensor) -> torch.Tensor:
    frequencies = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(128.0),
            8,
            device=horizons_s.device,
            dtype=horizons_s.dtype,
        )
    )
    angles = horizons_s[:, None] * frequencies[None, :] * (2.0 * math.pi)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def geometric_dynamics_targets(
    context_boxes: torch.Tensor,
    future_boxes: torch.Tensor,
    horizons_s: torch.Tensor,
    *,
    context_depth_m: torch.Tensor | None = None,
    future_depth_m: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build multi-horizon center, scale, depth and inverse-TTC targets."""

    if context_boxes.ndim != 3 or future_boxes.ndim != 4:
        msg = "Expected context boxes [B,O,4] and future boxes [B,H,O,4]."
        raise ValueError(msg)
    incompatible_batch = future_boxes.shape[0] != context_boxes.shape[0]
    incompatible_objects = future_boxes.shape[2] != context_boxes.shape[1]
    if incompatible_batch or incompatible_objects:
        msg = "Context and future box shapes are incompatible."
        raise ValueError(msg)
    horizon_count = future_boxes.shape[1]
    if horizons_s.shape != (horizon_count,):
        msg = "horizons_s must match the future box horizon axis."
        raise ValueError(msg)
    current = normalized_box_features(context_boxes)
    future = normalized_box_features(future_boxes)
    delta_center = future[..., :2] - current[:, None, :, :2]
    log_width_ratio = future[..., 2] - current[:, None, :, 2]
    log_height_ratio = future[..., 3] - current[:, None, :, 3]
    if context_depth_m is None or future_depth_m is None:
        relative_depth_delta = torch.zeros_like(log_width_ratio)
        inverse_ttc = (1.0 - torch.exp(-log_height_ratio)) / horizons_s[None, :, None]
    else:
        if context_depth_m.shape != context_boxes.shape[:2]:
            msg = "context_depth_m must have shape [B,O]."
            raise ValueError(msg)
        if future_depth_m.shape != future_boxes.shape[:3]:
            msg = "future_depth_m must have shape [B,H,O]."
            raise ValueError(msg)
        depth = context_depth_m[:, None, :].clamp_min(1e-3)
        relative_depth_delta = (future_depth_m - depth) / depth
        inverse_ttc = -relative_depth_delta / horizons_s[None, :, None]
    return torch.stack(
        (
            delta_center[..., 0],
            delta_center[..., 1],
            log_height_ratio,
            log_width_ratio,
            relative_depth_delta,
            inverse_ttc,
        ),
        dim=-1,
    )


def object_event_jepa_loss(
    output: ObjectJEPAOutput,
    geometry_target: torch.Tensor,
    *,
    ttc_target_s: torch.Tensor | None = None,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    latent_weight: float = 1.0,
    geometry_weight: float = 1.0,
    variance_weight: float = 0.05,
    covariance_weight: float = 0.01,
    ttc_weight: float = 1.0,
    risk_weight: float = 0.25,
    straightening_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute masked latent, geometric, calibrated TTC and risk objectives."""

    if geometry_target.shape != output.predicted_geometry.shape:
        msg = "geometry_target must match predicted_geometry."
        raise ValueError(msg)
    valid = output.future_mask
    if not torch.any(valid):
        msg = "Object-JEPA loss requires at least one valid future object target."
        raise ValueError(msg)
    predicted = functional.normalize(output.predicted_latents[valid], dim=-1)
    target = functional.normalize(output.target_latents[valid].detach(), dim=-1)
    latent_loss = (1.0 - (predicted * target).sum(dim=-1)).mean()
    residual = output.predicted_geometry[valid] - geometry_target[valid]
    log_variance = output.geometry_log_variance[valid]
    geometry_nll = (0.5 * torch.exp(-log_variance) * residual.square() + 0.5 * log_variance).mean()
    variance_loss, covariance_loss = _vicreg_terms(output.predicted_latents[valid])
    straightening = _trajectory_straightening(
        output.predicted_latents,
        valid,
        output.horizons_s,
    )
    zero = latent_loss.new_zeros(())
    ttc_nll = zero
    risk_loss = zero
    if ttc_target_s is not None:
        if ttc_target_s.shape != output.inverse_ttc_mean.shape:
            msg = "ttc_target_s must have shape [B,O]."
            raise ValueError(msg)
        supervised_mask = output.current_object_mask & torch.isfinite(ttc_target_s)
        supervised_mask &= ttc_target_s.abs() >= 0.1
        if torch.any(supervised_mask):
            inverse_target = torch.reciprocal(ttc_target_s[supervised_mask])
            inverse_residual = output.inverse_ttc_mean[supervised_mask] - inverse_target
            inverse_log_variance = output.inverse_ttc_log_variance[supervised_mask]
            ttc_nll = (
                0.5 * torch.exp(-inverse_log_variance) * inverse_residual.square()
                + 0.5 * inverse_log_variance
            ).mean()
            thresholds = output.risk_logits.new_tensor(risk_thresholds_s)
            if output.risk_logits.shape[-1] != thresholds.numel():
                msg = "risk_thresholds_s does not match the risk head width."
                raise ValueError(msg)
            labels = (
                (ttc_target_s[..., None] > 0.0)
                & (ttc_target_s[..., None] <= thresholds[None, None, :])
            ).to(output.risk_logits.dtype)
            risk_loss = functional.binary_cross_entropy_with_logits(
                output.risk_logits[supervised_mask],
                labels[supervised_mask],
            )
    total = (
        latent_weight * latent_loss
        + geometry_weight * geometry_nll
        + variance_weight * variance_loss
        + covariance_weight * covariance_loss
        + ttc_weight * ttc_nll
        + risk_weight * risk_loss
        + straightening_weight * straightening
    )
    return {
        "total": total,
        "latent": latent_loss,
        "geometry_nll": geometry_nll,
        "variance": variance_loss,
        "covariance": covariance_loss,
        "ttc_inverse_nll": ttc_nll,
        "risk_bce": risk_loss,
        "straightening": straightening,
    }


def _vicreg_terms(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        msg = "VICReg values must have shape [independent_samples, dimensions]."
        raise ValueError(msg)
    if values.shape[0] < 2:
        zero = values.new_zeros(())
        return zero, zero
    centered = values - values.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = functional.relu(1.0 - standard_deviation).mean()
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / float(values.shape[1])
    return variance_loss, covariance_loss


def _trajectory_straightening(
    latents: torch.Tensor,
    valid: torch.Tensor,
    horizons_s: torch.Tensor,
) -> torch.Tensor:
    if latents.shape[1] < 3:
        return latents.new_zeros(())
    if horizons_s.shape != (latents.shape[1],) or torch.any(torch.diff(horizons_s) <= 0):
        msg = "Trajectory horizons must be strictly increasing."
        raise ValueError(msg)
    intervals = torch.diff(horizons_s).to(device=latents.device, dtype=latents.dtype)
    velocity = (latents[:, 1:] - latents[:, :-1]) / intervals[None, :, None, None]
    curvature = velocity[:, 1:] - velocity[:, :-1]
    curvature_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    if not torch.any(curvature_mask):
        return latents.new_zeros(())
    return curvature[curvature_mask].square().mean()


def inverse_ttc_distribution_to_seconds(
    inverse_mean: torch.Tensor,
    inverse_log_variance: torch.Tensor,
    *,
    minimum_inverse_magnitude: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delta-method conversion from inverse-TTC Gaussian to TTC seconds."""

    if inverse_mean.shape != inverse_log_variance.shape:
        msg = "Inverse-TTC mean and log-variance shapes must match."
        raise ValueError(msg)
    sign = torch.where(inverse_mean >= 0, 1.0, -1.0)
    safe_mean = sign * inverse_mean.abs().clamp_min(minimum_inverse_magnitude)
    ttc_mean = torch.reciprocal(safe_mean)
    inverse_std = torch.exp(0.5 * inverse_log_variance)
    ttc_std = inverse_std / safe_mean.square()
    return ttc_mean, ttc_std


__all__ = [
    "MultiHorizonObjectPredictor",
    "ObjectCentricEventJEPA",
    "ObjectCentricRecurrentEncoder",
    "ObjectEncoderOutput",
    "ObjectJEPAConfig",
    "ObjectJEPAOutput",
    "ObjectTTCOutput",
    "geometric_dynamics_targets",
    "inverse_ttc_distribution_to_seconds",
    "normalized_box_features",
    "object_event_jepa_loss",
    "roi_sample",
]

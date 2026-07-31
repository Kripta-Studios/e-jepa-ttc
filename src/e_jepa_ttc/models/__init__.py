"""Model definitions."""

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig, GarlTTCReplica
from e_jepa_ttc.models.multimodal import (
    DINOv3FeatureTeacher,
    ObjectEventRGBFusion,
    RGBRecurrentObjectEncoder,
    multimodal_ttc_loss,
)
from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC, OGEConfig
from e_jepa_ttc.models.object_jepa import (
    ObjectCentricEventJEPA,
    ObjectCentricRecurrentEncoder,
    ObjectJEPAConfig,
    geometric_dynamics_targets,
    inverse_ttc_distribution_to_seconds,
    object_event_jepa_loss,
)
from e_jepa_ttc.models.tiny_cnn import TinyCNNEncoder, TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import (
    EventTokenTransformerEncoder,
    EventTokenTransformerRegressor,
    EventTubeletTransformerEncoder,
    EventTubeletTransformerRegressor,
)


def pool_object_embeddings(
    *,
    tokens: torch.Tensor,
    bbox_masks: list[torch.Tensor],
    geometry: "TubeletTokenGeometry",
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    batch_size, token_count, embed_dim = tokens.shape

    if token_count != geometry.token_count:
        raise RuntimeError(f"Expected {geometry.token_count} tokens, got {token_count}")

    if len(bbox_masks) != batch_size:
        raise RuntimeError(
            "Number of bbox-mask groups must equal "
            "encoder batch size: "
            f"{len(bbox_masks)} versus {batch_size}"
        )

    dense = tokens.reshape(
        batch_size,
        geometry.grid_t,
        geometry.grid_h,
        geometry.grid_w,
        embed_dim,
    )

    last_temporal_tokens = dense[:, -1]

    embeddings: list[torch.Tensor] = []
    object_indices: list[tuple[int, int]] = []

    for batch_index, sample_masks in enumerate(bbox_masks):
        sample_masks = sample_masks.to(
            device=tokens.device,
            dtype=torch.bool,
            non_blocking=True,
        )

        if sample_masks.shape[0] == 0:
            raise RuntimeError(f"Sample {batch_index} has no object masks")

        if sample_masks.ndim != 3:
            raise RuntimeError("bbox masks must have shape [objects, grid_h, grid_w]")

        if tuple(sample_masks.shape[1:]) != (
            geometry.grid_h,
            geometry.grid_w,
        ):
            raise RuntimeError(f"Mask/grid mismatch: {tuple(sample_masks.shape)} versus {geometry}")

        for object_index, mask in enumerate(sample_masks):
            if not bool(mask.any()):
                raise RuntimeError("Empty bbox mask reached trainer")

            selected = last_temporal_tokens[batch_index][mask]

            if selected.numel() == 0:
                raise RuntimeError("BBox selected zero tokens")

            embeddings.append(selected.mean(dim=0))
            object_indices.append((batch_index, object_index))

    if not embeddings:
        raise RuntimeError("Batch contains no valid TTC objects")

    return torch.stack(embeddings), object_indices


@dataclass(frozen=True)
class TubeletTokenGeometry:
    grid_t: int
    grid_h: int
    grid_w: int
    kernel_t: int
    kernel_h: int
    kernel_w: int
    stride_t: int
    stride_h: int
    stride_w: int

    @property
    def token_count(self) -> int:
        return self.grid_t * self.grid_h * self.grid_w


def unwrap_tubelet_encoder(module: nn.Module) -> nn.Module:
    if hasattr(module, "event_embed") and hasattr(module, "event_bins"):
        return module

    inner = getattr(module, "encoder", None)
    if inner is not None and hasattr(inner, "event_embed"):
        return inner

    raise TypeError(f"Expected EventTubeletTransformer encoder, got {type(module)!r}")


def infer_tubelet_token_geometry(
    encoder: nn.Module,
    *,
    input_height: int,
    input_width: int,
) -> TubeletTokenGeometry:
    base = unwrap_tubelet_encoder(encoder)
    conv = base.event_embed

    kernel_t, kernel_h, kernel_w = tuple(int(v) for v in conv.kernel_size)
    stride_t, stride_h, stride_w = tuple(int(v) for v in conv.stride)

    event_bins = int(base.event_bins)

    grid_t = (event_bins - kernel_t) // stride_t + 1
    grid_h = (input_height - kernel_h) // stride_h + 1
    grid_w = (input_width - kernel_w) // stride_w + 1

    geometry = TubeletTokenGeometry(
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        kernel_t=kernel_t,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_t=stride_t,
        stride_h=stride_h,
        stride_w=stride_w,
    )

    if geometry.token_count <= 0:
        raise ValueError(f"Invalid token geometry: {geometry}")

    return geometry


MODEL_NAMES = (
    "tiny-cnn",
    "token-transformer",
    "token-transformer-large",
    "event-tubelet-transformer",
    "event-tubelet-transformer-large",
    "event-tubelet-rope-transformer",
    "event-tubelet-rope-transformer-large",
)


def build_encoder(name: str, *, in_channels: int) -> nn.Module:
    """Build an encoder by public model name."""

    if name == "tiny-cnn":
        return TinyCNNEncoder(in_channels=in_channels)
    if name == "token-transformer":
        return EventTokenTransformerEncoder(in_channels=in_channels)
    if name == "token-transformer-large":
        return EventTokenTransformerEncoder(
            in_channels=in_channels,
            embed_dim=256,
            depth=6,
            num_heads=8,
        )
    if name == "event-tubelet-transformer":
        return EventTubeletTransformerEncoder(in_channels=in_channels)
    if name == "event-tubelet-transformer-large":
        return EventTubeletTransformerEncoder(
            in_channels=in_channels,
            embed_dim=256,
            depth=8,
            num_heads=8,
        )
    if name == "event-tubelet-rope-transformer":
        return EventTubeletTransformerEncoder(
            in_channels=in_channels,
            position_encoding="rope",
        )
    if name == "event-tubelet-rope-transformer-large":
        return EventTubeletTransformerEncoder(
            in_channels=in_channels,
            embed_dim=256,
            depth=8,
            num_heads=8,
            position_encoding="rope",
        )
    msg = f"Unknown model {name!r}; expected one of {MODEL_NAMES}."
    raise ValueError(msg)


def build_regressor(name: str, *, in_channels: int) -> nn.Module:
    """Build a TTC regressor by public model name."""

    if name == "tiny-cnn":
        return TinyCNNRegressor(in_channels=in_channels)
    if name == "token-transformer":
        return EventTokenTransformerRegressor(in_channels=in_channels)
    if name == "token-transformer-large":
        return EventTokenTransformerRegressor(
            in_channels=in_channels,
            embed_dim=256,
            depth=6,
            num_heads=8,
        )
    if name == "event-tubelet-transformer":
        return EventTubeletTransformerRegressor(in_channels=in_channels)
    if name == "event-tubelet-transformer-large":
        return EventTubeletTransformerRegressor(
            in_channels=in_channels,
            embed_dim=256,
            depth=8,
            num_heads=8,
        )
    if name == "event-tubelet-rope-transformer":
        return EventTubeletTransformerRegressor(
            in_channels=in_channels,
            position_encoding="rope",
        )
    if name == "event-tubelet-rope-transformer-large":
        return EventTubeletTransformerRegressor(
            in_channels=in_channels,
            embed_dim=256,
            depth=8,
            num_heads=8,
            position_encoding="rope",
        )
    msg = f"Unknown model {name!r}; expected one of {MODEL_NAMES}."
    raise ValueError(msg)


__all__ = [
    "MODEL_NAMES",
    "DINOv3FeatureTeacher",
    "GarlTTCConfig",
    "GarlTTCReplica",
    "ObjectEventRGBFusion",
    "RGBRecurrentObjectEncoder",
    "ObjectCentricEventJEPA",
    "ObjectCentricRecurrentEncoder",
    "ObjectJEPAConfig",
    "ObjectGeometryJEPATTC",
    "OGEConfig",
    "EventTubeletTransformerEncoder",
    "EventTubeletTransformerRegressor",
    "EventTokenTransformerEncoder",
    "EventTokenTransformerRegressor",
    "TinyCNNEncoder",
    "TinyCNNRegressor",
    "build_encoder",
    "build_regressor",
    "geometric_dynamics_targets",
    "inverse_ttc_distribution_to_seconds",
    "multimodal_ttc_loss",
    "object_event_jepa_loss",
]

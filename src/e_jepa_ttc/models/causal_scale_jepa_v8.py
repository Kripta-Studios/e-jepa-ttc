"""Label-free JEPA attribution module for the V8 causal-scale endpoint encoder.

This module deliberately operates on the *exact* ``CausalScaleTTC.encoder``
module.  It has no awareness of TTC, boxes, masks, categories, identifiers, or
router outputs.  The only prediction task is to forecast an unlabeled ``t2``
event representation from two earlier event representations (``t0``, ``t1``).
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig

ViewName = Literal["dense", "global", "foreground"]
JEPA_VIEW_NAMES: tuple[ViewName, ...] = ("dense", "global", "foreground")


@dataclass(frozen=True)
class CausalScaleJEPAV8Config:
    """Fixed V8 label-free attribution controls for a causal-scale encoder."""

    ema_start: float = 0.99
    ema_end: float = 0.9999
    ema_total_updates: int = 1_000
    predictor_hidden_dim: int = 64
    collapse_std_threshold: float = 1.0e-3
    collapse_fraction_threshold: float = 0.80
    collapse_patience: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_start <= self.ema_end < 1.0:
            raise ValueError("EMA values must satisfy 0 <= start <= end < 1.")
        if self.ema_total_updates <= 0 or self.predictor_hidden_dim <= 0:
            raise ValueError("EMA updates and predictor_hidden_dim must be positive.")
        if self.collapse_std_threshold <= 0.0:
            raise ValueError("collapse_std_threshold must be positive.")
        if not 0.0 < self.collapse_fraction_threshold <= 1.0:
            raise ValueError("collapse_fraction_threshold must lie in (0, 1].")
        if self.collapse_patience <= 0:
            raise ValueError("collapse_patience must be positive.")

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable closed-objective manifest."""

        return {
            "contract": "causal_scale_jepa_v8_label_free_three_view",
            "config": asdict(self),
            "views": list(JEPA_VIEW_NAMES),
            "loss": "mean(label_free_dense, label_free_global, label_free_foreground)",
            "forbidden": [
                "ttc",
                "bbox",
                "mask",
                "category",
                "metadata",
                "geometry_loss",
                "nce",
                "vicreg",
                "router",
            ],
        }


@dataclass
class CausalScaleJEPAV8Output:
    """Three-view predictions, frozen targets, and label-free diagnostics."""

    predicted_dense_features: torch.Tensor
    target_dense_features: torch.Tensor
    predicted_global_token: torch.Tensor
    target_global_token: torch.Tensor
    predicted_foreground_logits: torch.Tensor
    target_foreground_logits: torch.Tensor
    losses: dict[str, torch.Tensor]
    health: dict[str, dict[str, float]]

    @property
    def loss(self) -> torch.Tensor:
        """Return the exact equal-weight mean over the three view losses."""

        return sum(self.losses[name] for name in JEPA_VIEW_NAMES) / 3.0


class _DenseTemporalPredictor(nn.Module):
    """Small t0/t1-only dense predictor with no target-content path."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        groups = math.gcd(hidden, 8)
        self.network = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((t0, t1), dim=1))


class _GlobalTemporalPredictor(nn.Module):
    """Small t0/t1-only global-token predictor."""

    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((t0, t1), dim=-1))


class _ForegroundTemporalPredictor(nn.Module):
    """Small t0/t1-only foreground-logit predictor."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        groups = math.gcd(hidden, 8)
        self.network = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((t0, t1), dim=1))


def _encoder_from(value: CausalScaleTTC | nn.Module) -> nn.Module:
    if isinstance(value, CausalScaleTTC):
        return value.encoder
    return value


def ordered_state_sha256(module: CausalScaleTTC | nn.Module) -> str:
    """Hash names, shapes, dtypes and raw values in state-dict insertion order."""

    encoder = _encoder_from(module)
    digest = hashlib.sha256()
    for name, value in encoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def strict_encoder_transfer(
    source: CausalScaleTTC | nn.Module,
    destination: CausalScaleTTC | nn.Module,
) -> dict[str, Any]:
    """Copy an encoder only after exact ordered key and shape validation.

    The returned source and destination hashes are intentionally identical only
    after a successful copy.  This makes transfer provenance auditable rather
    than relying on ``load_state_dict``'s implicit matching alone.
    """

    source_encoder = _encoder_from(source)
    destination_encoder = _encoder_from(destination)
    source_state = source_encoder.state_dict()
    destination_state = destination_encoder.state_dict()
    if tuple(source_state) != tuple(destination_state):
        raise ValueError("JEPA transfer requires identical ordered encoder state keys.")
    mismatched = [
        name
        for name in source_state
        if source_state[name].shape != destination_state[name].shape
        or source_state[name].dtype != destination_state[name].dtype
    ]
    if mismatched:
        raise ValueError(f"JEPA transfer has incompatible encoder tensors: {mismatched}")
    source_sha256 = ordered_state_sha256(source_encoder)
    destination_encoder.load_state_dict(source_state, strict=True)
    destination_sha256 = ordered_state_sha256(destination_encoder)
    if source_sha256 != destination_sha256:
        raise RuntimeError("JEPA strict encoder transfer failed SHA equality verification.")
    return {
        "transfer": "strict_ordered_encoder_state",
        "source_encoder_sha256": source_sha256,
        "destination_encoder_sha256": destination_sha256,
        "parameter_count": len(source_state),
    }


def freeze_all_encoder_parameters(encoder: CausalScaleTTC | nn.Module) -> tuple[str, ...]:
    """Freeze the entire endpoint encoder and return its parameter names."""

    module = _encoder_from(encoder)
    names = tuple(name for name, _ in module.named_parameters())
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return names


def d3_partial_finetune_allowlist(encoder: CausalScaleTTC | nn.Module) -> tuple[str, ...]:
    """Enable exactly the final residual block in ``features`` plus ``foreground``.

    It derives names from the actual endpoint module rather than assuming a
    residual depth.  The construction intentionally fails closed if the model
    stops exposing a final ``_ResidualBlock``.
    """

    module = _encoder_from(encoder)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    candidates = [
        (name, child)
        for name, child in module.features.named_children()
        if child.__class__.__name__ == "_ResidualBlock"
    ]
    if not candidates:
        raise RuntimeError("D3 requires a final _ResidualBlock inside encoder.features.")
    final_name, final_block = candidates[-1]
    allow_prefixes = (f"features.{final_name}.", "foreground.")
    enabled: list[str] = []
    for name, parameter in module.named_parameters():
        allowed = name.startswith(allow_prefixes)
        parameter.requires_grad_(allowed)
        if allowed:
            enabled.append(name)
    if not enabled or not any(name.startswith("foreground.") for name in enabled):
        raise RuntimeError("D3 allowlist must include foreground parameters.")
    if not any(name.startswith(f"features.{final_name}.") for name in enabled):
        raise RuntimeError("D3 allowlist must include the final residual block.")
    return tuple(enabled)


def make_d0_scratch_model(config: CausalScaleTTCConfig) -> CausalScaleTTC:
    """Create the independently initialized D0 supervised-from-scratch endpoint."""

    return CausalScaleTTC(config)


def make_d1_random_frozen_model(config: CausalScaleTTCConfig) -> CausalScaleTTC:
    """Create an independently initialized, fully frozen D1 endpoint encoder."""

    model = CausalScaleTTC(config)
    freeze_all_encoder_parameters(model)
    return model


def apply_jepa_to_experts(
    *,
    a5_model: CausalScaleTTC,
    c2f_model: CausalScaleTTC,
    a5_jepa: CausalScaleJEPAV8,
    c2f_jepa: CausalScaleJEPAV8,
    mode: Literal["frozen", "partial"] = "frozen",
) -> dict[str, dict[str, Any]]:
    """Transfer independently trained JEPA encoders to A5 and C2F only.

    The router is intentionally absent: it is a post-hoc tabular selector and
    never an encoder or a JEPA transfer target.
    """

    result = {
        "a5": strict_encoder_transfer(a5_jepa.online_encoder, a5_model),
        "c2f": strict_encoder_transfer(c2f_jepa.online_encoder, c2f_model),
    }
    for model in (a5_model, c2f_model):
        if mode == "frozen":
            freeze_all_encoder_parameters(model)
        else:
            d3_partial_finetune_allowlist(model)
    result["mode"] = {"value": mode, "router_is_encoder": False}
    return result


def _view_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Use cosine where it is informative and Smooth-L1 for scalar logits.

    A one-dimensional cosine is constant almost everywhere, so it cannot train
    the required foreground-logit view.  Smooth-L1 remains fully label-free and
    gives every foreground parameter a nonzero path to the V8 objective.
    """

    if prediction.ndim == 4:
        prediction_flat = prediction.movedim(1, -1).reshape(-1, prediction.shape[1])
        target_flat = target.movedim(1, -1).reshape(-1, target.shape[1])
    else:
        prediction_flat = prediction.reshape(-1, prediction.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
    if prediction_flat.shape[-1] == 1:
        return functional.smooth_l1_loss(prediction_flat, target_flat.detach())
    return (
        1.0
        - functional.cosine_similarity(
            functional.normalize(prediction_flat, dim=-1),
            functional.normalize(target_flat.detach(), dim=-1),
            dim=-1,
        )
    ).mean()


def view_health(embeddings: torch.Tensor, *, std_threshold: float) -> dict[str, float]:
    """Compute independent collapse statistics for a dense/global/logit view."""

    if embeddings.ndim < 2:
        raise ValueError("JEPA view embeddings require a batch and a feature dimension.")
    if embeddings.ndim == 4:
        flat = embeddings.movedim(1, -1).reshape(-1, embeddings.shape[1]).float()
    else:
        flat = embeddings.reshape(-1, embeddings.shape[-1]).float()
    std = flat.std(dim=0, unbiased=False)
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(flat.shape[0], 1)
    singular = torch.linalg.svdvals(centered)
    probability = singular / singular.sum().clamp_min(1.0e-12)
    effective_rank = torch.exp(-(probability * probability.clamp_min(1.0e-12).log()).sum())
    if flat.shape[0] < 2:
        intersample_cosine = 0.0
    else:
        normalized = functional.normalize(flat, dim=-1)
        intersample_cosine = float((normalized[:-1] * normalized[1:]).sum(dim=-1).mean())
    return {
        "mean_dimension_std": float(std.mean()),
        "collapsed_dimension_fraction": float((std < std_threshold).float().mean()),
        "effective_rank": float(effective_rank),
        "covariance_trace": float(torch.diagonal(covariance).sum()),
        "intersample_cosine": intersample_cosine,
        "mean_embedding_norm": float(flat.norm(dim=-1).mean()),
    }


class CausalScaleJEPAV8(nn.Module):
    """Three-view EMA JEPA that is structurally tied to ``CausalScaleTTC``."""

    def __init__(
        self,
        source_model: CausalScaleTTC,
        config: CausalScaleJEPAV8Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CausalScaleJEPAV8Config()
        # These must remain exact endpoint-encoder copies, not wrappers.
        self.source_encoder_sha256 = ordered_state_sha256(source_model.encoder)
        self.online_encoder = copy.deepcopy(source_model.encoder)
        self.target_encoder = copy.deepcopy(source_model.encoder)
        freeze_all_encoder_parameters(self.target_encoder)
        self.target_encoder.eval()
        self.dense_predictor = _DenseTemporalPredictor(
            source_model.config.hidden_dim, self.config.predictor_hidden_dim
        )
        self.global_predictor = _GlobalTemporalPredictor(
            source_model.config.geometry_dim, self.config.predictor_hidden_dim
        )
        self.foreground_predictor = _ForegroundTemporalPredictor(self.config.predictor_hidden_dim)
        self._assert_exact_target_copy()

    def train(self, mode: bool = True) -> CausalScaleJEPAV8:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _assert_exact_target_copy(self) -> None:
        online = self.online_encoder.state_dict()
        target = self.target_encoder.state_dict()
        if tuple(online) != tuple(target):
            raise RuntimeError("Online and target causal-scale encoder keys diverged.")
        if any(not torch.equal(online[name], target[name]) for name in online):
            raise RuntimeError("JEPA target encoder must start as an exact deep copy.")
        if ordered_state_sha256(self.online_encoder) != self.source_encoder_sha256:
            raise RuntimeError("JEPA online encoder differs from its declared source encoder.")
        if ordered_state_sha256(self.target_encoder) != self.source_encoder_sha256:
            raise RuntimeError("JEPA target encoder differs from its declared source encoder.")
        if any(parameter.requires_grad for parameter in self.target_encoder.parameters()):
            raise RuntimeError("JEPA target encoder must be frozen.")

    def ema_momentum(self, update_index: int, *, total_updates: int | None = None) -> float:
        """Return the cosine-scheduled V8 EMA momentum from 0.99 to 0.9999."""

        total = self.config.ema_total_updates if total_updates is None else total_updates
        if update_index < 0 or total <= 0:
            raise ValueError("update_index must be non-negative and total_updates positive.")
        fraction = min(update_index, max(total - 1, 0)) / max(total - 1, 1)
        return self.config.ema_start + (self.config.ema_end - self.config.ema_start) * (
            0.5 - 0.5 * math.cos(math.pi * fraction)
        )

    @torch.no_grad()
    def update_target_ema(self, update_index: int, *, total_updates: int | None = None) -> float:
        """Update the frozen target with name-aligned EMA arithmetic."""

        momentum = self.ema_momentum(update_index, total_updates=total_updates)
        online = self.online_encoder.state_dict()
        target = self.target_encoder.state_dict()
        if tuple(online) != tuple(target):
            raise RuntimeError("Online and target causal-scale encoder states diverged.")
        for name, target_value in target.items():
            source_value = online[name]
            if torch.is_floating_point(target_value):
                target_value.mul_(momentum).add_(source_value, alpha=1.0 - momentum)
            else:
                target_value.copy_(source_value)
        freeze_all_encoder_parameters(self.target_encoder)
        self.target_encoder.eval()
        return momentum

    @staticmethod
    def _validate_inputs(t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> None:
        if any(value.ndim != 4 for value in (t0, t1, t2)):
            raise ValueError("JEPA t0, t1, and t2 must each have shape [B,C,H,W].")
        if t0.shape != t1.shape or t0.shape != t2.shape:
            raise ValueError("JEPA t0, t1, and t2 must have identical shapes.")

    def forward(
        self, t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor
    ) -> CausalScaleJEPAV8Output:
        """Forecast unlabeled t2 encoder views from t0/t1 only.

        ``t2`` is exclusively evaluated through the frozen EMA target under
        ``no_grad``.  Consequently neither this signature nor the predictor has
        a route for supervised target labels or future target content.
        """

        self._validate_inputs(t0, t1, t2)
        logits0, token0, dense0 = self.online_encoder(t0, return_dense_features=True)
        logits1, token1, dense1 = self.online_encoder(t1, return_dense_features=True)
        if dense0 is None or dense1 is None:
            raise RuntimeError("CausalScaleTTC encoder did not return dense features.")
        with torch.no_grad():
            target_logits, target_token, target_dense = self.target_encoder(
                t2, return_dense_features=True
            )
        if target_dense is None:
            raise RuntimeError("Frozen causal-scale target did not return dense features.")
        predicted_dense = self.dense_predictor(dense0, dense1)
        predicted_global = self.global_predictor(token0, token1)
        predicted_foreground = self.foreground_predictor(logits0, logits1)
        losses = {
            "dense": _view_prediction_loss(predicted_dense, target_dense),
            "global": _view_prediction_loss(predicted_global, target_token),
            "foreground": _view_prediction_loss(predicted_foreground, target_logits),
        }
        health = {
            "dense": view_health(
                predicted_dense.detach(), std_threshold=self.config.collapse_std_threshold
            ),
            "global": view_health(
                predicted_global.detach(), std_threshold=self.config.collapse_std_threshold
            ),
            "foreground": view_health(
                predicted_foreground.detach(), std_threshold=self.config.collapse_std_threshold
            ),
        }
        return CausalScaleJEPAV8Output(
            predicted_dense_features=predicted_dense,
            target_dense_features=target_dense.detach(),
            predicted_global_token=predicted_global,
            target_global_token=target_token.detach(),
            predicted_foreground_logits=predicted_foreground,
            target_foreground_logits=target_logits.detach(),
            losses=losses,
            health=health,
        )

    def encoder_manifest(self) -> dict[str, Any]:
        """Return exact online/target hashes and the closed objective manifest."""

        manifest = self.config.manifest()
        manifest.update(
            {
                "online_encoder_sha256": ordered_state_sha256(self.online_encoder),
                "target_encoder_sha256": ordered_state_sha256(self.target_encoder),
                "source_encoder_sha256": self.source_encoder_sha256,
                "target_frozen": True,
                "target_eval": not self.target_encoder.training,
            }
        )
        return manifest


__all__ = [
    "CausalScaleJEPAV8",
    "CausalScaleJEPAV8Config",
    "CausalScaleJEPAV8Output",
    "JEPA_VIEW_NAMES",
    "apply_jepa_to_experts",
    "d3_partial_finetune_allowlist",
    "freeze_all_encoder_parameters",
    "make_d0_scratch_model",
    "make_d1_random_frozen_model",
    "ordered_state_sha256",
    "strict_encoder_transfer",
    "view_health",
]

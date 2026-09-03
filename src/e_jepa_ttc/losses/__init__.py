"""Losses used by the auditable JEPA/TTC experiments."""

from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLoss,
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.losses.collision_clock import (
    UNIFORM_PHASE_REDUCTION,
    WEIGHTED_PHASE_REDUCTION,
    normalized_weighted_absolute_phase_error,
    uniform_benchmark_phase_loss,
)
from e_jepa_ttc.losses.garl_ttc import signed_log_ttc_loss
from e_jepa_ttc.losses.jepa_dense import cosine_prediction_loss, variance_covariance_loss

__all__ = [
    "CausalScaleTTCLoss",
    "CausalScaleTTCLossConfig",
    "UNIFORM_PHASE_REDUCTION",
    "WEIGHTED_PHASE_REDUCTION",
    "causal_scale_ttc_loss",
    "cosine_prediction_loss",
    "normalized_weighted_absolute_phase_error",
    "signed_log_ttc_loss",
    "uniform_benchmark_phase_loss",
    "variance_covariance_loss",
]

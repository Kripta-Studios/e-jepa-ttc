"""Losses used by the auditable JEPA/TTC experiments."""

from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLoss,
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.losses.garl_ttc import signed_log_ttc_loss
from e_jepa_ttc.losses.jepa_dense import cosine_prediction_loss, variance_covariance_loss

__all__ = [
    "CausalScaleTTCLoss",
    "CausalScaleTTCLossConfig",
    "causal_scale_ttc_loss",
    "cosine_prediction_loss",
    "signed_log_ttc_loss",
    "variance_covariance_loss",
]

"""Predictive and anti-collapse loss entry points."""

from e_jepa_ttc.losses.jepa_dense import cosine_prediction_loss, variance_covariance_loss

__all__ = ["cosine_prediction_loss", "variance_covariance_loss"]

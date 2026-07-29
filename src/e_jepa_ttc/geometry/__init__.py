"""Differentiable TTC geometry used by the EvTTC architecture gates."""

from e_jepa_ttc.geometry.affine_expansion_ttc import affine_expansion_inverse_ttc
from e_jepa_ttc.geometry.area_rate_ttc import area_rate_inverse_ttc
from e_jepa_ttc.geometry.event_contrast import event_contrast_inverse_ttc
from e_jepa_ttc.geometry.geometry_confidence import geometry_track_confidence
from e_jepa_ttc.geometry.height_ratio_ttc import height_ratio_inverse_ttc
from e_jepa_ttc.geometry.weighted_solver import weighted_inverse_ttc

__all__ = [
    "affine_expansion_inverse_ttc",
    "area_rate_inverse_ttc",
    "event_contrast_inverse_ttc",
    "geometry_track_confidence",
    "height_ratio_inverse_ttc",
    "weighted_inverse_ttc",
]

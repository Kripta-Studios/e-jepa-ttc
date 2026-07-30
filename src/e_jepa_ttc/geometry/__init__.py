"""Differentiable TTC geometry used by the EvTTC architecture gates."""

from e_jepa_ttc.geometry.affine_expansion_ttc import affine_expansion_inverse_ttc
from e_jepa_ttc.geometry.area_rate_ttc import area_rate_inverse_ttc
from e_jepa_ttc.geometry.cmax import maximize_radial_event_contrast
from e_jepa_ttc.geometry.event_contrast import event_contrast_inverse_ttc
from e_jepa_ttc.geometry.geometry_confidence import geometry_track_confidence
from e_jepa_ttc.geometry.height_ratio_ttc import height_ratio_inverse_ttc
from e_jepa_ttc.geometry.rgb_foe import (
    affine_foe_xy,
    farneback_affine_ttc,
    fit_affine_flow,
    ttc_from_affine_fit,
)
from e_jepa_ttc.geometry.strttc import (
    construct_strttc_system,
    inverse_ttc_at_endpoint,
    refine_strttc_on_time_surface,
    robust_linear_strttc,
    warp_strttc_events,
)
from e_jepa_ttc.geometry.weighted_solver import weighted_inverse_ttc

__all__ = [
    "affine_expansion_inverse_ttc",
    "area_rate_inverse_ttc",
    "maximize_radial_event_contrast",
    "event_contrast_inverse_ttc",
    "geometry_track_confidence",
    "height_ratio_inverse_ttc",
    "affine_foe_xy",
    "farneback_affine_ttc",
    "fit_affine_flow",
    "ttc_from_affine_fit",
    "construct_strttc_system",
    "inverse_ttc_at_endpoint",
    "refine_strttc_on_time_surface",
    "robust_linear_strttc",
    "warp_strttc_events",
    "weighted_inverse_ttc",
]

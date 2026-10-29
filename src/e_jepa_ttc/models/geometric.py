"""Analytical geometry baselines exposed under the generic model namespace."""

from e_jepa_ttc.geometry.affine_expansion_ttc import affine_expansion_inverse_ttc
from e_jepa_ttc.geometry.area_rate_ttc import area_rate_inverse_ttc
from e_jepa_ttc.geometry.height_ratio_ttc import height_ratio_inverse_ttc

__all__ = [
    "affine_expansion_inverse_ttc",
    "area_rate_inverse_ttc",
    "height_ratio_inverse_ttc",
]

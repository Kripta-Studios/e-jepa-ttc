"""Canonical names for the audited official Garl-TTC preprocessing."""

from __future__ import annotations

from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_square_box,
    official_timevolume_roi_np,
)

GARL_TIMEVOLUME20_ID = "garl_timevolume20_v1"


garl_timevolume20_v1 = official_timevolume_roi_np


__all__ = [
    "GARL_TIMEVOLUME20_ID",
    "garl_timevolume20_v1",
    "official_resize_feature",
    "official_square_box",
    "official_timevolume_roi_np",
]

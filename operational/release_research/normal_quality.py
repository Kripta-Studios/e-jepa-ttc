"""Target-free time-plane diagnostics in the release's original sensor ROI."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from e_jepa_ttc.models.normal_flow_routing_v10 import fit_time_plane
from operational.release_research.physics import affine_design, estimate_expansion, transport

PHYSICAL_FIELDS = (
    "null_leakage",
    "q_sensitivity",
    "normalized_residual",
    "temporal_innovation",
    "foreground_background_disagreement",
    "independent_patch_fraction",
)
SUPPORT_FIELDS = (
    "log_event_count",
    "plane_fraction",
    "pair_support",
    "expert_disagreement",
    "history_count",
    "roi_retained_fraction",
)


def make_surface(
    events: np.ndarray, roi: np.ndarray, resolution: tuple[int, int], start_us: int, end_us: int
) -> tuple[np.ndarray, tuple[int, int], int]:
    """Construct separate latest-event surfaces without resampling or rank subsampling."""
    if events.ndim != 2 or events.shape[1] != 4 or events.dtype.kind not in "iu":
        raise ValueError("Need integer x,y,t_us,p events[N,4]")
    if roi.shape != (4,) or not np.isfinite(roi).all() or end_us <= start_us:
        raise ValueError("Invalid sensor ROI/time bounds")
    if min(resolution) <= 0 or roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("Invalid sensor geometry")
    if len(events) and (
        np.any(np.diff(events[:, 2]) < 0)
        or events[:, 2].min() < start_us
        or events[:, 2].max() >= end_us
        or not np.isin(events[:, 3], [0, 1]).all()
    ):
        raise ValueError("Event ordering, polarity or half-open causal bounds violated")
    if len(events) and (
        np.any(events[:, :2] < 0)
        or np.any(events[:, 0] >= resolution[0])
        or np.any(events[:, 1] >= resolution[1])
    ):
        raise ValueError("Sensor event outside declared resolution")
    left, top = max(0, int(np.ceil(roi[0]))), max(0, int(np.ceil(roi[1])))
    right, bottom = (
        min(resolution[0], int(np.ceil(roi[2]))),
        min(resolution[1], int(np.ceil(roi[3]))),
    )
    shape = (2, max(0, bottom - top), max(0, right - left))
    surface = np.full(shape, np.iinfo(np.int64).min, np.int64)
    inside = (
        (events[:, 0] >= left)
        & (events[:, 0] < right)
        & (events[:, 1] >= top)
        & (events[:, 1] < bottom)
    )
    selected = events[inside]
    if len(selected):
        np.maximum.at(
            surface, (selected[:, 3], selected[:, 1] - top, selected[:, 0] - left), selected[:, 2]
        )
    return surface, (left, top), int(inside.sum())


def fit_surface(
    surface: np.ndarray,
    origin: tuple[int, int],
    roi: np.ndarray,
    object_box: np.ndarray,
    endpoint_us: int,
    *,
    age_us: int = 10000,
) -> dict:
    """Fit the fixed 16x16 grid of 9x9 local planes used in the historical diagnostic.

    q uses c in row(H), rather than rejecting every rank-deficient affine fit.
    The object-box comparison is a mixture proxy, not measured true visibility.
    Patch independence is a nonoverlap proxy, not a calibrated noise model.
    """
    if surface.ndim != 3 or surface.shape[0] != 2 or surface.dtype != np.int64:
        raise ValueError("Expected integer two-polarity timestamp surface")
    if roi.shape != (4,) or object_box.shape != (4,) or age_us <= 0:
        raise ValueError("Invalid plane geometry")
    size = roi[2] - roi[0]
    if size <= 0 or not np.isclose(size, roi[3] - roi[1], atol=1e-6):
        raise ValueError("Release uses a common isotropic square")
    left, top = origin
    height, width = surface.shape[1:]
    centers = sorted(
        {
            (
                int(np.floor(roi[0] + (ix + 0.5) * size / 16)),
                int(np.floor(roi[1] + (iy + 0.5) * size / 16)),
            )
            for iy in range(16)
            for ix in range(16)
        }
    )
    planes, selected = [], []
    for polarity in (0, 1):
        for cx, cy in centers:
            xs = np.arange(max(left, cx - 4), min(left + width, cx + 5))
            ys = np.arange(max(top, cy - 4), min(top + height, cy + 5))
            if len(xs) == 0 or len(ys) == 0:
                continue
            xx, yy = np.meshgrid(xs, ys)
            ticks = surface[polarity, yy - top, xx - left].ravel()
            usable = (ticks >= endpoint_us - age_us) & (ticks < endpoint_us)
            if usable.sum() < 9:
                continue
            xy = (
                np.column_stack((xx.ravel()[usable], yy.ravel()[usable])) - (roi[:2] + roi[2:]) / 2
            ) / size
            plane = fit_time_plane(
                xy, (ticks[usable] - np.int64(endpoint_us)).astype(np.float64) * 1e-6
            )
            if not plane.valid:
                continue
            location = (np.array([cx, cy]) - (roi[:2] + roi[2:]) / 2) / size
            foreground = bool(
                object_box[0] <= cx < object_box[2] and object_box[1] <= cy < object_box[3]
            )
            independent = all(
                pol != polarity or abs(cx - x) >= 9 or abs(cy - y) >= 9 for pol, x, y in selected
            )
            if independent:
                selected.append((polarity, cx, cy))
            planes.append(
                [
                    *location,
                    *plane.normal,
                    plane.speed,
                    polarity,
                    plane.r_squared,
                    plane.residual_fraction,
                    int(foreground),
                    int(independent),
                ]
            )
    data = np.asarray(planes, np.float64).reshape(-1, 10)

    def solve(values: np.ndarray) -> dict | None:
        if len(values) == 0:
            return None
        result = asdict(
            estimate_expansion(affine_design(values[:, :2], values[:, 2:4]), values[:, 4])
        )
        # Unit-free robust residual, not an estimated standard error of q.
        result["normalized_residual"] = result["median_absolute_residual"] / max(
            float(np.median(np.abs(values[:, 4]))), 1e-8
        )
        return result

    whole = solve(data)
    foreground = solve(data[data[:, 8] == 1])
    background = solve(data[data[:, 8] == 0])
    difference = None
    if foreground and background and foreground["q"] is not None and background["q"] is not None:
        difference = abs(foreground["q"] - background["q"])
    return {
        "endpoint_us": endpoint_us,
        "candidate_planes": 2 * len(centers),
        "planes": data.tolist(),
        "accepted_planes": len(data),
        "all": whole,
        "foreground": foreground,
        "background": background,
        "foreground_background_disagreement": difference,
        "independent_patch_fraction": len(selected) / len(data) if len(data) else None,
        "plane_fraction": len(data) / (2 * len(centers)),
    }


def physical_vector(earlier: dict, current: dict) -> list[float | None]:
    """Six prespecified indicators; unavailable evidence remains missing, never certain."""
    fit = current["all"]
    innovation = None
    if earlier["all"] and fit and earlier["all"]["q"] is not None and fit["q"] is not None:
        dt = (current["endpoint_us"] - earlier["endpoint_us"]) * 1e-6
        try:
            predicted, _ = transport(earlier["all"]["q"], dt)
        except ValueError:
            pass
        else:
            innovation = abs(fit["q"] - predicted)
    return [
        fit["null_leakage"] if fit else None,
        fit["sensitivity"] if fit else None,
        fit["normalized_residual"] if fit else None,
        innovation,
        current["foreground_background_disagreement"],
        current["independent_patch_fraction"],
    ]

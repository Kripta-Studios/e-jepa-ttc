"""Input-only retrospective D1 indices under the supplied-current-ROI amendment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes, shifted_precontext_window


def expansion_context_arrays(
    rows: Sequence[Mapping[str, Any]],
    exposures: Mapping[str, Mapping[str, Any]],
    bounds: Mapping[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Ignore target fields and shift the frozen three windows, never labelled query history.

    Bounds are actual first/last raw event timestamps. Their last timestamp is
    treated conservatively like the existing D0 index: window ends must be no
    later than it. Producer assignment and extraction authorization are separate.
    """
    ordered = sorted(rows, key=lambda row: str(row["sample_token"]))
    tokens = [str(row["sample_token"]) for row in ordered]
    if not tokens or len(set(tokens)) != len(tokens) or set(tokens) != set(exposures):
        raise ValueError("missing, duplicate or foreign exposure identity")
    count = len(tokens)
    arrays = {
        "tokens": np.asarray(tokens),
        "sequences": np.asarray([str(row["sequence_id"]) for row in ordered]),
        "base_windows_us": np.empty((count, 3, 2), dtype=np.int64),
        "square_xyxy": np.empty((count, 4), dtype=np.float64),
        "anchor_us": np.empty(count, dtype=np.int64),
        "roi_available_us": np.empty(count, dtype=np.int64),
        "lag_us": np.arange(15, -1, -1, dtype=np.int64) * 50_000,
        "valid": np.empty((count, 16), dtype=bool),
    }
    for index, row in enumerate(ordered):
        token, sequence = tokens[index], str(row["sequence_id"])
        windows = np.stack(row["event_windows_us"])
        if windows.shape != (2, 2) or windows.dtype.kind not in "iu":
            raise ValueError("two integer-microsecond endpoint windows required")
        if np.any(windows[:, 1] <= windows[:, 0]) or windows[0, 1] >= windows[1, 1]:
            raise ValueError("empty or unordered producer windows")
        first = tuple(int(value) for value in windows[0])
        base = np.asarray([shifted_precontext_window(first, shift_s=0.1), *windows], np.int64)
        exposure = exposures[token]
        anchor, available = int(exposure["anchor_us"]), int(exposure["selected_exposure_end_us"])
        if anchor != int(windows[1, 1]) or available < anchor:
            raise ValueError("exposure clock or dependency cutoff mismatch")
        start, end = bounds[sequence]
        if start > end or base.min() < start or base.max() > end:
            raise ValueError(f"current query lacks complete source support: {token}")
        arrays["base_windows_us"][index] = base
        arrays["square_xyxy"][index] = common_square_from_boxes(row["boxes_xyxy"], (0, 1))
        arrays["anchor_us"][index], arrays["roi_available_us"][index] = anchor, available
        arrays["valid"][index] = (base.min() - arrays["lag_us"] >= start) & (
            base.max() - arrays["lag_us"] <= end
        )
    return arrays

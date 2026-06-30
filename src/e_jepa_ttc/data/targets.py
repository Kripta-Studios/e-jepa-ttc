"""TTC target parsing and interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np

from e_jepa_ttc.data.validation import validate_ttc_table


class TTCTable(TypedDict):
    """Parsed whitespace-separated EvTTC TTC table."""

    frame_id: np.ndarray
    timestamp_s: np.ndarray
    distance: np.ndarray
    relative_speed: np.ndarray
    ttc_s: np.ndarray


def load_ttc_csv(path: str | Path) -> TTCTable:
    """Load local EvTTC `ttc.csv`.

    The observed local files are whitespace-separated and headerless:
    `frame_id timestamp_s distance relative_speed ttc_s`.
    """

    input_path = Path(path)
    data = np.genfromtxt(input_path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        msg = f"Expected at least five columns in {input_path}, got shape {data.shape}."
        raise ValueError(msg)

    table: TTCTable = {
        "frame_id": data[:, 0].astype(np.int64),
        "timestamp_s": data[:, 1].astype(np.float64),
        "distance": data[:, 2].astype(np.float64),
        "relative_speed": data[:, 3].astype(np.float64),
        "ttc_s": data[:, 4].astype(np.float64),
    }
    validate_ttc_table(table["timestamp_s"], table["ttc_s"])
    return table


def interpolate_ttc_seconds(table: TTCTable, timestamp_us: int) -> float | None:
    """Linearly interpolate TTC at a reference timestamp in microseconds."""

    timestamp_s = float(timestamp_us) / 1_000_000.0
    x = table["timestamp_s"]
    y = table["ttc_s"]
    if timestamp_s < float(x[0]) or timestamp_s > float(x[-1]):
        return None
    return float(np.interp(timestamp_s, x, y))

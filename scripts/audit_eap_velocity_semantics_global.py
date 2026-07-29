from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EAP_ROOT = Path(r"E:\eAP_dataset")
LABEL_ROOT = EAP_ROOT / "data" / "train"
METADATA_PATH = EAP_ROOT / "data" / "train.parquet"

OUTPUT_DIR = Path("artifacts/audit/eap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "eap_velocity_semantics_global.json"
OUTPUT_CSV = OUTPUT_DIR / "eap_track_velocity_comparison.csv"


def to_vector(value, expected: int) -> np.ndarray | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float64).reshape(-1)

    if len(array) < expected:
        return None

    array = array[:expected]

    if not np.all(np.isfinite(array)):
        return None

    return array


def rolling_median(values: np.ndarray, window: int =  nine) -> np.ndarray:
    raise RuntimeError("placeholder")

"""Closed-form P geometry calibration with paired height-change supervision."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from operational.range_transfer.common import freeze, read, sha

from .geometry_model import CALIBRATION
from .precision import Vectors
from .run import RUN

TEMPORAL_CALIBRATION = RUN / "temporal_geometry/P_HEIGHT_CALIBRATION.json"


def calibrate() -> dict:
    """Keep P normalization fixed and solve the prospectively registered objective."""
    started = time.perf_counter()
    control = read(CALIBRATION)
    data = Vectors("P")
    v = data.values
    z = (v["geometry_tokens"].astype(np.float64) - control["mean"]) / control["scale"]
    valid = v["target_valid"].astype(bool)
    x = z[valid]
    height = v["target_height_normalized"].astype(np.float64)
    original = v["visible_height"].astype(np.float64)
    y = np.log(height[valid]) - np.log(original[valid])
    pairs = valid[:, -1] & valid[:, -2]
    d = z[pairs, -1] - z[pairs, -2]
    target = np.log(height[pairs, -1] / height[pairs, -2]) - v["analytic"][pairs, -1]
    variance_y, variance_d = float(y.var()), float(target.var())
    if len(y) != 107780 or int(pairs.sum()) != 53890 or min(variance_y, variance_d) <= 0:
        raise ValueError("Unexpected geometry supervision support")
    factor = len(y) / len(d)
    xx, dd = x.T @ x, d.T @ d
    xy = x.T @ (y - control["intercept"])
    dy = d.T @ target
    matrix = xx / variance_y + factor * dd / variance_d + np.eye(128)
    rhs = xy / variance_y + factor * dy / variance_d
    beta = np.linalg.solve(matrix, rhs)
    result = {
        "role": "P",
        "observations": len(y),
        "pairs": len(d),
        "closed_form_fits_new": 1,
        "optimizer_updates": 0,
        "ridge_lambda": 1.0,
        "variance_height_target": variance_y,
        "variance_temporal_target": variance_d,
        "mean": control["mean"],
        "scale": control["scale"],
        "intercept": control["intercept"],
        "coefficient": beta.tolist(),
        "delta_clip": [-2.0, 2.0],
        "unclipped_training_height_rmse": float(
            np.sqrt(np.mean((x @ beta + control["intercept"] - y) ** 2))
        ),
        "unclipped_training_temporal_rmse": float(np.sqrt(np.mean((d @ beta - target) ** 2))),
        "control_temporal_rmse": float(
            np.sqrt(np.mean((d @ np.array(control["coefficient"]) - target) ** 2))
        ),
        "sources": {
            "control_sha256": sha(CALIBRATION),
            "code": sha(Path(__file__)),
            "cache": data.identity,
        },
    }
    freeze(TEMPORAL_CALIBRATION, result)
    cost = TEMPORAL_CALIBRATION.with_name("FIT_COST.json")
    if not cost.exists():
        freeze(cost, {"cpu_wall_s": time.perf_counter() - started, "scope": "P temporal ridge fit"})
    return result


if __name__ == "__main__":
    result = calibrate()
    print({key: value for key, value in result.items() if "rmse" in key})

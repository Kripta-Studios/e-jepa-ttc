"""Training-P annotation consistency audit, not an admissible oracle predictor."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import OLD, RUN


def main() -> None:
    folder = OLD / "vector_cache/P"
    count = read(folder / "COMPLETE.json")["rows"]
    parts = []
    for index in range(math.ceil(count / 128)):
        path = folder / f"part_{index:06d}.npz"
        if sha(path) != read(path.with_suffix(".json"))["sha256"]:
            raise ValueError("P vector shard changed")
        with np.load(path, allow_pickle=False) as a:
            height = a["target_height_normalized"].astype(np.float64)
            width = a["target_width_normalized"].astype(np.float64)
            valid = (
                a["target_valid"][:, -2:].all(1)
                & (height[:, -2:] > 0).all(1)
                & (width[:, -2:] > 0).all(1)
            )
            target = a["target_ttc"].astype(np.float64)
            dt = a["delta"][:, -1].astype(np.float64)
            valid = valid & ((target > 0) | (target < -dt))
            required = np.full(len(target), np.nan)
            rh = np.full(len(target), np.nan)
            rw = np.full(len(target), np.nan)
            required[valid] = np.log1p(dt[valid] / target[valid])
            rh[valid] = np.log(height[valid, -1] / height[valid, -2])
            rw[valid] = np.log(width[valid, -1] / width[valid, -2])
            parts.append(
                pd.DataFrame(
                    {
                        "sample_token": a["sample_token"],
                        "target_ttc": target,
                        "valid": valid,
                        "required_current_ratio": required,
                        "bbox_height_ratio": rh,
                        "bbox_width_ratio": rw,
                        "learned_analytic_ratio": a["analytic"][:, -1],
                        "height_prediction": a["visible_height"][:, -1],
                        "height_target": height[:, -1],
                    }
                )
            )
    data = pd.concat(parts, ignore_index=True)
    rows = []
    bands = {
        "0-1": (data.target_ttc > 0) & (data.target_ttc <= 1),
        "1-3": (data.target_ttc > 1) & (data.target_ttc <= 3),
        "3-6": (data.target_ttc > 3) & (data.target_ttc <= 6),
        "6+": data.target_ttc > 6,
        "negative": data.target_ttc < 0,
    }
    for band, mask in bands.items():
        part = data.loc[mask & data.valid]
        for source in ("bbox_height_ratio", "bbox_width_ratio", "learned_analytic_ratio"):
            gap = (part.required_current_ratio - part[source]).abs()
            rows.append(
                {
                    "band": band,
                    "source": source,
                    "population": int(mask.sum()),
                    "valid_n": len(part),
                    "median_required_ratio": float(part.required_current_ratio.median()),
                    "median_ratio": float(part[source].median()),
                    "median_abs_gap": float(gap.median()),
                    "p95_abs_gap": float(gap.quantile(0.95)),
                    "outside_C05_current_pair": int((gap > 0.05).sum()),
                    "outside_C20_current_pair": int((gap > 0.2).sum()),
                }
            )
    out = RUN / "diagnostics"
    data.to_csv(out / "P_GEOMETRY_CONSISTENCY_ROWS.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "P_GEOMETRY_CONSISTENCY.csv", index=False)
    write(
        out / "GEOMETRY_CONSISTENCY_SCOPE.json",
        {
            "role": "P",
            "optimizer_updates": 0,
            "rows": len(data),
            "scope": (
                "current pair log-ratio only; annotations are training targets, "
                "not inference inputs"
            ),
            "not_model_performance": True,
            "not_system_reach_bound": True,
            "target_annotation_visibility_and_motion_assumptions_apply": True,
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

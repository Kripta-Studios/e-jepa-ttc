"""Paired acquisition-group bootstrap for frozen cohort-complete predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.common import freeze, sha, write
from operational.rgb_port.evaluate import BUCKETS, WEIGHTS, _bucket_mask


def compare(reference: Path, candidate: Path, output: Path) -> dict:
    """Resample whole groups jointly across bands, retaining missing-band replicates."""
    a, b = pd.read_parquet(reference), pd.read_parquet(candidate)
    for column in ("sample_token", "target_ttc", "group_id"):
        if not np.array_equal(a[column], b[column]):
            raise ValueError("Bootstrap requires identical ordered cohorts")
    binding = {
        "reference_sha256": sha(reference),
        "candidate_sha256": sha(candidate),
        "seed": 12345,
        "replicates": 2000,
        "unit": "acquisition_group",
        "scope": "paired group-macro bucket MiD; descriptive development interval",
    }
    freeze(output / "BINDING.json", binding)
    arrays = []
    for frame in (a, b):
        target, pred = frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy()
        if not (np.isfinite(pred) & ((pred < 0) | (pred > 0.1))).all():
            result = {"status": "INVALID_PREDICTION_COVERAGE", "binding": binding}
            write(output / "SUMMARY.json", result)
            return result
        mid = 10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / target))
        by_group = []
        for group in sorted(frame.group_id.unique()):
            row = {}
            for name, lower, upper in BUCKETS:
                select = (frame.group_id.to_numpy() == group) & _bucket_mask(target, lower, upper)
                row[name] = float(mid[select].mean()) if select.any() else np.nan
            by_group.append(row)
        arrays.append(np.array([[row[name] for name in WEIGHTS] for row in by_group]))
    rng = np.random.default_rng(binding["seed"])
    draws = rng.integers(len(arrays[0]), size=(binding["replicates"], len(arrays[0])))
    means = []
    for array in arrays:
        selected = array[draws]
        count = np.isfinite(selected).sum(axis=1)
        total = np.nansum(selected, axis=1)
        means.append(np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0))
    delta = means[1] - means[0]
    table = pd.DataFrame(delta, columns=pd.Index([f"delta_{name}" for name in WEIGHTS]))
    table["delta_overall"] = delta @ np.array(list(WEIGHTS.values()))
    table.to_csv(output / "GROUP_BOOTSTRAP.csv", index=False)
    summaries = {}
    for column in table:
        values = table[column].to_numpy()
        valid = np.isfinite(values)
        summaries[column] = {
            "defined_replicates": int(valid.sum()),
            "undefined_replicates": int((~valid).sum()),
            "ci95_if_all_defined": np.quantile(values, [0.025, 0.975]).tolist()
            if valid.all()
            else None,
            "conditional_ci95_among_defined": np.quantile(values[valid], [0.025, 0.975]).tolist()
            if valid.any()
            else None,
        }
    result = {
        "status": "COMPLETE",
        "binding": binding,
        "intervals": summaries,
        "note": "Missing-band resamples remain undefined; conditional intervals are labeled.",
    }
    write(output / "SUMMARY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(compare(args.reference, args.candidate, args.output))

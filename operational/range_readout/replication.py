"""Aggregate seed losses per query; no posthoc ensemble of TTC predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import gate
from operational.rgb_port.evaluate import BUCKETS, WEIGHTS

from .common import RUN, freeze, write


def aggregate(frames: list[pd.DataFrame]) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["sample_token", "group_id", "target_ttc"]
    first = frames[0]
    if len(frames) != 3 or any(not first[keys].equals(f[keys]) for f in frames[1:]):
        raise ValueError("Three paired seed populations required")
    target = first.target_ttc.to_numpy()
    groups = first.group_id.to_numpy()
    # Restore native FP32 predictions before the scorer's FP64 arithmetic.
    pred = np.stack([f.prediction_ttc.to_numpy(np.float32).astype(np.float64) for f in frames])
    if not np.isfinite(pred).all() or not ((pred < 0) | (pred > 0.1)).all():
        raise ValueError("Replica coverage incomplete")
    mid = (10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / target)[None, :])).mean(0)
    ae = np.abs(pred - target[None, :]).mean(0)
    positive_over = np.maximum(pred - target[None, :], 0).mean(0)
    sign = (np.signbit(pred) != np.signbit(target)[None, :]).mean(0)
    warning = ((pred > 0.1) & (pred <= 1)).mean(0)
    c = (target > 0.1) & (target <= 3)
    imminent = (target > 0.1) & (target <= 1)
    safe = (target < 0) | (target > 1)
    bins = {}
    for name, lower, upper in BUCKETS:
        mask = (target > lower) & (target <= upper)
        means = [float(mid[mask & (groups == g)].mean()) for g in sorted(set(groups[mask]))]
        bins[name] = {
            "count": int(mask.sum()),
            "group_macro_mid": float(np.mean(means)) if means else None,
        }
    complete = all(item["group_macro_mid"] is not None for item in bins.values())
    overall = (
        sum(WEIGHTS[name] * bins[name]["group_macro_mid"] for name in WEIGHTS) if complete else None
    )
    result = {
        "schema": "v15_three_seed_mean_query_losses_v1",
        "overall_MiD": overall,
        "mean_MiD": float(mid.mean()),
        "complete_prediction_and_mid_coverage": True,
        "bins": bins,
        "group_crucial_mid": {
            str(g): float(mid[c & (groups == g)].mean()) for g in sorted(set(groups[c]))
        },
        "diagnostics": {"mae_s": float(ae.mean()), "sign_error_rate": float(sign.mean())},
        "p95_positive_crucial_overestimate_s": float(np.quantile(positive_over[c], 0.95)),
        "late_warning": {
            "denominator": int(imminent.sum()),
            "rate": float((1 - warning[imminent]).mean()),
        },
        "false_alert": {"denominator": int(safe.sum()), "rate": float(warning[safe].mean())},
        "predictions_ensembled": False,
        "p95_definition": "p95 over queries of seed-mean nonnegative overestimate",
        "shared_representation": True,
    }
    table = pd.DataFrame(first.loc[:, keys]).copy()
    table["mean_MiD_loss"] = mid
    table["mean_absolute_error_s"] = ae
    table["mean_positive_overestimate_s"] = positive_over
    table["mean_sign_error"] = sign
    table["mean_warning_indicator"] = warning
    return result, table


def summarize(family: str) -> dict[str, Any]:
    templates = {"S": ("S-M0-L0-S{}", "S-M1-L0-S{}"), "R": ("R-C05-CTX-S{}", "R-C20-CTX-S{}")}
    results = {}
    tables = {}
    directory = RUN / "replication"
    directory.mkdir(exist_ok=True)
    for label, template in zip(("control", "candidate"), templates[family], strict=True):
        frames = []
        for seed in (7, 13, 23):
            stage = "screen" if seed == 7 else "replication"
            frames.append(
                pd.read_csv(
                    RUN / "evaluation" / stage / template.format(seed) / "PREDICTIONS.csv",
                    converters={"target_ttc": float, "prediction_ttc": np.float32},
                )
            )
        results[label], tables[label] = aggregate(frames)
        tables[label].to_csv(directory / f"{label}_MEAN_LOSS_PER_QUERY.csv", index=False)
        write(directory / f"{label}_METRICS.json", results[label])
    decision = gate(results["control"], results["candidate"])
    a = results["control"]["group_crucial_mid"]
    b = results["candidate"]["group_crucial_mid"]
    delta = np.asarray([b[g] - a[g] for g in sorted(a)])
    rng = np.random.default_rng(12345)
    draws = delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(1)
    write(
        directory / "BOOTSTRAP.json",
        {
            "seed": 12345,
            "replicates": 1000,
            "delta_MiDc": float(delta.mean()),
            "CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "draws": draws.tolist(),
        },
    )
    result = {
        "status": "REPLICATED_CANDIDATE_TRANSFER_PENDING"
        if decision["eligible"]
        else "NO_REPLICATED_CANDIDATE",
        "family": family,
        "seeds": [7, 13, 23],
        "paired_aggregate_gate": decision,
        "aggregation": "mean losses per query; no TTC ensemble",
        "encoders_retrained": False,
        "historical_V14_reference_seed": 7,
        "V_is_exposed_development": True,
    }
    freeze(directory / "DECISION.json", result)
    return result

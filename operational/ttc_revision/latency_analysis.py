"""Paired latency uncertainty with whole scenario families as bootstrap units."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from operational.train40_system.durable_io import atomic_json


def paired_latency(raw: pd.DataFrame, destination: Path) -> pd.DataFrame:
    """Preserve query/repetition pairs and resample complete families, not timings."""
    measured = cast(pd.DataFrame, raw[~raw.warmup])
    keys = ["mode", "scenario_family", "query_id", "iteration"]
    if measured.duplicated([*keys, "system"]).any():
        raise ValueError("duplicate system/query/repetition timing")
    pairs, families = [], []
    for mode in measured["mode"].unique():
        chosen = cast(pd.DataFrame, measured[measured["mode"] == mode])
        baseline = cast(pd.DataFrame, chosen[chosen.system == "h8_legacy_three"])
        for system in chosen.system.unique():
            if system == "h8_legacy_three":
                continue
            candidate = cast(pd.DataFrame, chosen[chosen.system == system])
            joined = cast(pd.DataFrame, baseline[keys + ["e2e_ms"]]).merge(
                cast(pd.DataFrame, candidate[keys + ["e2e_ms"]]),
                on=keys,
                how="outer",
                validate="one_to_one",
                suffixes=("_baseline", "_candidate"),
                indicator=True,
            )
            if joined.empty or not (joined["_merge"] == "both").all():
                raise ValueError("all timing observations require a matching baseline")
            timings = joined[["e2e_ms_baseline", "e2e_ms_candidate"]].to_numpy(float)
            if not np.isfinite(timings).all() or (timings <= 0).any():
                raise ValueError("finite positive elapsed times required")
            grouped = joined.groupby("scenario_family")
            totals = cast(pd.DataFrame, grouped[["e2e_ms_baseline", "e2e_ms_candidate"]].sum())
            counts = grouped.size().to_numpy()
            old, new = totals.to_numpy().T
            result = {
                "mode": mode,
                "candidate": system,
                "baseline": "h8_legacy_three",
                "pairs": len(joined),
                "families": len(totals),
                "difference_ms": float((new.sum() - old.sum()) / counts.sum()),
                "reduction_percent": float(100 * (1 - new.sum() / old.sum())),
            }
            if len(totals) >= 2:
                sampled = np.random.default_rng(20261009).integers(
                    len(totals), size=(10000, len(totals))
                )
                differences = (new[sampled] - old[sampled]).sum(1) / counts[sampled].sum(1)
                reductions = 100 * (1 - new[sampled].sum(1) / old[sampled].sum(1))
                result.update(
                    difference_ci_low_ms=float(np.quantile(differences, 0.025)),
                    difference_ci_high_ms=float(np.quantile(differences, 0.975)),
                    reduction_ci_low_percent=float(np.quantile(reductions, 0.025)),
                    reduction_ci_high_percent=float(np.quantile(reductions, 0.975)),
                )
            pairs.append(result)
            for index, family in enumerate(totals.index):
                families.append(
                    {
                        "mode": mode,
                        "candidate": system,
                        "scenario_family": family,
                        "pairs": int(counts[index]),
                        "baseline_ms": old[index] / counts[index],
                        "candidate_ms": new[index] / counts[index],
                        "reduction_percent": 100 * (1 - new[index] / old[index]),
                    }
                )
    destination.mkdir(parents=True, exist_ok=True)
    result_table = pd.DataFrame(pairs)
    result_table.to_csv(destination / "latency_paired_bootstrap.csv", index=False)
    pd.DataFrame(families).to_csv(destination / "latency_families.csv", index=False)
    atomic_json(
        destination / "LATENCY_STATISTICS.json",
        {
            "draws": 10000,
            "seed": 20261009,
            "unit": "whole scenario family, including all queries and repetitions",
            "interval": "percentile 95%",
            "scope": "conditional on measured queries, host and workload; not deployment assurance",
            "comparisons": "each mode independently; no paired cache-capacity claim across trials",
        },
    )
    return result_table

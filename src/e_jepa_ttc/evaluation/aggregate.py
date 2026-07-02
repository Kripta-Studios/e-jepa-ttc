"""Aggregate evaluation metric JSON files across seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_METRIC_NAMES = (
    "mae_s",
    "mean_abs_relative_error_pct",
    "median_abs_error_s",
    "median_abs_relative_error_pct",
    "rmse_s",
    "log_mae",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}."
        raise ValueError(msg)
    return payload


def aggregate_metric_files(
    paths: list[Path],
    *,
    split: str,
    metric_names: tuple[str, ...] = DEFAULT_METRIC_NAMES,
) -> dict[str, Any]:
    """Aggregate one split from saved evaluation metrics."""

    if not paths:
        msg = "At least one metrics path is required."
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        split_payload = payload.get("splits", {}).get(split)
        if not isinstance(split_payload, dict):
            msg = f"Split {split!r} not found in {path}."
            raise ValueError(msg)
        metrics = split_payload.get("metrics")
        if not isinstance(metrics, dict):
            msg = f"Split {split!r} in {path} has no metrics object."
            raise ValueError(msg)
        row = {
            "path": path.as_posix(),
            "seed": payload.get("seed", payload.get("checkpoint_seed")),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "count": split_payload.get("count"),
            "metrics": {name: float(metrics[name]) for name in metric_names if name in metrics},
        }
        rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.array(
            [row["metrics"][metric_name] for row in rows if metric_name in row["metrics"]],
            dtype=np.float64,
        )
        if values.size:
            summary[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return {
        "split": split,
        "metric_names": list(metric_names),
        "count": len(rows),
        "rows": rows,
        "summary": summary,
    }

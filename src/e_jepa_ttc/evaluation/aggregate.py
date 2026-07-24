"""Aggregate evaluation metric JSON files across seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.split import assert_split_claim_allowed

DEFAULT_METRIC_NAMES = (
    "mae_s",
    "mean_abs_relative_error_pct",
    "median_abs_error_s",
    "median_abs_relative_error_pct",
    "rmse_s",
    "log_mae",
    "log_rmse",
    "signed_log1p_mae",
    "mae_s_15s",
    "rmse_s_15s",
    "log_mae_15s",
    "log_rmse_15s",
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
    split_protocol_path: str | Path | None = None,
    claim_level: str = "diagnostic",
) -> dict[str, Any]:
    """Aggregate one split from saved evaluation metrics."""

    if claim_level in {"official", "final"} and split_protocol_path is None:
        raise ValueError("split_protocol_path is required for official/final result tables.")
    claim_gate = (
        assert_split_claim_allowed(split_protocol_path, claim_level=claim_level)
        if split_protocol_path is not None
        else {
            "requested_claim_level": claim_level,
            "claim_allowed": claim_level in {"development", "diagnostic"},
            "status": "unverified_no_split_protocol",
        }
    )
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
        pretrained = payload.get("pretrained_encoder")
        if not isinstance(pretrained, dict):
            pretrained = {}
        downstream_seed = payload.get(
            "downstream_seed",
            payload.get("seed", payload.get("checkpoint_seed")),
        )
        pretrain_seed = payload.get("pretrain_seed", pretrained.get("source_seed"))
        row = {
            "path": path.as_posix(),
            "seed": downstream_seed,
            "downstream_seed": downstream_seed,
            "pretrain_seed": pretrain_seed,
            "pretrain_checkpoint_role": pretrained.get("checkpoint_role"),
            "pretrain_checkpoint_selected_by": pretrained.get("checkpoint_selected_by"),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "count": split_payload.get("count"),
            "metrics": {name: float(metrics[name]) for name in metric_names if name in metrics},
        }
        rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        # Hierarchical grouping: group by pretrain_seed first
        groups: dict[Any, list[float]] = {}
        for row in rows:
            if metric_name in row["metrics"]:
                pt_seed = row.get("pretrain_seed")
                # If no pretrain seed, fallback to downstream seed (scratch models)
                if pt_seed is None:
                    pt_seed = row.get("downstream_seed")
                groups.setdefault(pt_seed, []).append(row["metrics"][metric_name])
                
        if not groups:
            continue
            
        group_means = [float(np.mean(vals)) for vals in groups.values()]
        group_means_arr = np.array(group_means, dtype=np.float64)
        all_values = np.array([v for vals in groups.values() for v in vals], dtype=np.float64)
        
        std = float(group_means_arr.std(ddof=1)) if group_means_arr.size > 1 else 0.0
        sem = std / np.sqrt(group_means_arr.size) if group_means_arr.size > 1 else 0.0
        
        summary[metric_name] = {
            "mean": float(group_means_arr.mean()),
            "std": std,
            "sem": sem,
            "min": float(all_values.min()),
            "max": float(all_values.max()),
            "pooled_std": float(all_values.std(ddof=1)) if all_values.size > 1 else 0.0,
        }
    pretrain_seeds = sorted(
        {row["pretrain_seed"] for row in rows if row["pretrain_seed"] is not None}
    )
    downstream_seeds = sorted(
        {row["downstream_seed"] for row in rows if row["downstream_seed"] is not None}
    )
    if len(pretrain_seeds) == 1 and len(downstream_seeds) > 1:
        uncertainty_scope = "downstream_only_conditional_on_single_pretrain_seed"
    elif len(pretrain_seeds) > 1:
        uncertainty_scope = "multiple_pretrain_and_downstream_seeds"
    elif not pretrain_seeds:
        uncertainty_scope = "pretrain_seed_not_recorded"
    else:
        uncertainty_scope = "single_run_or_single_seed"
    return {
        "split": split,
        "metric_names": list(metric_names),
        "count": len(rows),
        "pretrain_seed_count": len(pretrain_seeds),
        "pretrain_seeds": pretrain_seeds,
        "downstream_seed_count": len(downstream_seeds),
        "downstream_seeds": downstream_seeds,
        "uncertainty_scope": uncertainty_scope,
        "claim_gate": claim_gate,
        "rows": rows,
        "summary": summary,
    }

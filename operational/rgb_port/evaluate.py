"""Cohort-complete signed TTC evaluation for RGB-PORT."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .accounting import atomic_write_json

BUCKETS = (
    ("crucial", 0.0, 3.0),
    ("small", 3.0, 6.0),
    ("large", 6.0, 10.0),
    ("negative", -10.0, 0.0),
)
WEIGHTS = {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1}
HEAD_IDS = {"E_H1_MATCHED", "E_CTX_MATCHED", "R_H1", "R_CTX", "F_TRUE", "F_ZERO"}
REQUIRED_CONTRASTS = {
    "E_CTX_MATCHED=E_H1_MATCHED",
    "R_CTX=R_H1",
    "F_TRUE=F_ZERO",
    "F_TRUE=E_CTX_MATCHED",
}


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _bucket_mask(target: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (target > lower) & (target <= upper)


def score_signed_ttc(
    target_ttc: Iterable[float],
    prediction_ttc: Iterable[float],
    groups: Iterable[str] | None = None,
    *,
    delta_t_s: float = 0.1,
) -> dict[str, Any]:
    """Score all rows and compute the preregistered group-macro bucket MiD."""
    target = np.asarray(list(target_ttc), dtype=np.float64).reshape(-1)
    prediction = np.asarray(list(prediction_ttc), dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != prediction.shape or delta_t_s <= 0:
        raise ValueError("Non-empty shape-matched arrays and positive delta_t_s are required")
    if not np.isfinite(target).all():
        raise ValueError("Ground-truth TTC must be finite float64")
    group = (
        np.asarray(list(groups)).astype(str).reshape(-1)
        if groups is not None
        else np.repeat("__missing_group_contract__", target.size)
    )
    if group.shape != target.shape:
        raise ValueError("Group IDs must match TTC arrays")
    domain = np.zeros(target.shape, dtype=bool)
    failed = ~np.isfinite(prediction) | (np.abs(prediction) < delta_t_s)
    bins: dict[str, Any] = {}
    formula_parts: dict[str, float] = {}
    for name, lower, upper in BUCKETS:
        selected = _bucket_mask(target, lower, upper)
        domain |= selected
        truth, estimate = target[selected], prediction[selected]
        count = int(selected.sum())
        bucket_failed = failed[selected]
        finite = np.isfinite(estimate)
        with np.errstate(divide="ignore", invalid="ignore"):
            truth_eta = 1.0 - delta_t_s / truth
            estimate_eta = 1.0 - delta_t_s / estimate
            mid_rows = np.abs(np.log(truth_eta) - np.log(estimate_eta)) * 1e4
            rte_rows = np.abs(estimate - truth) / np.abs(truth) * 100.0
        usable_mid = np.isfinite(mid_rows)
        phase_invalid = ~np.isfinite(mid_rows)
        mid = float(np.mean(mid_rows[usable_mid])) if np.any(usable_mid) else float("nan")
        rte = float(np.mean(rte_rows[finite])) if np.any(finite) else float("nan")
        group_mid: dict[str, float] = {}
        for group_id in sorted(np.unique(group[selected]).tolist()):
            in_group = group[selected] == group_id
            values = mid_rows[in_group]
            if values.size and np.isfinite(values).all():
                group_mid[group_id] = float(values.mean())
        complete = (
            groups is not None
            and count > 0
            and not np.any(bucket_failed)
            and bool(np.all(np.isfinite(mid_rows)))
            and bool(group_mid)
        )
        if complete:
            formula_parts[name] = float(np.mean(list(group_mid.values())))
        bins[name] = {
            "count": count,
            "failure_count": int(bucket_failed.sum()),
            "phase_invalid_count": int(phase_invalid.sum()),
            "coverage": float(finite.mean()) if count else 0.0,
            "pooled_diagnostic_mid": _finite_or_none(mid),
            "diagnostic_rte_pct": _finite_or_none(rte),
            "group_macro_mid": _finite_or_none(formula_parts.get(name, float("nan"))),
            "groups_with_bucket": len(group_mid),
            "formula_bucket_admissible": complete,
        }
    outside = int((~domain).sum())
    complete = len(formula_parts) == len(BUCKETS)
    overall = sum(WEIGHTS[name] * formula_parts[name] for name in WEIGHTS) if complete else None
    finite_prediction = np.isfinite(prediction)
    errors = prediction[finite_prediction] - target[finite_prediction]
    abs_errors = np.abs(errors)
    crucial_valid = finite_prediction & (target > 0) & (target <= 3)
    crucial_over = prediction[crucial_valid] > target[crucial_valid]
    crucial_over_rte = (
        (prediction[crucial_valid][crucial_over] - target[crucial_valid][crucial_over])
        / target[crucial_valid][crucial_over]
        * 100.0
    )
    return {
        "schema": "rgb_port_signed_metrics_v1",
        "protocol": "garl_signed_v1_exact_buckets",
        "num_rows": int(target.size),
        "num_failed_predictions": int(failed.sum()),
        "num_targets_outside_protocol": outside,
        "coverage": float(finite_prediction.mean()),
        "formula_admitted": complete,
        "official_comparison_admitted": False,
        "group_macro_bucket_MiD": overall,
        "bins": bins,
        "diagnostics": {
            "mae_s": _finite_or_none(float(abs_errors.mean()) if abs_errors.size else float("nan")),
            "median_ae_s": _finite_or_none(
                float(np.median(abs_errors)) if abs_errors.size else float("nan")
            ),
            "rmse_s": _finite_or_none(
                float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan")
            ),
            "p90_ae_s": _finite_or_none(
                float(np.quantile(abs_errors, 0.90)) if abs_errors.size else float("nan")
            ),
            "p95_ae_s": _finite_or_none(
                float(np.quantile(abs_errors, 0.95)) if abs_errors.size else float("nan")
            ),
            "rte_pct": _finite_or_none(
                float(np.mean(abs_errors / np.abs(target[finite_prediction])) * 100.0)
                if errors.size
                else float("nan")
            ),
            "sign_error_rate": _finite_or_none(
                float(
                    np.mean(
                        np.signbit(prediction[finite_prediction])
                        != np.signbit(target[finite_prediction])
                    )
                )
                if errors.size
                else float("nan")
            ),
            "crucial_mae_s": _finite_or_none(
                float(np.mean(np.abs(prediction[crucial_valid] - target[crucial_valid])))
                if np.any(crucial_valid)
                else float("nan")
            ),
            "crucial_positive_overestimate_rate": _finite_or_none(
                float(np.mean(crucial_over)) if crucial_over.size else float("nan")
            ),
            "crucial_positive_overestimate_mean_s": _finite_or_none(
                float(np.mean(np.maximum(prediction[crucial_valid] - target[crucial_valid], 0.0)))
                if crucial_over.size
                else float("nan")
            ),
            "crucial_positive_overestimate_rte_p90_pct": _finite_or_none(
                float(np.quantile(crucial_over_rte, 0.90))
                if crucial_over_rte.size
                else float("nan")
            ),
            "crucial_positive_overestimate_rte_p95_pct": _finite_or_none(
                float(np.quantile(crucial_over_rte, 0.95))
                if crucial_over_rte.size
                else float("nan")
            ),
        },
    }


def common_cap60_metrics(
    target: Iterable[float], prediction: Iterable[float], groups: Iterable[str] | None = None
) -> dict[str, Any]:
    prediction_array = np.asarray(list(prediction), dtype=np.float64)
    capped = np.where(
        np.isfinite(prediction_array), np.clip(prediction_array, -60.0, 60.0), prediction_array
    )
    result = score_signed_ttc(target, capped, groups)
    result["analysis_scope"] = "common_prediction_cap_60s_separate_from_native"
    return result


def paired_group_bootstrap(
    target: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int = 2000,
    seed: int = 7,
) -> dict[str, Any]:
    arrays = [
        np.asarray(value).reshape(-1) for value in (target, prediction_a, prediction_b, groups)
    ]
    if not arrays[0].size or any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("Paired bootstrap arrays must be non-empty and shape matched")
    unique = np.unique(arrays[3].astype(str))
    if unique.size < 2:
        return {"status": "BLOCKED_INSUFFICIENT_GROUPS", "group_count": int(unique.size)}
    by_group = {group: np.flatnonzero(arrays[3].astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        blocks = [by_group[group_id] for group_id in sampled]
        indices = np.concatenate(blocks)
        sampled_groups = np.concatenate(
            [
                np.repeat(f"draw_{position}:{group_id}", len(block))
                for position, (group_id, block) in enumerate(zip(sampled, blocks, strict=True))
            ]
        )
        a = score_signed_ttc(arrays[0][indices], arrays[1][indices], sampled_groups)
        b = score_signed_ttc(arrays[0][indices], arrays[2][indices], sampled_groups)
        if a["formula_admitted"] and b["formula_admitted"]:
            differences.append(float(a["group_macro_bucket_MiD"] - b["group_macro_bucket_MiD"]))
    if not differences:
        return {"status": "BLOCKED_NO_ADMISSIBLE_RESAMPLES", "group_count": int(unique.size)}
    values = np.asarray(differences, dtype=np.float64)
    return {
        "status": "COMPLETE",
        "group_count": int(unique.size),
        "repetitions_requested": repetitions,
        "repetitions_admissible": int(values.size),
        "difference_a_minus_b_mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "seed": seed,
    }


def screening_gate(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if not candidate.get("formula_admitted") or not control.get("formula_admitted"):
        return {"passed": False, "status": "INADMISSIBLE_METRICS"}
    c_mid = float(candidate["group_macro_bucket_MiD"])
    b_mid = float(control["group_macro_bucket_MiD"])
    cd, bd = candidate["diagnostics"], control["diagnostics"]
    coverage_equal = abs(float(candidate["coverage"]) - float(control["coverage"])) <= 1e-12
    checks = {
        "mid_improvement_at_least_5pct": c_mid <= 0.95 * b_mid,
        "rte_not_worse_over_5pct": float(cd["rte_pct"]) <= 1.05 * float(bd["rte_pct"]),
        "crucial_mae_not_worse_over_5pct": float(cd["crucial_mae_s"])
        <= 1.05 * float(bd["crucial_mae_s"]),
        "sign_error_not_worse_over_0_5pp": float(cd["sign_error_rate"])
        <= float(bd["sign_error_rate"]) + 0.005,
        "coverage_equal": coverage_equal,
    }
    return {"passed": all(checks.values()), "status": "COMPLETE", "checks": checks}


def _read_prediction_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_token", "target_ttc", "prediction_ttc", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Prediction CSV requires columns {sorted(required)}")
    return (
        np.asarray([float(row["target_ttc"]) for row in rows], dtype=np.float64),
        np.asarray([float(row["prediction_ttc"]) for row in rows], dtype=np.float64),
        np.asarray([row["group_id"] for row in rows]),
        np.asarray([row["sample_token"] for row in rows]),
    )


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("Expected NAME=PATH")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score frozen RGB-PORT V predictions without selecting models."
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--candidate", action="append", default=[], help="Frozen endpoint as NAME=CSV"
    )
    parser.add_argument(
        "--contrast", action="append", default=[], help="Screening contrast as CANDIDATE=CONTROL"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.predictions) == bool(args.candidate):
        raise ValueError("Use either --predictions or one or more --candidate NAME=CSV")
    if args.predictions:
        target, prediction, groups, _ = _read_prediction_csv(args.predictions)
        result = {
            "native": score_signed_ttc(target, prediction, groups),
            "common_cap60": common_cap60_metrics(target, prediction, groups),
        }
    else:
        tables = {
            name: _read_prediction_csv(path)
            for name, path in map(_parse_named_path, args.candidate)
        }
        if set(tables) != HEAD_IDS:
            raise ValueError("Campaign evaluation requires all six frozen head endpoints")
        if len(tables) != len(args.candidate):
            raise ValueError("Candidate names must be unique")
        if (
            len(args.contrast) != len(REQUIRED_CONTRASTS)
            or set(args.contrast) != REQUIRED_CONTRASTS
        ):
            raise ValueError(
                "Campaign evaluation requires exactly the four preregistered contrasts"
            )
        reference_target, _, reference_groups, reference_tokens = next(iter(tables.values()))
        if any(
            not np.array_equal(target, reference_target)
            or not np.array_equal(groups, reference_groups)
            or not np.array_equal(tokens, reference_tokens)
            for target, _, groups, tokens in tables.values()
        ):
            raise ValueError("Candidate V cohorts are not exactly matched")
        scores = {
            name: {
                "native": score_signed_ttc(target, prediction, groups),
                "common_cap60": common_cap60_metrics(target, prediction, groups),
            }
            for name, (target, prediction, groups, _) in tables.items()
        }
        gates: dict[str, Any] = {}
        for contrast in args.contrast:
            candidate, separator, control = contrast.partition("=")
            if not separator or candidate not in tables or control not in tables:
                raise ValueError(f"Invalid contrast: {contrast}")
            gate = screening_gate(scores[candidate]["native"], scores[control]["native"])
            gate["bootstrap"] = paired_group_bootstrap(
                reference_target,
                tables[candidate][1],
                tables[control][1],
                reference_groups,
            )
            gates[contrast] = gate
        result = {
            "schema": "rgb_port_v_campaign_evaluation_v1",
            "status": "COMPLETE",
            "selection_performed": False,
            "scores": scores,
            "screening_gates": gates,
        }
    atomic_write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

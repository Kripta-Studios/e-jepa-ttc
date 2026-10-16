"""CPU diagnostics of frozen experts; ground-truth oracles are never candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, phase_to_ttc
from operational.range_transfer.analyze import gate

from .common import RUN, SOURCE_ROOT, load_population, score, sha, source_bindings, write_json

NAMES = ["A5", "NATIVE", "CORR", "C1", "B1", "FULL", "LAST"]
PAIRS = [("FULL", "A5"), ("FULL", "NATIVE"), ("CORR", "NATIVE"), ("B1", "NATIVE")]
IDENTITY_COLUMNS = ["sample_token", "group_id", "target_ttc"]


def _array_hash(values: np.ndarray) -> str:
    """Hash the exact selected prediction vector as little-endian float64."""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _tokens_hash(table: pd.DataFrame) -> str:
    values = table.sample_token.astype(str).tolist()
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def _mid(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return 10000.0 * np.abs(np.log1p(-0.1 / target) - np.log1p(-0.1 / prediction))


def _validate(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    if not tables or "FULL" not in tables:
        raise ValueError("A nonempty population with FULL is required")
    reference = tables["FULL"]
    if reference.empty:
        raise ValueError("Empty population")
    roles = set(reference.role.unique()) if "role" in reference else set()
    if len(roles) != 1 or not roles <= {"P", "H"}:
        raise ValueError("An explicit, unique P or H role is required; no V diagnostics")
    role = str(next(iter(roles)))
    for name, table in tables.items():
        required = {*IDENTITY_COLUMNS, "prediction_ttc", "raw_phase", "role"}
        if not required <= set(table.columns):
            raise ValueError(f"Missing identity/prediction columns for {name}")
        if table.sample_token.isna().any() or table.sample_token.duplicated().any():
            raise ValueError(f"Null or duplicated query IDs for {name}")
        if table.group_id.isna().any() or set(table.role.unique()) != {role}:
            raise ValueError(f"Invalid group or role for {name}")
        columns = IDENTITY_COLUMNS + (["sequence_id"] if "sequence_id" in reference else [])
        for column in columns:
            if column not in table or not np.array_equal(reference[column], table[column]):
                raise ValueError(f"Population identity/order mismatch: {name}/{column}")
        for column in ["target_ttc", "prediction_ttc", "raw_phase"]:
            if not np.isfinite(table[column].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"Nonfinite {column} for {name}; no rows may be dropped")
        pred = table.prediction_ttc.to_numpy(dtype=np.float64)
        if not ((pred < 0) | (pred > 0.1)).all():
            raise ValueError(f"Prediction outside the signed phase domain: {name}")
    target = reference.target_ttc.to_numpy(dtype=np.float64)
    negative = (target > -10) & (target < 0)
    positive = (target > 0.1) & (target <= 10)
    if not (negative | positive).all():
        raise ValueError("Targets outside registered four-band population")
    return reference, role


def _error_summary(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    """Pooled diagnostic quantities; these are not the group-macro scorer."""
    count = int(mask.sum())
    if not count:
        return {
            "count": 0,
            "mean_MiD": None,
            "mae_s": None,
            "absolute_error_sum_s": 0.0,
            "median_ae_s": None,
            "p95_ae_s": None,
            "sign_error_count": 0,
            "sign_error_rate": None,
        }
    truth, pred = target[mask], prediction[mask]
    absolute = np.abs(truth - pred)
    wrong = np.signbit(truth) != np.signbit(pred)
    return {
        "count": count,
        "mean_MiD": float(_mid(truth, pred).mean()),
        "mae_s": float(absolute.mean()),
        "absolute_error_sum_s": float(absolute.sum()),
        "median_ae_s": float(np.median(absolute)),
        "p95_ae_s": float(np.quantile(absolute, 0.95)),
        "sign_error_count": int(wrong.sum()),
        "sign_error_rate": float(wrong.mean()),
    }


def _by_group(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, groups: np.ndarray
) -> dict:
    return {
        str(group): _error_summary(target, prediction, mask & (groups == group))
        for group in np.unique(groups)
    }


def _saturation(table: pd.DataFrame, name: str) -> tuple[np.ndarray, dict]:
    """Use the native float32 conversion threshold, without decimal equality tests."""
    raw = torch.tensor(table.raw_phase.to_numpy(dtype=np.float32), device="cpu")
    clipped = raw.clamp(MIN_PHASE, MAX_PHASE)
    q = -torch.expm1(-clipped) / 0.1
    saturated = (q.abs() <= 1.0 / 60.0).numpy()
    reconstructed = phase_to_ttc(raw).numpy()
    error = np.abs(reconstructed - table.prediction_ttc.to_numpy(dtype=np.float64))
    return saturated, {
        "method": "native CPU float32: abs(-expm1(-clamp(raw_phase))/0.1) <= 1/60",
        "output_support_seconds": 60,
        "reconversion_max_abs_difference_s": float(error.max()),
        "phase_reconstructed_from_prediction_caution": name in {"A5", "NATIVE"},
        "caution": (
            "A5/NATIVE may store phase reconstructed from TTC; threshold-boundary membership "
            "does not establish the producer's original pre-conversion saturation state."
        ),
    }


def _model_diagnostics(table: pd.DataFrame, name: str) -> dict:
    target = table.target_ttc.to_numpy(dtype=np.float64)
    pred = table.prediction_ttc.to_numpy(dtype=np.float64)
    groups = table.group_id.to_numpy()
    negative = target < 0
    wrong = np.signbit(target) != np.signbit(pred)
    saturated, native = _saturation(table, name)
    negative_error = float(np.abs(target[negative] - pred[negative]).sum())
    cells = {}
    for is_wrong in (False, True):
        for is_saturated in (False, True):
            label = f"sign_{'wrong' if is_wrong else 'correct'}_saturated_{is_saturated}"
            mask = negative & (wrong == is_wrong) & (saturated == is_saturated)
            values = _error_summary(target, pred, mask)
            values["fraction_of_negative_absolute_error"] = (
                values["absolute_error_sum_s"] / negative_error if negative_error else None
            )
            cells[label] = {**values, "by_group": _by_group(target, pred, mask, groups)}
    return {
        "metrics": score(table, pred),
        "negative": _error_summary(target, pred, negative),
        "positive": _error_summary(target, pred, ~negative),
        "negative_correct_sign": _error_summary(target, pred, negative & ~wrong),
        "negative_saturated_count": int((negative & saturated).sum()),
        "negative_by_group": _by_group(target, pred, negative, groups),
        "positive_by_group": _by_group(target, pred, ~negative, groups),
        "native_saturation": native,
        "negative_sign_saturation_cells": cells,
    }


def sign_cross_cells(full: pd.DataFrame, comparator: pd.DataFrame) -> dict:
    """Compare experts on the same negative queries in four disjoint sign cells."""
    target = full.target_ttc.to_numpy(dtype=np.float64)
    first = full.prediction_ttc.to_numpy(dtype=np.float64)
    second = comparator.prediction_ttc.to_numpy(dtype=np.float64)
    groups = full.group_id.to_numpy()
    negative = target < 0
    first_correct = np.signbit(first) == np.signbit(target)
    second_correct = np.signbit(second) == np.signbit(target)
    cells = {}
    membership = np.zeros(len(target), dtype=np.int8)
    for first_ok in (False, True):
        for second_ok in (False, True):
            mask = negative & (first_correct == first_ok) & (second_correct == second_ok)
            membership += mask
            label = f"FULL_correct_{first_ok}_comparator_correct_{second_ok}"
            cells[label] = {
                "count": int(mask.sum()),
                "query_ids_sha256": _tokens_hash(full.loc[mask]),
                "FULL": _error_summary(target, first, mask),
                "comparator": _error_summary(target, second, mask),
                "by_group": {
                    str(group): {
                        "FULL": _error_summary(target, first, mask & (groups == group)),
                        "comparator": _error_summary(target, second, mask & (groups == group)),
                    }
                    for group in np.unique(groups)
                },
            }
    if not np.array_equal(membership, negative.astype(np.int8)):
        raise AssertionError("Negative sign cells must be exhaustive and disjoint")
    return {"negative_count": int(negative.sum()), "cells": cells}


def _oracle_record(table: pd.DataFrame, prediction: np.ndarray) -> dict:
    # Exactly one scorer call consumes the final vector for every reported metric.
    values = score(table, prediction)
    return {
        "status": "ORACLE_NON_DEPLOYABLE_DIAGNOSTIC",
        "candidate": False,
        "system_promotion": False,
        "role": "P",
        "ground_truth_used_for_selection": True,
        "prediction_vector_sha256_float64_le": _array_hash(prediction),
        "metrics_from_one_vector": values,
        "cost": "UNMEASURED; all required producers must be counted for a learned router",
    }


def compute(tables: dict[str, pd.DataFrame]) -> dict:
    """Compute frozen-table diagnostics; H cannot emit oracle results."""
    reference, role = _validate(tables)
    target = reference.target_ttc.to_numpy(dtype=np.float64)
    result = {
        "schema": "decision_recovery_frozen_diagnostics_v1",
        "role": role,
        "rows": len(reference),
        "query_ids_sha256": _tokens_hash(reference),
        "negative_domain": "(-10, 0] with zero excluded by signed TTC domain",
        "negative_domain_equals_target_below_zero": True,
        "scope": "P training diagnostic" if role == "P" else "H exposed development decomposition",
        "metric_contract": {
            "per_query_MiD": "10000*abs(log1p(-0.1/target)-log1p(-0.1/prediction))",
            "overall": "group-macro within bands; masses crucial/small/large/negative=.5/.3/.1/.1",
            "cell_metrics": "pooled on exactly the cell's queries; per-group values also provided",
        },
        "models": {name: _model_diagnostics(table, name) for name, table in tables.items()},
        "negative_sign_cross_vs_FULL": {
            name: sign_cross_cells(reference, table)
            for name, table in tables.items()
            if name != "FULL"
        },
        "oracles": {},
        "candidate": False,
        "system_promotion": False,
    }
    if role == "H":
        return result
    for specialist, baseline in PAIRS:
        if specialist not in tables or baseline not in tables:
            continue
        special = tables[specialist].prediction_ttc.to_numpy(dtype=np.float64)
        base = tables[baseline].prediction_ttc.to_numpy(dtype=np.float64)
        selected = _mid(target, special) < _mid(target, base)
        prediction = np.where(selected, special, base)
        record = _oracle_record(reference, prediction)
        record.update(
            {
                "objective": "minimum per-query MiD; ties choose baseline",
                "specialist": specialist,
                "baseline": baseline,
                "selected_counts": {
                    specialist: int(selected.sum()),
                    baseline: int((~selected).sum()),
                },
                "selected_specialist_query_ids_sha256": _tokens_hash(reference.loc[selected]),
                "shared_sign_error_count": int(
                    (
                        (np.signbit(special) != np.signbit(target))
                        & (np.signbit(base) != np.signbit(target))
                    ).sum()
                ),
                "accuracy_gate_diagnostic_not_admission": {
                    name: gate(result["models"][name]["metrics"], record["metrics_from_one_vector"])
                    for name in (specialist, baseline)
                },
            }
        )
        result["oracles"][f"{specialist}_vs_{baseline}_minimum_MiD"] = record
    full = reference.prediction_ttc.to_numpy(dtype=np.float64)
    corrected = np.copysign(np.abs(full), target)
    if not ((corrected < 0) | (corrected > 0.1)).all():
        raise ValueError("Sign-only oracle leaves native signed domain; do not clip or repair it")
    record = _oracle_record(reference, corrected)
    record.update(
        {
            "objective": "correct only FULL sign using GT; preserve every absolute TTC magnitude",
            "magnitude_preserved_exactly": bool(np.array_equal(np.abs(full), np.abs(corrected))),
            "changed_sign_count": int((np.signbit(full) != np.signbit(corrected)).sum()),
            "distinct_from_selection_oracle": True,
            "accuracy_gate_diagnostic_not_admission": {
                name: gate(result["models"][name]["metrics"], record["metrics_from_one_vector"])
                for name in ("FULL", "A5", "NATIVE")
                if name in tables
            },
        }
    )
    result["oracles"]["FULL_sign_only"] = record
    return result


def run() -> dict:
    """Write diagnostics only when explicitly invoked by the registered campaign."""
    destination = RUN / "diagnostics"
    sources = {
        "source_root": str(SOURCE_ROOT),
        "diagnostic_implementation_sha256": sha(Path(__file__)),
        "prediction_bindings": {role: source_bindings(role, NAMES) for role in ("P", "H")},
        "historical_contracts": {
            relative: sha(SOURCE_ROOT / relative)
            for relative in (
                "operational/range_transfer/analyze.py",
                "operational/joint_measurement/evaluate.py",
                "src/e_jepa_ttc/simplex_t/phase.py",
            )
        },
    }
    results = {role: compute(load_population(role, NAMES)) for role in ("P", "H")}
    for role, result in results.items():
        write_json(destination / f"{role}.json", result)
    summary = {
        "status": "COMPLETE_FROZEN_CPU_DIAGNOSTICS_NOT_CANDIDATES",
        "sources": sources,
        "population_rows": {role: result["rows"] for role, result in results.items()},
        "artifacts": {f"{role}.json": sha(destination / f"{role}.json") for role in results},
        "optimizer_updates": 0,
        "inference_runs": 0,
        "GPU_used": False,
        "oracle_roles": ["P"],
        "H_scope": "already exposed development decomposition; no training or oracle",
    }
    write_json(destination / "COMPLETE.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

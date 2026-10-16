"""Fixed output-only relative-cost routing; H results are development evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES = (
    "phase_base",
    "phase_special",
    "diff",
    "absdiff",
    "absphase_base",
    "absphase_special",
    "sign_disagreement",
    "log_magnitude_difference",
)
BANDS = (
    ("crucial", 0.1, 3.0, 0.5),
    ("small", 3.0, 6.0, 0.3),
    ("large", 6.0, 10.0, 0.1),
    ("negative", -10.0, 0.0, 0.1),
)
PAIRS = (("NATIVE", "FULL"), ("A5", "FULL"), ("NATIVE", "CORR"))


def _phase(ttc: np.ndarray) -> np.ndarray:
    values = np.asarray(ttc, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("TTC must be a nonempty finite vector")
    if not ((values < 0) | (values > 0.1)).all():
        raise ValueError("TTC outside the finite signed phase domain")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        phase = -np.log1p(-0.1 / values)
    if not np.isfinite(phase).all():
        raise ValueError("TTC does not produce finite phase")
    return phase


def features(base_ttc: np.ndarray, special_ttc: np.ndarray) -> np.ndarray:
    """Compute exactly eight features, accepting expert outputs only."""
    a, b = np.asarray(base_ttc, dtype=np.float64), np.asarray(special_ttc, dtype=np.float64)
    pa, pb = _phase(a), _phase(b)
    if a.shape != b.shape:
        raise ValueError("Expert prediction shapes differ")
    difference = pb - pa
    return np.column_stack(
        (
            pa,
            pb,
            difference,
            abs(difference),
            abs(pa),
            abs(pb),
            (np.sign(a) != np.sign(b)).astype(np.float64),
            np.log1p(abs(b)) - np.log1p(abs(a)),
        )
    )


def training_weights(target: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Equal group mass within each band, calculated from this training fold only."""
    target = np.asarray(target, dtype=np.float64)
    _phase(target)
    groups = np.asarray(groups).astype(str)
    if groups.shape != target.shape:
        raise ValueError("Training group shape differs")
    weight = np.zeros(len(target), dtype=np.float64)
    for _, lower, upper, mass in BANDS:
        selected = (target > lower) & (target <= upper)
        present = np.unique(groups[selected])
        if not len(present):
            raise ValueError("Training fold has a missing band")
        for group in present:
            cell = selected & (groups == group)
            weight[cell] = mass / (len(present) * cell.sum())
    if not (weight > 0).all() or not np.isclose(weight.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Training targets outside registered bands")
    return weight


def fit(x: np.ndarray, relative_cost: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Fit fixed ridge .01; costs must already be (special MiD-base MiD)/100."""
    x = np.asarray(x, dtype=np.float64)
    y, w = np.asarray(relative_cost, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 8 or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError("Ridge shapes disagree")
    if not len(x) or not all(np.isfinite(v).all() for v in (x, y, w)):
        raise ValueError("Ridge inputs must be finite")
    if not (w > 0).all() or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Global training weights must sum to one")
    mean = w @ x
    scale = np.maximum(np.sqrt(w @ np.square(x - mean)), 1e-8)
    z = (x - mean) / scale
    intercept = float(w @ y)
    lhs = z.T @ (w[:, None] * z) + 0.01 * np.eye(8)
    rhs = z.T @ (w * (y - intercept))
    coefficient = np.linalg.solve(lhs, rhs)
    if not np.allclose(lhs @ coefficient, rhs, rtol=1e-10, atol=1e-10):
        raise ArithmeticError("Ridge solve failed")
    return {
        "schema": "relative_cost_router_v1",
        "features": list(FEATURES),
        "ridge": 0.01,
        "relative_cost_divisor": 100.0,
        "threshold": 0.0,
        "tie": "base",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": intercept,
        "scope": "DEVELOPMENT_MODEL_NOT_PROMOTION",
        "system_promotion": False,
    }


def predict(model: dict[str, Any], base_ttc: np.ndarray, special_ttc: np.ndarray) -> np.ndarray:
    """Emit a whole expert output; no blending, targets or oracle decision."""
    chosen = _choose_special(model, base_ttc, special_ttc)
    return np.where(chosen, np.asarray(special_ttc), np.asarray(base_ttc))


def _choose_special(
    model: dict[str, Any], base_ttc: np.ndarray, special_ttc: np.ndarray
) -> np.ndarray:
    """Use the same rule for emission and provenance, including equal outputs."""
    x = features(base_ttc, special_ttc)
    if (
        model["schema"] != "relative_cost_router_v1"
        or model["features"] != list(FEATURES)
        or model["ridge"] != 0.01
        or model["threshold"] != 0.0
        or model["relative_cost_divisor"] != 100.0
        or model["tie"] != "base"
    ):
        raise ValueError("Router export contract differs")
    mean, scale, coefficient = (
        np.asarray(model[k], dtype=np.float64) for k in ("mean", "scale", "coefficient")
    )
    if any(v.shape != (8,) or not np.isfinite(v).all() for v in (mean, scale, coefficient)):
        raise ValueError("Invalid router parameters")
    if not (scale > 0).all() or not np.isfinite(model["intercept"]):
        raise ValueError("Invalid router scale/intercept")
    regret = ((x - mean) / scale) @ coefficient + model["intercept"]
    if not np.isfinite(regret).all():
        raise ValueError("Nonfinite predicted regret")
    return regret < 0


def _aligned(tables: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    reference = tables[names[0]]
    if reference.empty or reference.sample_token.duplicated().any():
        raise ValueError("Empty or duplicated population")
    for name in names:
        table = tables[name]
        for column in ("sample_token", "group_id", "sequence_id", "target_ttc", "role"):
            if not np.array_equal(reference[column], table[column]):
                raise ValueError(f"Population mismatch: {name}/{column}")
        _phase(table.prediction_ttc.to_numpy())
    _phase(reference.target_ttc.to_numpy())
    if reference.group_id.isna().any():
        raise ValueError("Missing acquisition groups")
    return reference


def _bootstrap(table: pd.DataFrame, reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Paired group bootstrap of fixed OOF vectors, not a refit bootstrap."""
    target, groups = table.target_ttc.to_numpy(), table.group_id.astype(str).to_numpy()
    unique = np.unique(groups)
    loss = 10000 * (
        abs(_phase(candidate) - _phase(target)) - abs(_phase(reference) - _phase(target))
    )
    rows = np.full((len(unique), 4), np.nan)
    for i, group in enumerate(unique):
        for j, (_, lower, upper, _) in enumerate(BANDS):
            chosen = (groups == group) & (target > lower) & (target <= upper)
            if chosen.any():
                rows[i, j] = loss[chosen].mean()
    draws = np.random.default_rng(12345).integers(len(unique), size=(2000, len(unique)))
    sampled = rows[draws]
    counts = np.isfinite(sampled).sum(axis=1)
    delta = np.divide(
        np.nansum(sampled, axis=1), counts, out=np.full((2000, 4), np.nan), where=counts > 0
    )
    values = np.column_stack((delta, delta @ np.array([b[3] for b in BANDS])))
    intervals = {}
    for j, name in enumerate([b[0] for b in BANDS] + ["overall"]):
        defined = np.isfinite(values[:, j])
        intervals[name] = {
            "defined_replicates": int(defined.sum()),
            "undefined_replicates": int((~defined).sum()),
            "ci95_if_all_defined": np.quantile(values[:, j], [0.025, 0.975]).tolist()
            if defined.all()
            else None,
        }
    return {
        "seed": 12345,
        "replicates": 2000,
        "unit": "acquisition_group",
        "scope": "descriptive_fixed_folds_not_independent_confirmation",
        "refitted_per_bootstrap": False,
        "intervals": intervals,
    }


def development_gate(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep the V18 A5 development gate distinct from stricter system safeguards."""
    if reference["overall_MiD"] is None or candidate["overall_MiD"] is None:
        return {"eligible": False, "status": "LIMITED_VALIDATION_SUPPORT", "checks": {}}

    def ratio(value: float, baseline: float) -> float | None:
        if baseline == 0:
            return 1.0 if value == 0 else None
        return float(value / baseline)

    overall = ratio(candidate["overall_MiD"], reference["overall_MiD"])
    ratios = {
        name: ratio(
            candidate["bins"][name]["group_macro_mid"], reference["bins"][name]["group_macro_mid"]
        )
        for name, _, _, _ in BANDS
    }
    checks = {
        "coverage": bool(
            reference["complete_prediction_and_mid_coverage"]
            and candidate["complete_prediction_and_mid_coverage"]
        ),
        "sign": candidate["diagnostics"]["sign_error_rate"]
        <= reference["diagnostics"]["sign_error_rate"] + 0.005,
        "bands": all(value is not None and value <= 1.1 for value in ratios.values()),
        "aggregate_or_crucial": bool(
            overall is not None
            and (
                overall <= 0.97
                or (ratios["crucial"] is not None and ratios["crucial"] <= 0.95 and overall <= 1.02)
            )
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "overall_ratio": overall,
        "band_ratios": ratios,
        "status_scope": "DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION",
    }


def evaluate_pair(
    tables: dict[str, pd.DataFrame], base: str, special: str, *, output_dir: Path | None = None
) -> dict[str, Any]:
    """Evaluate registered H LOGO routing and its shuffled-label negative control.

    Caller must complete prospective registration and ancestry admission first.
    No population is loaded here; this function cannot open V or launch GPU work.
    """
    from operational.range_transfer.analyze import gate

    from .common import RUN, score, sha, write_json

    if (base, special) not in PAIRS:
        raise ValueError("Pair is not in the fixed registered matrix")
    names = list(dict.fromkeys((base, special, "A5")))
    table = _aligned(tables, names)
    if set(table.role) != {"H"}:
        raise ValueError("LOGO evaluation requires H; P is diagnostic only")
    target = table.target_ttc.to_numpy()
    groups = table.group_id.astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three acquisition groups required")
    a, b = tables[base].prediction_ttc.to_numpy(), tables[special].prediction_ttc.to_numpy()
    x = features(a, b)
    relative = 100 * (abs(_phase(b) - _phase(target)) - abs(_phase(a) - _phase(target)))
    directory = output_dir if output_dir is not None else RUN / "router" / f"{base}__{special}"
    if not directory.resolve().is_relative_to(RUN.resolve()):
        raise ValueError("Router outputs must stay in the V19 artifact directory")
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {"R0": np.full(len(table), np.nan), "SHUFFLED": np.full(len(table), np.nan)}
    choices = {kind: np.zeros(len(table), dtype=bool) for kind in outputs}
    fold_records = []
    for index, group in enumerate(unique):
        train, test = groups != group, groups == group
        w = training_weights(target[train], groups[train])
        train_cost = relative[train]
        permutation = np.random.default_rng(7).permutation(train.sum())
        record = {
            "held_out_group": str(group),
            "training_groups": unique[unique != group].tolist(),
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "models": {},
        }
        for kind, costs in (("R0", train_cost), ("SHUFFLED", train_cost[permutation])):
            model = fit(x[train], costs, w)
            model.update(
                {
                    "base": base,
                    "special": special,
                    "held_out_group": str(group),
                    "training_groups": record["training_groups"],
                    "kind": kind,
                }
            )
            path = directory / f"fold_{index:03d}_{kind}.json"
            write_json(path, model)
            record["models"][kind] = {"path": path.name, "sha256": sha(path)}
            outputs[kind][test] = predict(model, a[test], b[test])
            choices[kind][test] = _choose_special(model, a[test], b[test])
        fold_records.append(record)
    final = fit(x, relative, training_weights(target, groups))
    final.update(
        {
            "base": base,
            "special": special,
            "training_groups": unique.tolist(),
            "kind": "R0_ALL_H_DEPLOYMENT_EXPORT_NOT_OOF",
            "H_in_sample": True,
        }
    )
    final_path = directory / "ALL_H_DEVELOPMENT_MODEL.json"
    write_json(final_path, final)
    references = {name: score(table, tables[name].prediction_ttc.to_numpy()) for name in names}
    results = {}
    bindings = {}
    for kind, prediction in outputs.items():
        _phase(prediction)
        if not ((prediction == a) | (prediction == b)).all():
            raise AssertionError("Router emitted a value outside its expert outputs")
        frame = table.copy()
        frame["prediction_ttc"] = prediction
        frame["raw_phase"] = np.where(
            choices[kind], tables[special].raw_phase.to_numpy(), tables[base].raw_phase.to_numpy()
        )
        frame["emitted_phase"] = _phase(prediction)
        frame["router_selected_special"] = choices[kind]
        path = directory / f"H_{kind}_OOF.parquet"
        frame.to_parquet(path, index=False)
        bindings[kind] = {"path": path.name, "sha256": sha(path)}
        candidate = score(table, prediction)
        results[kind] = {
            "metrics": candidate,
            "gates": {name: gate(metric, candidate) for name, metric in references.items()},
            "development_gate_A5": development_gate(references["A5"], candidate),
            "bootstrap": {
                name: _bootstrap(table, tables[name].prediction_ttc.to_numpy(), prediction)
                for name in names
            },
        }
    result = {
        "status": "DEVELOPMENT_LOGO_COMPLETE_NOT_PROMOTION",
        "pair": [base, special],
        "role": "H",
        "system_promotion": False,
        "official_confirmation": False,
        "scope": "H reused development; fixed producer ancestry must be admitted separately",
        "folds": fold_records,
        "predictions": bindings,
        "references": references,
        "results": results,
        "final_model_sha256": sha(final_path),
        "full_sensor_cost_measured": False,
        "both_expert_costs_required": True,
        "winner_selection": False,
    }
    write_json(directory / "RESULT.json", result)
    return result

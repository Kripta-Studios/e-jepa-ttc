"""The single prospectively fixed R1 descriptor extension; CPU development only."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import common, registration, router, sign_probe

CONFIG = common.ROOT / "configs/decision_recovery/feature_router_r1.json"
DIRECTORY = common.RUN / "feature_router_r1"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
R0_DIRECTORY = common.RUN / "router/NATIVE__FULL"
FEATURES = (*router.FEATURES, *sign_probe.FEATURE_NAMES)


def _sources() -> dict[str, str]:
    paths = [
        CONFIG,
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_feature_router.py",
        *registration.source_files(),
        Path(sign_probe.__file__),
        common.ROOT / "tests/unit/test_decision_recovery_sign_probe.py",
        common.ROOT / "operational/decision_recovery/training.py",
    ]
    return {p.relative_to(common.ROOT).as_posix(): common.sha(p) for p in paths}


def _config() -> dict[str, Any]:
    value = common.read_json(CONFIG)
    expected = {
        "schema": "v19_feature_router_r1_protocol_v1",
        "base": "NATIVE",
        "special": "FULL",
        "feature_names": list(FEATURES),
        "feature_count": 202,
        "descriptor_count": 194,
        "output_feature_count": 8,
        "ridge": 0.01,
        "relative_cost_divisor": 100.0,
        "threshold": 0.0,
        "tie": "base",
        "shuffled_seed": 7,
        "bootstrap_seed": 12345,
        "bootstrap_replicates": 2000,
        "band_masses": {b[0]: b[3] for b in router.BANDS},
        "descriptors_contain_targets": False,
        "ids_as_features": False,
        "encoder_updates": 0,
        "gpu_required": False,
        "hyperparameter_sweep": False,
        "feature_selection_from_H_probe_results": False,
        "V_opened": False,
        "system_promotion": False,
        "sota_claim": False,
        "one_extension_only": True,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("R1 fixed configuration differs")
    if value["sign_probe_registration_sha256"] != common.sha(sign_probe.REGISTRATION):
        raise ValueError("Descriptor source registration changed")
    return value


def _recovery_binding(probe: dict[str, Any]) -> str:
    """Bind the reviewed serialization-only repair while retaining original science."""
    sign_probe.verify_recovery_amendment(probe)
    return common.sha(sign_probe.AMENDMENT)


def register() -> dict[str, Any]:
    """Freeze executable sources before any H R1 fit, even if extraction is pending."""
    from .training import committed_sources

    if REGISTRATION.exists():
        return verify_registration()
    config = _config()
    base = registration.verify()
    probe = common.read_json(sign_probe.REGISTRATION)
    r0 = common.read_json(R0_DIRECTORY / "RESULT.json")
    decision = common.read_json(common.RUN / "router/DECISION.json")
    if decision.get("status") != "NO_PRIMARY_ROUTER_CANDIDATE":
        raise ValueError("Single R1 extension requires the registered R0 rejection")
    if (
        r0["pair"] != ["NATIVE", "FULL"]
        or r0["status"] != "DEVELOPMENT_LOGO_COMPLETE_NOT_PROMOTION"
    ):
        raise ValueError("Saved primary R0 is incomplete")
    sources = _sources()
    record = {
        "schema": "v19_feature_router_r1_registration_v1",
        "status": "REGISTERED_BEFORE_R1_H_FIT",
        "utc": datetime.now(UTC).isoformat(),
        "source_commit": committed_sources(sources),
        "sources": sources,
        "config": config,
        "config_sha256": common.sha(CONFIG),
        "base_registration_sha256": common.sha(registration.REGISTRATION),
        "sign_probe_registration_sha256": common.sha(sign_probe.REGISTRATION),
        "sign_probe_technical_recovery_sha256": _recovery_binding(probe),
        "expected_H_feature_binding": sign_probe._binding("H", probe),
        "expected_H_population": probe["populations"]["H"],
        "R0_result_sha256": common.sha(R0_DIRECTORY / "RESULT.json"),
        "R0_predictions_sha256": r0["predictions"]["R0"]["sha256"],
        "R0_decision_sha256": common.sha(common.RUN / "router/DECISION.json"),
        "P_diagnostics_sha256": common.sha(common.RUN / "diagnostics/P.json"),
        "ancestry": base["ancestry"],
        "authorization_scope": (
            "Separate single R1 extension authorization; probe diagnostic scope unchanged"
        ),
        "system_promotion": False,
        "official_SOTA": False,
    }
    if common.sha(R0_DIRECTORY / "H_R0_OOF.parquet") != record["R0_predictions_sha256"]:
        raise ValueError("Saved R0 predictions changed")
    common.write_json(REGISTRATION, record)
    return record


def verify_registration() -> dict[str, Any]:
    """Revalidate bytes without changing the historical commit or timestamp."""
    record = common.read_json(REGISTRATION)
    registration.verify()
    probe = common.read_json(sign_probe.REGISTRATION)
    expected = {
        "schema": "v19_feature_router_r1_registration_v1",
        "status": "REGISTERED_BEFORE_R1_H_FIT",
        "sources": _sources(),
        "config": _config(),
        "config_sha256": common.sha(CONFIG),
        "base_registration_sha256": common.sha(registration.REGISTRATION),
        "sign_probe_registration_sha256": common.sha(sign_probe.REGISTRATION),
        "sign_probe_technical_recovery_sha256": _recovery_binding(probe),
        "expected_H_feature_binding": sign_probe._binding("H", probe),
        "expected_H_population": probe["populations"]["H"],
        "R0_result_sha256": common.sha(R0_DIRECTORY / "RESULT.json"),
        "R0_predictions_sha256": common.sha(R0_DIRECTORY / "H_R0_OOF.parquet"),
        "R0_decision_sha256": common.sha(common.RUN / "router/DECISION.json"),
        "P_diagnostics_sha256": common.sha(common.RUN / "diagnostics/P.json"),
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Registered R1 evidence changed")
    return record


def features(a: np.ndarray, b: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    """Accept only two expert outputs and the ordered causal descriptor matrix."""
    output = router.features(a, b)
    d = np.asarray(descriptors)
    if d.shape != (len(output), 194) or not np.issubdtype(d.dtype, np.number):
        raise ValueError("R1 requires exactly 194 numeric descriptors per query")
    if not np.isfinite(d).all():
        raise ValueError("Nonfinite causal descriptors")
    return np.column_stack((output, d)).astype(np.float64)


def fit(x: np.ndarray, relative_cost: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """R0's unchanged weighted ridge algebra with its dimension fixed to 202."""
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, relative_cost, weights))
    if x.ndim != 2 or x.shape[1] != 202 or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError("R1 ridge shapes disagree")
    if not len(x) or not all(np.isfinite(v).all() for v in (x, y, w)):
        raise ValueError("R1 ridge inputs must be finite")
    if not (w > 0).all() or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Global training weights must sum to one")
    mean = w @ x
    scale = np.maximum(np.sqrt(w @ np.square(x - mean)), 1e-8)
    z = (x - mean) / scale
    intercept = float(w @ y)
    lhs = z.T @ (w[:, None] * z) + 0.01 * np.eye(202)
    rhs = z.T @ (w * (y - intercept))
    coefficient = np.linalg.solve(lhs, rhs)
    if not np.allclose(lhs @ coefficient, rhs, rtol=1e-10, atol=1e-10):
        raise ArithmeticError("R1 ridge solve failed")
    return {
        "schema": "relative_cost_feature_router_r1_v1",
        "features": list(FEATURES),
        "ridge": 0.01,
        "relative_cost_divisor": 100.0,
        "threshold": 0.0,
        "tie": "base",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": intercept,
        "base": "NATIVE",
        "special": "FULL",
        "scope": "DEVELOPMENT_MODEL_NOT_PROMOTION",
        "system_promotion": False,
    }


def choose_special(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Threshold zero, with ties assigned to NATIVE, exactly as R0."""
    expected = {
        "schema": "relative_cost_feature_router_r1_v1",
        "features": list(FEATURES),
        "ridge": 0.01,
        "threshold": 0.0,
        "tie": "base",
        "relative_cost_divisor": 100.0,
        "base": "NATIVE",
        "special": "FULL",
    }
    if any(model.get(k) != v for k, v in expected.items()):
        raise ValueError("R1 model contract differs")
    x = np.asarray(x, dtype=np.float64)
    params = [np.asarray(model[k], dtype=np.float64) for k in ("mean", "scale", "coefficient")]
    if x.ndim != 2 or x.shape[1] != 202 or not np.isfinite(x).all():
        raise ValueError("Invalid R1 prediction matrix")
    if any(p.shape != (202,) or not np.isfinite(p).all() for p in params):
        raise ValueError("Invalid R1 model parameters")
    mean, scale, coefficient = params
    if not (scale > 0).all() or not np.isfinite(model["intercept"]):
        raise ValueError("Invalid R1 scale/intercept")
    regret = ((x - mean) / scale) @ coefficient + model["intercept"]
    if not np.isfinite(regret).all():
        raise ValueError("Nonfinite R1 predicted regret")
    return regret < 0


def _same_folds(table: pd.DataFrame, r0: dict[str, Any]) -> list[dict[str, Any]]:
    groups = table.group_id.astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 3 or set(table.role) != {"H"}:
        raise ValueError("R1 requires H and at least three groups")
    folds = [
        {
            "held_out_group": str(g),
            "training_groups": unique[unique != g].tolist(),
            "train_rows": int((groups != g).sum()),
            "test_rows": int((groups == g).sum()),
        }
        for g in unique
    ]
    saved = [{k: f[k] for k in folds[0]} for f in r0["folds"]]
    if saved != folds:
        raise ValueError("R1 folds differ from saved R0")
    return folds


def _diagnostics(table: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    target = table.target_ttc.to_numpy()
    mid = 10000 * abs(router._phase(prediction) - router._phase(target))

    def cell(mask: np.ndarray) -> dict[str, Any]:
        negative = mask & (target < 0)
        return {
            "rows": int(mask.sum()),
            "negative_rows": int(negative.sum()),
            "mae_s": float(abs(prediction[mask] - target[mask]).mean()),
            "sign_error_rate": float((np.sign(prediction[mask]) != np.sign(target[mask])).mean()),
            "negative_sign_error_rate": float((prediction[negative] > 0).mean())
            if negative.any()
            else None,
            "negative_mae_s": float(abs(prediction[negative] - target[negative]).mean())
            if negative.any()
            else None,
            "band_mean_MiD": {
                name: float(mid[cell].mean()) if cell.any() else None
                for name, lower, upper, _ in router.BANDS
                for cell in [mask & (target > lower) & (target <= upper)]
            },
        }

    return {
        "all": cell(np.ones(len(table), dtype=bool)),
        "groups": {
            str(g): cell(table.group_id.astype(str).to_numpy() == str(g))
            for g in sorted(table.group_id.unique())
        },
    }


def evaluate(
    tables: dict[str, pd.DataFrame],
    descriptors: np.ndarray,
    r0: dict[str, Any],
    r0_predictions: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Pure CPU bounded evaluation; production caller must admit registration/materialization."""
    from operational.range_transfer.analyze import gate

    if not output_dir.resolve().is_relative_to(common.RUN.resolve()):
        raise ValueError("R1 outputs must stay in V19 artifacts")
    tables = {**tables, "R0": r0_predictions}
    names = ["NATIVE", "FULL", "A5", "R0"]
    table = router._aligned(tables, names)
    folds = _same_folds(table, r0)
    a, b = (tables[n].prediction_ttc.to_numpy() for n in ("NATIVE", "FULL"))
    target, groups = table.target_ttc.to_numpy(), table.group_id.astype(str).to_numpy()
    x = features(a, b, descriptors)
    relative = 100 * (
        abs(router._phase(b) - router._phase(target))
        - abs(router._phase(a) - router._phase(target))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {k: np.full(len(table), np.nan) for k in ("R1", "SHUFFLED_R1")}
    choices = {k: np.zeros(len(table), dtype=bool) for k in outputs}
    for index, fold in enumerate(folds):
        train, test = groups != fold["held_out_group"], groups == fold["held_out_group"]
        w, costs = router.training_weights(target[train], groups[train]), relative[train]
        permutation = np.random.default_rng(7).permutation(train.sum())
        fold["models"] = {}
        for kind, y in (("R1", costs), ("SHUFFLED_R1", costs[permutation])):
            model = fit(x[train], y, w)
            model.update({k: v for k, v in fold.items() if k != "models"})
            model["kind"] = kind
            path = output_dir / f"fold_{index:03d}_{kind}.json"
            common.write_json(path, model)
            fold["models"][kind] = {"path": path.name, "sha256": common.sha(path)}
            chosen = choose_special(model, x[test])
            outputs[kind][test], choices[kind][test] = np.where(chosen, b[test], a[test]), chosen
    final = fit(x, relative, router.training_weights(target, groups))
    final.update(
        {
            "kind": "R1_ALL_H_DEVELOPMENT_EXPORT_NOT_OOF",
            "H_in_sample": True,
            "training_groups": sorted(np.unique(groups).tolist()),
        }
    )
    final_path = output_dir / "ALL_H_DEVELOPMENT_MODEL.json"
    common.write_json(final_path, final)
    references = {
        name: common.score(table, tables[name].prediction_ttc.to_numpy()) for name in names
    }
    reference_vectors = {name: tables[name].prediction_ttc.to_numpy() for name in names}
    results, bindings = {}, {}
    for kind, prediction in outputs.items():
        router._phase(prediction)
        if not ((prediction == a) | (prediction == b)).all():
            raise AssertionError("R1 emitted values outside its experts")
        frame = table.copy()
        frame["prediction_ttc"] = prediction
        frame["raw_phase"] = np.where(
            choices[kind], tables["FULL"].raw_phase, tables["NATIVE"].raw_phase
        )
        frame["emitted_phase"], frame["router_selected_special"] = (
            router._phase(prediction),
            choices[kind],
        )
        path = output_dir / f"H_{kind}_OOF.parquet"
        frame.to_parquet(path, index=False)
        bindings[kind] = {"path": path.name, "sha256": common.sha(path)}
        metrics = common.score(table, prediction)
        comparisons = {
            **reference_vectors,
            **({"SHUFFLED_R1": outputs["SHUFFLED_R1"]} if kind == "R1" else {}),
        }
        results[kind] = {
            "metrics": metrics,
            "gates": {n: gate(m, metrics) for n, m in references.items()},
            "development_gate_A5": router.development_gate(references["A5"], metrics),
            "diagnostics": _diagnostics(table, prediction),
            "bootstrap": {
                n: router._bootstrap(table, p, prediction) for n, p in comparisons.items()
            },
        }
    candidate = results["R1"]
    metric, a5 = candidate["metrics"], references["A5"]
    dev = candidate["development_gate_A5"]
    overall = metric["overall_MiD"]
    checks = {
        "A5_development_gate": dev["eligible"],
        "A5_coverage": metric["complete_prediction_and_mid_coverage"],
        "A5_bands": dev["checks"].get("bands", False),
        "A5_sign": metric["diagnostics"]["sign_error_rate"]
        <= a5["diagnostics"]["sign_error_rate"] + 0.005,
        "base_original_system_numerical_gates": candidate["gates"]["NATIVE"]["eligible"],
        "beats_best_fixed_aggregate": overall is not None
        and all(
            references[n]["overall_MiD"] is not None and overall < references[n]["overall_MiD"]
            for n in ("NATIVE", "FULL")
        ),
        "beats_shuffled_aggregate": overall is not None
        and results["SHUFFLED_R1"]["metrics"]["overall_MiD"] is not None
        and overall < results["SHUFFLED_R1"]["metrics"]["overall_MiD"],
        "beats_saved_R0_aggregate": overall is not None
        and references["R0"]["overall_MiD"] is not None
        and overall < references["R0"]["overall_MiD"],
    }
    result = {
        "status": "DEVELOPMENT_LOGO_COMPLETE_NOT_PROMOTION",
        "pair": ["NATIVE", "FULL"],
        "role": "H",
        "feature_count": 202,
        "folds": folds,
        "predictions": bindings,
        "references": references,
        "reference_diagnostics": {n: _diagnostics(table, p) for n, p in reference_vectors.items()},
        "results": results,
        "final_model_sha256": common.sha(final_path),
        "advancement": {"checks": checks, "further_evaluation_enabled": all(checks.values())},
        "system_promotion": False,
        "official_SOTA": False,
        "V_opened": False,
        "full_sensor_cost_measured": False,
        "both_expert_costs_required": True,
        "scope": (
            "Single fixed R1 extension on reused H; fixed-fold bootstrap is descriptive; "
            "no independent confirmation"
        ),
    }
    common.write_json(output_dir / "RESULT.json", result)
    return result


def run() -> dict[str, Any]:
    """Admit frozen descriptor receipts and exact R0 population before the only R1 fit."""
    record = verify_registration()
    if (DIRECTORY / "COMPLETE.json").exists():
        complete = common.read_json(DIRECTORY / "COMPLETE.json")
        if complete.get("status") != "COMPLETE":
            raise ValueError("R1 completion status differs")
        for name, expected in complete["files"].items():
            if common.sha(DIRECTORY / name) != expected:
                raise ValueError("Completed R1 evidence changed")
        materialized = common.read_json(DIRECTORY / "MATERIALIZATION.json")
        source = sign_probe.DIRECTORY / "features/H/features.npz"
        if (
            materialized["features_sha256"] != common.sha(source)
            or materialized["complete_sha256"] != common.sha(source.with_name("COMPLETE.json"))
            or materialized["registration_sha256"] != common.sha(REGISTRATION)
        ):
            raise ValueError("Materialized R1 descriptor evidence changed")
        return common.read_json(DIRECTORY / "RESULT.json")
    if (DIRECTORY / "FIT_STARTED.json").exists():
        raise ValueError("Interrupted R1 fit requires explicit audit; refusing silent refit")
    tables = common.load_population("H", ["NATIVE", "FULL", "A5"])
    receipt = sign_probe._admit_complete_features(
        "H", record["expected_H_feature_binding"], tables["FULL"]
    )
    source = sign_probe.DIRECTORY / "features/H/features.npz"
    with np.load(source, allow_pickle=False) as stored:
        descriptors = stored["features"].copy()
    materialization = {
        "status": "ADMITTED",
        "feature_binding": record["expected_H_feature_binding"],
        "features_sha256": common.sha(source),
        "complete_sha256": common.sha(source.with_name("COMPLETE.json")),
        "rows": receipt["rows"],
        "contains_targets": False,
        "registration_sha256": common.sha(REGISTRATION),
    }
    common.write_json(DIRECTORY / "MATERIALIZATION.json", materialization)
    r0 = common.read_json(R0_DIRECTORY / "RESULT.json")
    predictions = pd.read_parquet(R0_DIRECTORY / "H_R0_OOF.parquet")
    router._aligned({**tables, "R0": predictions}, ["NATIVE", "FULL", "A5", "R0"])
    _same_folds(tables["NATIVE"], r0)
    common.write_json(
        DIRECTORY / "FIT_STARTED.json",
        {"utc": datetime.now(UTC).isoformat(), "registration_sha256": common.sha(REGISTRATION)},
    )
    result = evaluate(tables, descriptors, r0, predictions, output_dir=DIRECTORY)
    result.update(
        {
            "registration_sha256": common.sha(REGISTRATION),
            "materialization": materialization,
            "ancestry": record["ancestry"],
            "R0_result_sha256": record["R0_result_sha256"],
        }
    )
    common.write_json(DIRECTORY / "RESULT.json", result)
    common.write_json(
        DIRECTORY / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "files": {
                p.name: common.sha(p)
                for p in DIRECTORY.iterdir()
                if p.is_file() and p.name != "COMPLETE.json"
            },
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    arguments = parser.parse_args()
    answer = register() if arguments.action == "register" else run()
    print(json.dumps({"action": arguments.action, "status": answer["status"]}))

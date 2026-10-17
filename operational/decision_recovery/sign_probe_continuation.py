"""Registered CPU-only continuation of the frozen FULL sign probe.

The original 2,000-iteration LBFGS fit remains a recorded failed attempt.  This
continuation changes only ``max_iter`` to 10,000, fits on P, and scores reused
H only after both the real-label probe and its frozen shuffled-label control
have converged.  It performs no encoder forward or optimizer update.
"""

from __future__ import annotations

import json
import os
import platform
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import common, sign_probe, training

DIRECTORY = common.RUN / "sign_probe_v2_continuation"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
FAILURE_RECEIPT = DIRECTORY / "ORIGINAL_FAILURE.json"
ERROR_LOG = common.RUN / "SIGN_PROBE_V2_RECOVERY_ERROR.log"
MAX_ITER = 10_000
CONTRACT = {**sign_probe.LOGISTIC, "max_iter": MAX_ITER}


class FitLimitReachedError(RuntimeError):
    """The sole registered continuation limit was reached without convergence."""

    def __init__(self, iterations: int, convergence_warnings: list[str]) -> None:
        super().__init__("Continued sign probe reached 10000 iterations")
        self.iterations = iterations
        self.convergence_warnings = convergence_warnings


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_sign_probe_continuation.py",
        common.ROOT / "operational/decision_recovery/sign_probe.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def _software() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _failure() -> dict[str, Any]:
    if not ERROR_LOG.is_file():
        raise ValueError("Preserved 2,000-iteration failure log is missing")
    text = ERROR_LOG.read_text(encoding="utf-8", errors="replace")
    required = (
        "lbfgs failed to converge after 2000 iteration(s)",
        "TOTAL NO. OF ITERATIONS REACHED LIMIT",
        'RuntimeError("Fixed sign probe did not converge")',
    )
    if any(value not in text for value in required):
        raise ValueError("Original convergence failure evidence changed")
    forbidden = [
        sign_probe.DIRECTORY / "RESULT.json",
        sign_probe.DIRECTORY / "COMPLETE.json",
        sign_probe.DIRECTORY / "H_SIGN_PREDICTIONS.parquet",
        sign_probe.DIRECTORY / "models/S1.json",
        sign_probe.DIRECTORY / "models/SHUFFLED.json",
    ]
    if any(path.exists() for path in forbidden):
        raise ValueError("Original failed fit unexpectedly created scientific outputs")
    return {
        "schema": "v19_sign_probe_original_fit_failure_v1",
        "status": "FAILED_CLOSED_AT_REGISTERED_ITERATION_LIMIT",
        "error_log": {
            "path": ERROR_LOG.relative_to(common.RUN).as_posix(),
            "sha256": common.sha(ERROR_LOG),
        },
        "attempt": {
            "C": sign_probe.LOGISTIC["C"],
            "solver": sign_probe.LOGISTIC["solver"],
            "tol": sign_probe.LOGISTIC["tol"],
            "max_iter": sign_probe.LOGISTIC["max_iter"],
            "observed_iterations": sign_probe.LOGISTIC["max_iter"],
        },
        "outputs_created": False,
        "H_scores_created": False,
        "encoder_optimizer_updates": 0,
    }


def _feature_binding(role: str) -> dict[str, Any]:
    parent = sign_probe.DIRECTORY / "features" / role
    rows = common.load_population(role, ["FULL"])["FULL"]
    receipt = sign_probe._admit_complete_features(
        role, sign_probe._binding(role, common.read_json(sign_probe.REGISTRATION)), rows
    )
    return {
        "role": role,
        "rows": len(rows),
        "complete_sha256": common.sha(parent / "COMPLETE.json"),
        "features_sha256": common.sha(parent / "features.npz"),
        "binding": receipt["binding"],
        "parity": receipt["parity"],
        "contains_targets": receipt["contains_targets"],
        "encoder_optimizer_updates": receipt["encoder_optimizer_updates"],
    }


def prepare_registration() -> dict[str, Any]:
    """Freeze the sole iteration-limit continuation before any new CPU fit."""
    sign_record = sign_probe.verify_registration()
    sign_probe.verify_recovery_amendment(sign_record)
    if (sign_probe.DIRECTORY / "RESULT.json").exists():
        raise ValueError("Original probe unexpectedly produced a result")
    sources = _sources()
    source_commit = training.committed_sources(sources)
    failure = _failure()
    if FAILURE_RECEIPT.exists():
        if common.read_json(FAILURE_RECEIPT) != failure:
            raise ValueError("Original failure receipt changed")
    else:
        common.write_json(FAILURE_RECEIPT, failure)
    record = {
        "schema": "v19_sign_probe_cpu_continuation_v1",
        "status": "REGISTERED_BEFORE_CONTINUATION_FIT",
        "utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "sources": sources,
        "software": _software(),
        "original_registration_sha256": common.sha(sign_probe.REGISTRATION),
        "technical_recovery_sha256": common.sha(sign_probe.AMENDMENT),
        "original_failure_sha256": common.sha(FAILURE_RECEIPT),
        "features": {role: _feature_binding(role) for role in ("P", "H")},
        "change": {
            "only_change": "logistic max_iter 2000 -> 10000",
            "original_contract": sign_probe.LOGISTIC,
            "continuation_contract": CONTRACT,
            "reason": "registered limit reached with sklearn status=1",
            "selection_used_H": False,
            "further_escalation_authorized": False,
        },
        "execution_order": [
            "revalidate frozen P/H feature cache identities without fitting or scoring",
            "fit S1 on complete P",
            "fit SHUFFLED on the same complete P",
            "require both to converge below 10000",
            "then and only then compute H scores and evaluate once",
        ],
        "interpretation_gate": sign_record["interpretation_gate"],
        "interpretation": sign_record["interpretation"],
        "flags": sign_record["flags"],
    }
    if REGISTRATION.exists():
        return registered()
    common.write_json(REGISTRATION, record)
    return record


def registered() -> dict[str, Any]:
    """Replay every frozen continuation input without fitting a model."""
    record = common.read_json(REGISTRATION)
    sign_record = sign_probe.verify_registration()
    sign_probe.verify_recovery_amendment(sign_record)
    training.committed_sources(record["sources"])
    if (
        record.get("schema") != "v19_sign_probe_cpu_continuation_v1"
        or record.get("status") != "REGISTERED_BEFORE_CONTINUATION_FIT"
        or record.get("sources") != _sources()
        or record.get("software") != _software()
        or record.get("original_registration_sha256") != common.sha(sign_probe.REGISTRATION)
        or record.get("technical_recovery_sha256") != common.sha(sign_probe.AMENDMENT)
        or record.get("original_failure_sha256") != common.sha(FAILURE_RECEIPT)
        or common.read_json(FAILURE_RECEIPT) != _failure()
        or record.get("features") != {role: _feature_binding(role) for role in ("P", "H")}
        or record.get("change", {}).get("original_contract") != sign_probe.LOGISTIC
        or record.get("change", {}).get("continuation_contract") != CONTRACT
        or record.get("change", {}).get("selection_used_H") is not False
        or record.get("change", {}).get("further_escalation_authorized") is not False
        or record.get("interpretation_gate") != sign_record["interpretation_gate"]
        or record.get("interpretation") != sign_record["interpretation"]
        or record.get("flags") != sign_record["flags"]
    ):
        raise ValueError("Registered sign-probe continuation changed")
    return record


def _fit_logistic(x: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    """Fit the registered continuation and fail closed at its sole new limit."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    if (
        x.ndim != 2
        or x.shape[1] != sign_probe.FEATURE_COUNT
        or label.shape != (len(x),)
        or not np.isfinite(x).all()
        or set(np.unique(label)) != {0, 1}
    ):
        raise ValueError("Complete finite binary P continuation inputs required")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(
        C=CONTRACT["C"],
        class_weight=CONTRACT["class_weight"],
        solver=CONTRACT["solver"],
        max_iter=CONTRACT["max_iter"],
        tol=CONTRACT["tol"],
        fit_intercept=CONTRACT["fit_intercept"],
        random_state=sign_probe.SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaler.transform(x), label)
    convergence_warnings = [
        str(item.message) for item in caught if item.category is ConvergenceWarning
    ]
    iterations = int(classifier.n_iter_[0])
    if convergence_warnings or iterations >= MAX_ITER:
        raise FitLimitReachedError(iterations, convergence_warnings)
    return {
        "schema": "v19_full_sign_logistic_continuation_v1",
        "features": list(sign_probe.FEATURE_NAMES),
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefficient": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "logistic_fit_iterations": iterations,
        "contract": CONTRACT,
        "prior_failed_limit": sign_probe.LOGISTIC["max_iter"],
        "fit_rows": len(x),
        "fit_positive": int(label.sum()),
        "fit_negative": int((label == 0).sum()),
        "fit_role": "P",
        "system_promotion": False,
        "encoder_optimizer_updates": 0,
    }


def _predict_probability(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    if (
        model.get("schema") != "v19_full_sign_logistic_continuation_v1"
        or model.get("contract") != CONTRACT
    ):
        raise ValueError("Continued sign-probe model contract changed")
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if (
        x.shape != (len(x), sign_probe.FEATURE_COUNT)
        or any(value.shape != (sign_probe.FEATURE_COUNT,) for value in (mean, scale, coefficient))
        or not all(np.isfinite(value).all() for value in (x, mean, scale, coefficient))
        or not (scale > 0).all()
    ):
        raise ValueError("Invalid continued sign-probe model")
    probability = expit(((x - mean) / scale) @ coefficient + model["intercept"])
    if probability.shape != (len(x),) or not np.isfinite(probability).all():
        raise ValueError("Continued sign probe emitted invalid probabilities")
    return probability


def _result_receipt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "registration_sha256": common.sha(REGISTRATION),
        "result_sha256": common.sha(DIRECTORY / "RESULT.json"),
        "encoder_optimizer_updates": 0,
    }


def _admit_result(record: dict[str, Any]) -> dict[str, Any]:
    result = common.read_json(DIRECTORY / "RESULT.json")
    if (
        result.get("schema") != "v19_sign_probe_cpu_continuation_result_v1"
        or result.get("status") != "COMPLETE_DIAGNOSTIC_NOT_CANDIDATE"
        or result.get("registration_sha256") != common.sha(REGISTRATION)
        or result.get("original_failure_sha256") != record["original_failure_sha256"]
        or result.get("encoder_optimizer_updates") != 0
        or result.get("system_promotion") is not False
        or result.get("official_confirmation") is not False
        or result.get("SOTA_claim") is not False
        or result.get("models", {}).get("S1") != common.sha(DIRECTORY / "models/S1.json")
        or result.get("models", {}).get("SHUFFLED")
        != common.sha(DIRECTORY / "models/SHUFFLED.json")
        or result.get("prediction_sha256") != common.sha(DIRECTORY / "H_SIGN_PREDICTIONS.parquet")
        or common.read_json(DIRECTORY / "COMPLETE.json") != _result_receipt(result)
    ):
        raise ValueError("Completed continued sign-probe result changed")
    return result


def _write_limit_failure(
    failed_fit: str,
    error: FitLimitReachedError,
    completed_fit_iterations: dict[str, int],
) -> dict[str, Any]:
    failure = {
        "schema": "v19_sign_probe_cpu_continuation_failure_v1",
        "status": "FAILED_CLOSED_AT_CONTINUATION_LIMIT",
        "registration_sha256": common.sha(REGISTRATION),
        "prior_failed_limit": sign_probe.LOGISTIC["max_iter"],
        "continuation_limit": MAX_ITER,
        "failed_fit": failed_fit,
        "observed_iterations": error.iterations,
        "completed_fit_iterations": completed_fit_iterations,
        "convergence_warnings": error.convergence_warnings,
        "H_cache_bound_during_registration": True,
        "H_evaluation_started": False,
        "H_used_for_fit_or_selection": False,
        "H_scores_created": False,
        "further_escalation_authorized": False,
        "encoder_optimizer_updates": 0,
    }
    common.write_json(DIRECTORY / "CONTINUATION_FAILURE.json", failure)
    return failure


def _run_registered() -> dict[str, Any]:
    record = registered()
    continuation_failure = DIRECTORY / "CONTINUATION_FAILURE.json"
    if continuation_failure.exists():
        failure = common.read_json(continuation_failure)
        if (
            failure.get("status") != "FAILED_CLOSED_AT_CONTINUATION_LIMIT"
            or failure.get("registration_sha256") != common.sha(REGISTRATION)
            or failure.get("continuation_limit") != MAX_ITER
            or failure.get("failed_fit") not in {"S1", "SHUFFLED"}
            or failure.get("observed_iterations") != MAX_ITER
            or not isinstance(failure.get("completed_fit_iterations"), dict)
            or any(
                not isinstance(value, int) or not 0 < value < MAX_ITER
                for value in failure.get("completed_fit_iterations", {}).values()
            )
            or failure.get("H_cache_bound_during_registration") is not True
            or failure.get("H_evaluation_started") is not False
            or failure.get("H_used_for_fit_or_selection") is not False
            or failure.get("H_scores_created") is not False
            or failure.get("further_escalation_authorized") is not False
            or failure.get("encoder_optimizer_updates") != 0
        ):
            raise ValueError("Continuation failure receipt changed")
        return failure
    if (DIRECTORY / "RESULT.json").exists():
        return _admit_result(record)
    if (DIRECTORY / "COMPLETE.json").exists():
        raise ValueError("Continuation receipt exists without its result")

    # Registration revalidates the frozen H cache identity. No H score is
    # computed and no H row participates in fitting or selection before both
    # P-only fits have converged.
    p = sign_probe._load_features("P", common.read_json(sign_probe.REGISTRATION))
    p_table = common.load_population("P", ["FULL"])["FULL"]
    p_label = (p_table.target_ttc.to_numpy() > 0).astype(np.int64)
    completed_fit_iterations = {}
    try:
        model = _fit_logistic(p["features"], p_label)
    except FitLimitReachedError as error:
        return _write_limit_failure("S1", error, completed_fit_iterations)
    completed_fit_iterations["S1"] = model["logistic_fit_iterations"]
    shuffled_label = p_label[np.random.default_rng(sign_probe.SEED).permutation(len(p_label))]
    try:
        shuffled = _fit_logistic(p["features"], shuffled_label)
    except FitLimitReachedError as error:
        return _write_limit_failure("SHUFFLED", error, completed_fit_iterations)
    completed_fit_iterations["SHUFFLED"] = shuffled["logistic_fit_iterations"]

    h = sign_probe._load_features("H", common.read_json(sign_probe.REGISTRATION))
    h_table = common.load_population("H", ["FULL"])["FULL"]
    h_label = (h_table.target_ttc.to_numpy() > 0).astype(np.int64)
    probability = _predict_probability(model, h["features"])
    shuffled_probability = _predict_probability(shuffled, h["features"])
    baseline_hard = h["prediction_ttc"] > 0
    candidate_hard = probability >= CONTRACT["threshold"]
    baseline = sign_probe._errors(h_label.astype(bool), baseline_hard)
    candidate = sign_probe._metrics(h_label, probability)
    negative_control = sign_probe._metrics(h_label, shuffled_probability)
    bootstrap = sign_probe._bootstrap_balanced_error(
        h_label.astype(bool), baseline_hard, candidate_hard, h_table.group_id.to_numpy()
    )
    p_prevalence = float(p_label.mean())
    constant_brier = float(np.mean(np.square(h_label - p_prevalence)))
    checks = {
        "negative_error": candidate["negative_error"] <= 0.75 * baseline["negative_error"],
        "positive_error": candidate["positive_error"] <= baseline["positive_error"] + 0.005,
        "balanced_error_ci": bootstrap["ci95_if_all_defined"] is not None
        and bootstrap["ci95_if_all_defined"][1] < 0.0,
        "brier": candidate["brier"] < constant_brier,
        "shuffled": candidate["balanced_error"] < negative_control["balanced_error"],
    }
    model["training_population"] = {
        "role": "P",
        "tokens_sha256": common.read_json(sign_probe.REGISTRATION)["populations"]["P"][
            "ordered_tokens_sha256"
        ],
    }
    shuffled.update(
        training_population=model["training_population"],
        label_permutation_seed=sign_probe.SEED,
    )
    common.write_json(DIRECTORY / "models/S1.json", model)
    common.write_json(DIRECTORY / "models/SHUFFLED.json", shuffled)
    frame = h_table[common.IDENTITY].copy()
    frame["full_emitted_sign_positive"] = baseline_hard
    frame["sign_probability_positive"] = probability
    frame["sign_prediction_positive"] = candidate_hard
    frame["shuffled_probability_positive"] = shuffled_probability
    frame["shuffled_prediction_positive"] = shuffled_probability >= CONTRACT["threshold"]
    prediction_path = DIRECTORY / "H_SIGN_PREDICTIONS.parquet"
    pending = prediction_path.with_suffix(".pending")
    frame.to_parquet(pending, index=False)
    os.replace(pending, prediction_path)
    result = {
        "schema": "v19_sign_probe_cpu_continuation_result_v1",
        "status": "COMPLETE_DIAGNOSTIC_NOT_CANDIDATE",
        "registration_sha256": common.sha(REGISTRATION),
        "original_failure_sha256": record["original_failure_sha256"],
        "models": {
            "S1": common.sha(DIRECTORY / "models/S1.json"),
            "SHUFFLED": common.sha(DIRECTORY / "models/SHUFFLED.json"),
        },
        "prediction_sha256": common.sha(prediction_path),
        "fit": {
            "role": "P",
            "rows": len(p_label),
            "positive": int(p_label.sum()),
            "negative": int((p_label == 0).sum()),
            "prevalence": p_prevalence,
            "prior_failed_limit": sign_probe.LOGISTIC["max_iter"],
            "continuation_limit": MAX_ITER,
            "iterations": {
                "S1": model["logistic_fit_iterations"],
                "SHUFFLED": shuffled["logistic_fit_iterations"],
            },
        },
        "evaluation": {
            "role": "H_REUSED_DEVELOPMENT",
            "rows": len(h_label),
            "FULL_emitted_sign": baseline,
            "S1": candidate,
            "SHUFFLED": negative_control,
            "P_prevalence_constant_brier": constant_brier,
            "FULL_by_group": sign_probe._group_errors(
                h_label.astype(bool), baseline_hard, h_table.group_id.to_numpy()
            ),
            "S1_by_group": sign_probe._group_metrics(
                h_label, probability, h_table.group_id.to_numpy()
            ),
            "SHUFFLED_by_group": sign_probe._group_metrics(
                h_label, shuffled_probability, h_table.group_id.to_numpy()
            ),
            "S1_vs_FULL_contingency": sign_probe._contingency(
                h_label.astype(bool), baseline_hard, candidate_hard
            ),
            "bootstrap": bootstrap,
        },
        "interpretation_gate": {"eligible": all(checks.values()), "checks": checks},
        "interpretation": record["interpretation"],
        **record["flags"],
    }
    common.write_json(DIRECTORY / "RESULT.json", result)
    common.write_json(DIRECTORY / "COMPLETE.json", _result_receipt(result))
    return result


def run() -> dict[str, Any]:
    """Run only an already registered continuation under the shared CPU lock."""
    if not REGISTRATION.is_file():
        raise ValueError("Register the continuation before fitting")
    with training.file_lock(DIRECTORY / "DRIVER.lock"), training.runtime_context():
        return _run_registered()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    arguments = parser.parse_args()
    value = prepare_registration() if arguments.action == "register" else run()
    print(json.dumps(value, indent=2, sort_keys=True))

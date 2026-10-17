"""Frozen A1-FULL sign-information probe on causal spatial descriptors.

This is a diagnostic-only amendment to V19.  It performs no encoder updates,
does not alter a TTC prediction and cannot promote a system.  A fixed logistic
probe is fit on complete P descriptors and evaluated once on reused H
development data.  Target sign is accepted only by the CPU fitter, never by
the descriptor extractor.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import common, registration, training

if TYPE_CHECKING:
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement

DIRECTORY = common.RUN / "sign_probe"
REGISTRATION = DIRECTORY / "REGISTRATION.json"
ENDPOINT = common.SOURCE_RUN / "adaptive/A1_adaptation_depth/fits/A1-FULL-S7/checkpoint_last.pt"
FEATURE_NAMES = (
    *(f"spatial_mean_{index:03d}" for index in range(96)),
    *(f"spatial_max_{index:03d}" for index in range(96)),
    "native_delta_t_s",
    "log_native_delta_t_s",
)
FEATURE_COUNT = 194
POPULATIONS = {"P": 53_890, "H": 18_422}
BATCH = 4
CHUNK = 256
SEED = 7
BOOTSTRAP_SEED = 12_345
BOOTSTRAP_REPLICATES = 2_000
LOGISTIC = {
    "C": 1.0,
    "class_weight": None,
    "solver": "lbfgs",
    "max_iter": 2_000,
    "tol": 1.0e-10,
    "fit_intercept": True,
    "threshold": 0.5,
}
PARITY = {
    "historical_batch": BATCH,
    "exact_expected": True,
    "fallback_atol": 1.0e-5,
    "fallback_rtol": 0.0,
    "fallback_requires_zero_raw_sign_flips": True,
    "fallback_requires_zero_ttc_sign_flips": True,
}


def _contracts() -> dict[str, Any]:
    """Return every scientific choice that must remain frozen after registration."""
    from operational.decision_recovery.sensor_runtime import runtime_contract

    return {
        "features": {
            "count": FEATURE_COUNT,
            "names": list(FEATURE_NAMES),
            "causal_inputs": "native event P2 pixels and native positive delta only",
            "schema": "mean96 + max96 from FULL spatial tower + dt + log(dt)",
            "targets_in_cache": False,
        },
        "extract": {
            "batch": BATCH,
            "chunk": CHUNK,
            "device": "CUDA_FP32_no_TF32",
            "runtime": runtime_contract()["evaluation"],
            "raw_phase_and_ttc": "parity fields only; excluded from probe features",
            "parity": PARITY,
            "gpu_scope_seconds": 900,
        },
        "probe": {
            **LOGISTIC,
            "seed": SEED,
            "normalizer": "StandardScaler fit on complete P only",
            "fit_role": "P",
            "evaluation_role": "H",
            "negative_control": "Permute P labels once with numpy seed7",
            "configuration_selected_using_H": False,
            "label": "1[target_ttc > 0]",
        },
        "metrics": {
            "ece_bins": 10,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "acquisition_group",
            "group_reports": ["FULL_emitted_sign", "S1", "SHUFFLED"],
            "contingency": "rescues_and_damages_by_true_sign",
        },
        "interpretation_gate": {
            "negative_error_ratio_lte": 0.75,
            "positive_error_increase_lte": 0.005,
            "balanced_error_delta_ci95_upper_lt": 0.0,
            "brier_better_than_P_prevalence_constant": True,
            "balanced_error_lt_shuffled": True,
            "meaning": "supports later registered head comparison; never promotion",
        },
        "interpretation": (
            "Passing supports a separately registered head experiment only. Failure means "
            "this fixed linear readout did not recover sign; neither outcome is a TTC candidate."
        ),
        "flags": {
            "system_promotion": False,
            "official_confirmation": False,
            "SOTA_claim": False,
            "encoder_optimizer_updates": 0,
        },
    }


def _software() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_sign_probe.py",
        common.ROOT / "operational/decision_recovery/common.py",
        common.ROOT / "operational/decision_recovery/registration.py",
        common.ROOT / "operational/decision_recovery/training.py",
        common.ROOT / "operational/decision_recovery/sensor_runtime.py",
        common.ROOT / "operational/joint_measurement/geometry_model.py",
        common.ROOT / "operational/joint_measurement/fusion_pair_extract.py",
        common.ROOT / "src/e_jepa_ttc/joint_measurement/model.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def _endpoint_binding() -> dict[str, Any]:
    fit = ENDPOINT.parent
    receipt = common.read_json(fit / "CHECKPOINT_RECEIPT.json")
    identity = common.read_json(fit / "IDENTITY.json")
    provenance = common.read_json(fit / "RUN_PROVENANCE.json")
    verification = common.read_json(
        common.SOURCE_RUN / "adaptive/A1_adaptation_depth/verification/A1-FULL-S7/ENDPOINT.json"
    )
    checkpoint_sha = common.sha(ENDPOINT)
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != 12_000
        or receipt["checkpoint_sha256"] != checkpoint_sha
        or verification["status"] != "PASS"
        or verification["checkpoint_sha256"] != checkpoint_sha
        or identity["fit_id"] != "A1-FULL-S7"
        or provenance["run_name"] != "A1-FULL-S7"
        or provenance["split_version"] != "native P/H/V"
    ):
        raise ValueError("Historical A1-FULL endpoint is incomplete or changed")
    return {
        "fit_id": "A1-FULL-S7",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_receipt_sha256": common.sha(fit / "CHECKPOINT_RECEIPT.json"),
        "identity_sha256": common.sha(fit / "IDENTITY.json"),
        "provenance_sha256": common.sha(fit / "RUN_PROVENANCE.json"),
        "verification_sha256": common.sha(
            common.SOURCE_RUN / "adaptive/A1_adaptation_depth/verification/A1-FULL-S7/ENDPOINT.json"
        ),
        "parent_sha256": provenance["parent_sha256"],
        "updates": 12_000,
    }


def _sensor_binding(role: str) -> dict[str, Any]:
    from operational.range_transfer.common import digest

    rows = training.role_rows(role)
    folder = training.cache_root(role)
    complete = common.read_json(folder / "COMPLETE.json")
    binding = common.read_json(folder / "BINDING.json")
    manifest = training.manifest_path(role)
    if (
        len(rows) != POPULATIONS[role]
        or complete["status"] != "COMPLETE"
        or complete["rows"] != len(rows)
        or complete["binding"] != digest(binding)
        or binding["manifest_sha256"] != common.sha(manifest)
    ):
        raise ValueError(f"Incomplete native sensor population: {role}")
    return {
        "rows": len(rows),
        "groups": sorted(rows.group_id.astype(str).unique().tolist()),
        "manifest_sha256": common.sha(manifest),
        "sensor_complete_sha256": common.sha(folder / "COMPLETE.json"),
        "sensor_binding_sha256": common.sha(folder / "BINDING.json"),
        "ordered_tokens_sha256": _strings_sha(rows.sample_token.astype(str).to_numpy()),
    }


def register() -> dict[str, Any]:
    """Freeze the diagnostic before any descriptor extraction or H fit."""
    base = registration.verify()
    training_record = training.registered()
    sources = _sources()
    source_commit = training.committed_sources(sources)
    contracts = _contracts()
    record = {
        "schema": "v19_frozen_full_sign_probe_v1",
        "status": "REGISTERED_BEFORE_DESCRIPTOR_EXTRACTION",
        "utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "training_registration_sha256": common.sha(
            training.destination("training/REGISTRATION.json")
        ),
        "training_source_closure": training_record["sources"],
        "base_registration_sha256": common.sha(registration.REGISTRATION),
        "base_protocol_sha256": common.sha(registration.PROTOCOL),
        "authorization_scope": (
            "Diagnostic-only frozen linear probe; no encoder training, TTC candidate, "
            "router feature extension, specialized head, promotion or SOTA claim"
        ),
        "exposure": {
            "P": "Producer training and probe fitting; not representation generalization",
            "H": "Repeatedly exposed development; evaluated once by this fixed probe",
            "V": "Not opened",
        },
        "endpoint": _endpoint_binding(),
        "populations": {role: _sensor_binding(role) for role in POPULATIONS},
        **contracts,
        "sources": sources,
        "software": _software(),
        "base_lineage_status": base["status"],
    }
    if REGISTRATION.exists():
        if common.read_json(REGISTRATION) != record:
            # UTC and source commit are historical values after first freeze.
            return verify_registration()
        return record
    common.write_json(REGISTRATION, record)
    return record


def verify_registration() -> dict[str, Any]:
    """Revalidate immutable source, endpoint, population and base bindings."""
    record = common.read_json(REGISTRATION)
    registration.verify()
    training_record = training.registered()
    training.committed_sources(record["sources"])
    contracts = _contracts()
    if (
        record["schema"] != "v19_frozen_full_sign_probe_v1"
        or record["base_registration_sha256"] != common.sha(registration.REGISTRATION)
        or record["base_protocol_sha256"] != common.sha(registration.PROTOCOL)
        or record["training_registration_sha256"]
        != common.sha(training.destination("training/REGISTRATION.json"))
        or record["training_source_closure"] != training_record["sources"]
        or record["endpoint"] != _endpoint_binding()
        or any(record[name] != value for name, value in contracts.items())
        or record["software"] != _software()
    ):
        raise ValueError("Sign-probe registration contract changed")
    if record["sources"] != _sources():
        raise ValueError("Sign-probe registered source changed")
    for role in POPULATIONS:
        if record["populations"][role] != _sensor_binding(role):
            raise ValueError(f"Registered sign-probe population changed: {role}")
    return record


def _strings_sha(values: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _binding(role: str, record: dict[str, Any]) -> str:
    from operational.range_transfer.common import digest

    return digest(
        {
            "schema": record["schema"],
            "endpoint": record["endpoint"],
            "population": record["populations"][role],
            "features": record["features"],
            "extract": record["extract"],
            "sources": record["sources"],
        }
    )


def _load_model(record: dict[str, Any]) -> GeometricPairMeasurement:
    import torch

    payload = torch.load(ENDPOINT, map_location="cpu", weights_only=False)
    model = training.make_model()
    if (
        common.sha(ENDPOINT) != record["endpoint"]["checkpoint_sha256"]
        or payload["completed_updates"] != 12_000
        or payload["model_state_dict"].keys() != model.state_dict().keys()
    ):
        raise ValueError("A1-FULL payload contract changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def _setup_extraction_runtime() -> None:
    """Apply the native deterministic FP32 policy before touching the CUDA model."""
    from operational.joint_measurement.train import setup

    setup(SEED)


def _chunk_path(role: str, index: int) -> Path:
    return DIRECTORY / "features" / role / f"chunk_{index:05d}.npz"


def _parity(
    raw: np.ndarray,
    prediction: np.ndarray,
    expected_raw: np.ndarray,
    expected_prediction: np.ndarray,
) -> dict[str, Any]:
    """Prefer exact historical replay; admit only bounded, sign-preserving fallback drift."""
    arrays = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (raw, prediction, expected_raw, expected_prediction)
    )
    if len({value.shape for value in arrays}) != 1 or not all(
        np.isfinite(value).all() for value in arrays
    ):
        raise ValueError("Parity arrays must be aligned finite FP32 values")
    raw, prediction, expected_raw, expected_prediction = arrays
    raw_exact = np.array_equal(raw, expected_raw)
    prediction_exact = np.array_equal(prediction, expected_prediction)
    raw_maxabs = float(np.max(np.abs(raw - expected_raw), initial=0.0))
    prediction_maxabs = float(np.max(np.abs(prediction - expected_prediction), initial=0.0))
    raw_sign_flips = int(np.count_nonzero((raw > 0) != (expected_raw > 0)))
    prediction_sign_flips = int(np.count_nonzero((prediction > 0) != (expected_prediction > 0)))
    exact = raw_exact and prediction_exact
    fallback = (
        raw_maxabs <= PARITY["fallback_atol"]
        and prediction_maxabs <= PARITY["fallback_atol"]
        and raw_sign_flips == 0
        and prediction_sign_flips == 0
    )
    if not (exact or fallback):
        raise ValueError("Frozen descriptor pass changed historical FULL outputs")
    return {
        "status": "EXACT" if exact else "TOLERATED_SIGN_PRESERVING",
        "raw_phase_exact": raw_exact,
        "prediction_ttc_exact": prediction_exact,
        "raw_phase_max_abs": raw_maxabs,
        "prediction_ttc_max_abs": prediction_maxabs,
        "raw_phase_sign_flips": raw_sign_flips,
        "prediction_ttc_sign_flips": prediction_sign_flips,
        "atol": PARITY["fallback_atol"],
        "rtol": PARITY["fallback_rtol"],
    }


def _validate_parity(value: dict[str, Any]) -> None:
    exact = (
        value.get("status") == "EXACT"
        and value.get("raw_phase_exact") is True
        and value.get("prediction_ttc_exact") is True
    )
    fallback = (
        value.get("status") == "TOLERATED_SIGN_PRESERVING"
        and value.get("raw_phase_max_abs", np.inf) <= PARITY["fallback_atol"]
        and value.get("prediction_ttc_max_abs", np.inf) <= PARITY["fallback_atol"]
        and value.get("raw_phase_sign_flips") == 0
        and value.get("prediction_ttc_sign_flips") == 0
        and value.get("atol") == PARITY["fallback_atol"]
        and value.get("rtol") == PARITY["fallback_rtol"]
    )
    if not (exact or fallback):
        raise ValueError("Chunk parity evidence changed")


def _read_chunk(
    path: Path, *, binding: str, begin: int, stop: int, tokens: np.ndarray
) -> dict[str, np.ndarray]:
    receipt = common.read_json(path.with_suffix(".json"))
    if (
        set(receipt) != {"binding", "start", "stop", "sha256", "parity"}
        or receipt["binding"] != binding
        or receipt["start"] != begin
        or receipt["stop"] != stop
        or receipt["sha256"] != common.sha(path)
    ):
        raise ValueError("Sign-probe feature chunk receipt changed")
    _validate_parity(receipt["parity"])
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"sample_token", "features", "raw_phase", "prediction_ttc"}:
            raise ValueError("Unexpected sign-probe feature fields")
        values = {name: stored[name] for name in stored.files}
    count = stop - begin
    if (
        values["sample_token"].shape != (count,)
        or not np.array_equal(values["sample_token"].astype(str), tokens[begin:stop])
        or values["features"].shape != (count, FEATURE_COUNT)
        or values["features"].dtype != np.float32
        or values["raw_phase"].shape != (count,)
        or values["raw_phase"].dtype != np.float32
        or values["prediction_ttc"].shape != (count,)
        or values["prediction_ttc"].dtype != np.float32
        or not all(np.isfinite(values[name]).all() for name in values if name != "sample_token")
    ):
        raise ValueError("Invalid sign-probe feature chunk")
    return values


def _write_chunk(
    path: Path,
    *,
    binding: str,
    begin: int,
    stop: int,
    tokens: np.ndarray,
    features: np.ndarray,
    raw_phase: np.ndarray,
    prediction_ttc: np.ndarray,
    parity: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + f".{os.getpid()}.pending.npz")
    np.savez_compressed(
        pending,
        sample_token=tokens[begin:stop],
        features=features,
        raw_phase=raw_phase,
        prediction_ttc=prediction_ttc,
    )
    os.replace(pending, path)
    common.write_json(
        path.with_suffix(".json"),
        {
            "binding": binding,
            "start": begin,
            "stop": stop,
            "sha256": common.sha(path),
            "parity": parity,
        },
    )


def _finish_features(role: str, binding: str, rows: pd.DataFrame) -> dict[str, Any]:
    pieces = [
        _read_chunk(
            _chunk_path(role, begin // CHUNK),
            binding=binding,
            begin=begin,
            stop=min(begin + CHUNK, len(rows)),
            tokens=rows.sample_token.astype(str).to_numpy(),
        )
        for begin in range(0, len(rows), CHUNK)
    ]
    concatenated = {
        name: np.concatenate([piece[name] for piece in pieces])
        for name in ("sample_token", "features", "raw_phase", "prediction_ttc")
    }
    parity = _parity(
        concatenated["raw_phase"],
        concatenated["prediction_ttc"],
        rows.raw_phase.to_numpy(dtype=np.float32),
        rows.prediction_ttc.to_numpy(dtype=np.float32),
    )
    final = DIRECTORY / "features" / role / "features.npz"
    pending = final.with_name(f"features.{os.getpid()}.pending.npz")
    np.savez_compressed(pending, **concatenated)
    os.replace(pending, final)
    receipt = {
        "status": "COMPLETE",
        "role": role,
        "rows": len(rows),
        "feature_count": FEATURE_COUNT,
        "binding": binding,
        "sha256": common.sha(final),
        "contains_targets": False,
        "encoder_optimizer_updates": 0,
        "parity": parity,
        "chunks": {
            _chunk_path(role, begin // CHUNK).with_suffix(".json").name: common.sha(
                _chunk_path(role, begin // CHUNK).with_suffix(".json")
            )
            for begin in range(0, len(rows), CHUNK)
        },
    }
    common.write_json(final.parent / "COMPLETE.json", receipt)
    return receipt


def _admit_complete_features(role: str, binding: str, rows: pd.DataFrame) -> dict[str, Any]:
    """Replay every chunk and final cache receipt before accepting completed extraction."""
    folder = DIRECTORY / "features" / role
    receipt = common.read_json(folder / "COMPLETE.json")
    final = folder / "features.npz"
    expected_chunks = {
        _chunk_path(role, begin // CHUNK).with_suffix(".json").name: common.sha(
            _chunk_path(role, begin // CHUNK).with_suffix(".json")
        )
        for begin in range(0, len(rows), CHUNK)
    }
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("role") != role
        or receipt.get("rows") != len(rows)
        or receipt.get("feature_count") != FEATURE_COUNT
        or receipt.get("binding") != binding
        or receipt.get("sha256") != common.sha(final)
        or receipt.get("contains_targets") is not False
        or receipt.get("encoder_optimizer_updates") != 0
        or "parity" not in receipt
        or receipt.get("chunks") != expected_chunks
    ):
        raise ValueError("Completed sign-probe feature receipt changed")
    tokens = rows.sample_token.astype(str).to_numpy()
    for begin in range(0, len(rows), CHUNK):
        _read_chunk(
            _chunk_path(role, begin // CHUNK),
            binding=binding,
            begin=begin,
            stop=min(begin + CHUNK, len(rows)),
            tokens=tokens,
        )
    with np.load(final, allow_pickle=False) as stored:
        if set(stored.files) != {"sample_token", "features", "raw_phase", "prediction_ttc"}:
            raise ValueError("Completed feature cache fields changed")
        values = {name: stored[name] for name in stored.files}
    parity = _parity(
        values["raw_phase"],
        values["prediction_ttc"],
        rows.raw_phase.to_numpy(dtype=np.float32),
        rows.prediction_ttc.to_numpy(dtype=np.float32),
    )
    _validate_parity(receipt["parity"])
    if (
        values["features"].shape != (len(rows), FEATURE_COUNT)
        or values["features"].dtype != np.float32
        or not np.array_equal(values["sample_token"].astype(str), tokens)
        or values["raw_phase"].shape != (len(rows),)
        or values["raw_phase"].dtype != np.float32
        or values["prediction_ttc"].shape != (len(rows),)
        or values["prediction_ttc"].dtype != np.float32
        or not all(
            np.isfinite(values[name]).all() for name in ("features", "raw_phase", "prediction_ttc")
        )
        or receipt["parity"] != parity
    ):
        raise ValueError("Completed feature cache population changed")
    return receipt


def _extract_registered(role: str) -> dict[str, Any]:
    """Extract complete frozen descriptors inside the admitted runtime context."""
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.decision_recovery.sensor_runtime import make_evaluation_sensors
    from operational.joint_measurement.fusion_pair_extract import extract_pair

    if role not in POPULATIONS:
        raise ValueError("Sign probe admits only complete P/H populations")
    record = verify_registration()
    binding = _binding(role, record)
    rows = common.load_population(role, ["FULL"])["FULL"]
    completed = DIRECTORY / "features" / role / "COMPLETE.json"
    if completed.exists():
        return _admit_complete_features(role, binding, rows)
    dataset = make_evaluation_sensors(role)
    model = None
    try:
        if (
            not dataset.rows[common.IDENTITY]
            .reset_index(drop=True)
            .equals(rows[common.IDENTITY].reset_index(drop=True))
        ):
            raise ValueError("Sign-probe sensor metadata differs from admitted predictions")
        with training.gpu_charge(f"V19_SIGN_PROBE_FEATURES_{role}", reserve=900) as deadline:
            _setup_extraction_runtime()
            model = _load_model(record).cuda()
            tokens = rows.sample_token.astype(str).to_numpy()
            with torch.inference_mode():
                for begin in range(0, len(rows), CHUNK):
                    stop = min(begin + CHUNK, len(rows))
                    path = _chunk_path(role, begin // CHUNK)
                    if path.with_suffix(".json").exists():
                        _read_chunk(
                            path,
                            binding=binding,
                            begin=begin,
                            stop=stop,
                            tokens=tokens,
                        )
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "role": role, "next_row": begin}
                    training.require_resources()
                    features, raw, prediction = [], [], []
                    for start in range(begin, stop, BATCH):
                        if time.perf_counter() >= deadline:
                            return {"status": "PARTIAL", "role": role, "next_row": begin}
                        batch = dataset.batch(list(range(start, min(start + BATCH, stop))))
                        values, estimates = extract_pair(
                            model, batch.events.cuda(), batch.delta_t_s.cuda().reshape(-1)
                        )
                        features.append(values[:, :FEATURE_COUNT])
                        raw.append(values[:, FEATURE_COUNT])
                        prediction.append(estimates)
                    feature_array = np.concatenate(features).astype(np.float32, copy=False)
                    raw_array = np.concatenate(raw).astype(np.float32, copy=False)
                    prediction_array = np.concatenate(prediction).astype(np.float32, copy=False)
                    expected_raw = rows.raw_phase.iloc[begin:stop].to_numpy(dtype=np.float32)
                    expected_ttc = rows.prediction_ttc.iloc[begin:stop].to_numpy(dtype=np.float32)
                    parity = _parity(raw_array, prediction_array, expected_raw, expected_ttc)
                    replay_ttc = phase_to_ttc(torch.from_numpy(raw_array)).numpy()
                    if not np.array_equal(replay_ttc, prediction_array):
                        raise ValueError("Extracted raw phase and emitted TTC are inconsistent")
                    _write_chunk(
                        path,
                        binding=binding,
                        begin=begin,
                        stop=stop,
                        tokens=tokens,
                        features=feature_array,
                        raw_phase=raw_array,
                        prediction_ttc=prediction_array,
                        parity=parity,
                    )
        return _finish_features(role, binding, rows)
    finally:
        dataset.shutdown()
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def extract(role: str) -> dict[str, Any]:
    """Extract complete frozen descriptors; safe to resume by immutable chunk."""
    if role not in POPULATIONS:
        raise ValueError("Sign probe admits only complete P/H populations")
    with (
        training.file_lock(DIRECTORY / "DRIVER.lock"),
        training.runtime_context(),
    ):
        return _extract_registered(role)


def _load_features(role: str, record: dict[str, Any]) -> dict[str, np.ndarray]:
    folder = DIRECTORY / "features" / role
    rows = common.load_population(role, ["FULL"])["FULL"]
    receipt = _admit_complete_features(role, _binding(role, record), rows)
    path = folder / "features.npz"
    if (
        receipt["rows"] != POPULATIONS[role]
        or receipt["feature_count"] != FEATURE_COUNT
        or receipt["encoder_optimizer_updates"] != 0
    ):
        raise ValueError(f"Incomplete sign-probe features: {role}")
    with np.load(path, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    if (
        set(values) != {"sample_token", "features", "raw_phase", "prediction_ttc"}
        or values["features"].shape != (len(rows), FEATURE_COUNT)
        or values["features"].dtype != np.float32
        or not np.array_equal(values["sample_token"].astype(str), rows.sample_token.astype(str))
    ):
        raise ValueError(f"Sign-probe feature population changed: {role}")
    return values


def _fit_logistic(x: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    if (
        x.ndim != 2
        or x.shape[1] != FEATURE_COUNT
        or label.shape != (len(x),)
        or not np.isfinite(x).all()
        or set(np.unique(label)) != {0, 1}
    ):
        raise ValueError("Complete finite binary P probe inputs required")
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    classifier = LogisticRegression(
        C=LOGISTIC["C"],
        class_weight=LOGISTIC["class_weight"],
        solver=LOGISTIC["solver"],
        max_iter=LOGISTIC["max_iter"],
        tol=LOGISTIC["tol"],
        fit_intercept=LOGISTIC["fit_intercept"],
        random_state=SEED,
    ).fit(scaled, label)
    if classifier.n_iter_[0] >= LOGISTIC["max_iter"]:
        raise RuntimeError("Fixed sign probe did not converge")
    return {
        "schema": "v19_frozen_full_sign_logistic_v1",
        "features": list(FEATURE_NAMES),
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefficient": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "logistic_fit_iterations": int(classifier.n_iter_[0]),
        "contract": LOGISTIC,
        "fit_rows": len(x),
        "fit_positive": int(label.sum()),
        "fit_negative": int((label == 0).sum()),
        "fit_role": "P",
        "scope": "DIAGNOSTIC_ONLY_NOT_TTC_CANDIDATE",
        "system_promotion": False,
    }


def _predict_probability(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    if (
        model["schema"] != "v19_frozen_full_sign_logistic_v1"
        or model["features"] != list(FEATURE_NAMES)
        or model["contract"] != LOGISTIC
    ):
        raise ValueError("Sign-probe model contract changed")
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != FEATURE_COUNT
        or any(value.shape != (FEATURE_COUNT,) for value in (mean, scale, coefficient))
        or not all(np.isfinite(value).all() for value in (x, mean, scale, coefficient))
        or not (scale > 0).all()
    ):
        raise ValueError("Invalid serialized sign probe")
    probability = expit(((x - mean) / scale) @ coefficient + model["intercept"])
    if probability.shape != (len(x),) or not np.isfinite(probability).all():
        raise ValueError("Sign probe emitted invalid probabilities")
    return probability


def _errors(label: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    label, predicted = np.asarray(label, bool), np.asarray(predicted, bool)
    if label.shape != predicted.shape or not len(label):
        raise ValueError("Aligned nonempty sign arrays are required")
    negative, positive = ~label, label
    negative_error = (
        float((predicted[negative] != label[negative]).mean()) if negative.any() else None
    )
    positive_error = (
        float((predicted[positive] != label[positive]).mean()) if positive.any() else None
    )
    conditional = [value for value in (negative_error, positive_error) if value is not None]
    return {
        "rows": len(label),
        "positive_rows": int(positive.sum()),
        "negative_rows": int(negative.sum()),
        "error": float((predicted != label).mean()),
        "negative_error": negative_error,
        "positive_error": positive_error,
        "balanced_error": float(np.mean(conditional)),
        "true_positive": int((predicted & label).sum()),
        "true_negative": int((~predicted & ~label).sum()),
        "false_positive": int((predicted & ~label).sum()),
        "false_negative": int((~predicted & label).sum()),
    }


def _metrics(label: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    label = np.asarray(label, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if (
        label.shape != probability.shape
        or not len(label)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Invalid sign probabilities")
    hard = probability >= LOGISTIC["threshold"]
    result = _errors(label.astype(bool), hard)
    result.update(
        brier=float(brier_score_loss(label, probability)),
        roc_auc=float(roc_auc_score(label, probability)) if len(np.unique(label)) == 2 else None,
        average_precision=float(average_precision_score(label, probability))
        if len(np.unique(label)) == 2
        else None,
    )
    edges = np.linspace(0.0, 1.0, 11)
    bins, ece = [], 0.0
    for index in range(10):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9 else probability < edges[index + 1]
        )
        count = int(selected.sum())
        confidence = float(probability[selected].mean()) if count else None
        frequency = float(label[selected].mean()) if count else None
        if count:
            ece += count / len(label) * abs(float(confidence) - float(frequency))
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": count,
                "mean_probability": confidence,
                "positive_frequency": frequency,
            }
        )
    result.update(ece10=float(ece), calibration_bins=bins)
    return result


def _group_metrics(
    label: np.ndarray, probability: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    groups = np.asarray(groups).astype(str)
    return {
        str(group): _metrics(label[groups == group], probability[groups == group])
        for group in np.unique(groups)
    }


def _group_errors(label: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    groups = np.asarray(groups).astype(str)
    return {
        str(group): _errors(label[groups == group], predicted[groups == group])
        for group in np.unique(groups)
    }


def _contingency(label: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Count candidate rescues and damages against FULL within each true-sign stratum."""
    label = np.asarray(label, bool)
    baseline, candidate = np.asarray(baseline, bool), np.asarray(candidate, bool)
    if label.shape != baseline.shape or label.shape != candidate.shape:
        raise ValueError("Contingency arrays must align")
    result = {}
    for name, selected in (("negative", ~label), ("positive", label)):
        baseline_correct = baseline[selected] == label[selected]
        candidate_correct = candidate[selected] == label[selected]
        result[name] = {
            "rows": int(selected.sum()),
            "both_correct": int((baseline_correct & candidate_correct).sum()),
            "rescued_by_S1": int((~baseline_correct & candidate_correct).sum()),
            "damaged_by_S1": int((baseline_correct & ~candidate_correct).sum()),
            "both_wrong": int((~baseline_correct & ~candidate_correct).sum()),
        }
    return result


def _bootstrap_balanced_error(
    label: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    label = np.asarray(label, bool)
    baseline, candidate = np.asarray(baseline, bool), np.asarray(candidate, bool)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    # Retain class-specific counts and mistakes for each complete acquisition
    # group.  A bootstrap draw then reconstructs the pooled conditional errors;
    # rows from a group are never sampled independently.
    rows = np.empty((len(unique), 6), dtype=np.float64)
    for index, group in enumerate(unique):
        selected = groups == group
        negative, positive = selected & ~label, selected & label
        rows[index] = (
            negative.sum(),
            positive.sum(),
            (baseline[negative] != label[negative]).sum(),
            (baseline[positive] != label[positive]).sum(),
            (candidate[negative] != label[negative]).sum(),
            (candidate[positive] != label[positive]).sum(),
        )
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(unique), size=(BOOTSTRAP_REPLICATES, len(unique))
    )
    sampled = rows[draws].sum(axis=1)
    defined = (sampled[:, 0] > 0) & (sampled[:, 1] > 0)
    baseline_negative = np.divide(
        sampled[:, 2], sampled[:, 0], out=np.zeros(len(sampled)), where=sampled[:, 0] > 0
    )
    baseline_positive = np.divide(
        sampled[:, 3], sampled[:, 1], out=np.zeros(len(sampled)), where=sampled[:, 1] > 0
    )
    candidate_negative = np.divide(
        sampled[:, 4], sampled[:, 0], out=np.zeros(len(sampled)), where=sampled[:, 0] > 0
    )
    candidate_positive = np.divide(
        sampled[:, 5], sampled[:, 1], out=np.zeros(len(sampled)), where=sampled[:, 1] > 0
    )
    baseline_error = 0.5 * (baseline_negative + baseline_positive)
    candidate_error = 0.5 * (candidate_negative + candidate_positive)
    delta = candidate_error - baseline_error
    point_baseline = _errors(label, baseline)["balanced_error"]
    point_candidate = _errors(label, candidate)["balanced_error"]
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "acquisition_group",
        "statistic": "pooled_conditional_balanced_error_candidate_minus_FULL_emitted_sign",
        "point": float(point_candidate - point_baseline),
        "defined_replicates": int(defined.sum()),
        "undefined_replicates": int((~defined).sum()),
        "ci95_if_all_defined": np.quantile(delta, [0.025, 0.975]).tolist()
        if defined.all()
        else None,
        "refitted_per_bootstrap": False,
        "scope": "fixed_H_predictions_descriptive_not_independent_confirmation",
    }


def _validate_result_files(record: dict[str, Any]) -> dict[str, Any]:
    result_path = DIRECTORY / "RESULT.json"
    result = common.read_json(result_path)
    prediction_path = DIRECTORY / "H_SIGN_PREDICTIONS.parquet"
    if (
        result.get("schema") != "v19_frozen_full_sign_probe_result_v1"
        or result.get("status") != "COMPLETE_DIAGNOSTIC_NOT_CANDIDATE"
        or result.get("registration_sha256") != common.sha(REGISTRATION)
        or result.get("encoder_optimizer_updates") != 0
        or result.get("system_promotion") is not False
        or result.get("official_confirmation") is not False
        or result.get("SOTA_claim") is not False
        or result.get("models", {}).get("S1") != common.sha(DIRECTORY / "models/S1.json")
        or result.get("models", {}).get("SHUFFLED")
        != common.sha(DIRECTORY / "models/SHUFFLED.json")
        or result.get("prediction_sha256") != common.sha(prediction_path)
        or result.get("interpretation") != record["interpretation"]
        or record["flags"]["encoder_optimizer_updates"] != 0
    ):
        raise ValueError("Completed sign-probe result files changed")
    return result


def _result_receipt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "registration_sha256": common.sha(REGISTRATION),
        "result_sha256": common.sha(DIRECTORY / "RESULT.json"),
        "encoder_optimizer_updates": 0,
    }


def _admit_result(record: dict[str, Any]) -> dict[str, Any]:
    result = _validate_result_files(record)
    if common.read_json(DIRECTORY / "COMPLETE.json") != _result_receipt(result):
        raise ValueError("Completed sign-probe result receipt changed")
    return result


def _fit_and_evaluate_registered() -> dict[str, Any]:
    """Fit the one registered P probe and evaluate fixed H exactly once."""
    record = verify_registration()
    result_path, complete_path = DIRECTORY / "RESULT.json", DIRECTORY / "COMPLETE.json"
    if result_path.exists():
        result = _validate_result_files(record)
        if not complete_path.exists():
            common.write_json(complete_path, _result_receipt(result))
        return _admit_result(record)
    if complete_path.exists():
        raise ValueError("Sign-probe receipt exists without its result")
    p, h = _load_features("P", record), _load_features("H", record)
    p_table = common.load_population("P", ["FULL"])["FULL"]
    h_table = common.load_population("H", ["FULL"])["FULL"]
    p_label = (p_table.target_ttc.to_numpy() > 0).astype(np.int64)
    h_label = (h_table.target_ttc.to_numpy() > 0).astype(np.int64)
    model = _fit_logistic(p["features"], p_label)
    shuffled_label = p_label[np.random.default_rng(SEED).permutation(len(p_label))]
    shuffled = _fit_logistic(p["features"], shuffled_label)
    probability = _predict_probability(model, h["features"])
    shuffled_probability = _predict_probability(shuffled, h["features"])
    baseline_hard = h["prediction_ttc"] > 0
    candidate_hard = probability >= LOGISTIC["threshold"]
    baseline = _errors(h_label.astype(bool), baseline_hard)
    candidate = _metrics(h_label, probability)
    negative_control = _metrics(h_label, shuffled_probability)
    bootstrap = _bootstrap_balanced_error(
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
    model.update(
        training_population={
            "role": "P",
            "tokens_sha256": record["populations"]["P"]["ordered_tokens_sha256"],
        }
    )
    shuffled.update(
        training_population={
            "role": "P",
            "tokens_sha256": record["populations"]["P"]["ordered_tokens_sha256"],
        },
        label_permutation_seed=SEED,
    )
    common.write_json(DIRECTORY / "models/S1.json", model)
    common.write_json(DIRECTORY / "models/SHUFFLED.json", shuffled)
    frame = h_table[common.IDENTITY].copy()
    frame["full_emitted_sign_positive"] = baseline_hard
    frame["sign_probability_positive"] = probability
    frame["sign_prediction_positive"] = candidate_hard
    frame["shuffled_probability_positive"] = shuffled_probability
    frame["shuffled_prediction_positive"] = shuffled_probability >= LOGISTIC["threshold"]
    prediction_path = DIRECTORY / "H_SIGN_PREDICTIONS.parquet"
    pending = prediction_path.with_suffix(".pending")
    frame.to_parquet(pending, index=False)
    os.replace(pending, prediction_path)
    result = {
        "schema": "v19_frozen_full_sign_probe_result_v1",
        "status": "COMPLETE_DIAGNOSTIC_NOT_CANDIDATE",
        "registration_sha256": common.sha(REGISTRATION),
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
        },
        "evaluation": {
            "role": "H_REUSED_DEVELOPMENT",
            "rows": len(h_label),
            "FULL_emitted_sign": baseline,
            "S1": candidate,
            "SHUFFLED": negative_control,
            "P_prevalence_constant_brier": constant_brier,
            "FULL_by_group": _group_errors(
                h_label.astype(bool), baseline_hard, h_table.group_id.to_numpy()
            ),
            "S1_by_group": _group_metrics(h_label, probability, h_table.group_id.to_numpy()),
            "SHUFFLED_by_group": _group_metrics(
                h_label, shuffled_probability, h_table.group_id.to_numpy()
            ),
            "S1_vs_FULL_contingency": _contingency(
                h_label.astype(bool), baseline_hard, candidate_hard
            ),
            "bootstrap": bootstrap,
        },
        "interpretation_gate": {"eligible": all(checks.values()), "checks": checks},
        "interpretation": record["interpretation"],
        "logistic_fit_iterations": {
            "S1": model["logistic_fit_iterations"],
            "SHUFFLED": shuffled["logistic_fit_iterations"],
        },
        **record["flags"],
    }
    common.write_json(DIRECTORY / "RESULT.json", result)
    common.write_json(DIRECTORY / "COMPLETE.json", _result_receipt(result))
    return result


def fit_and_evaluate() -> dict[str, Any]:
    """Serialize the CPU fit with extraction and make completed replay idempotent."""
    with (
        training.file_lock(DIRECTORY / "DRIVER.lock"),
        training.runtime_context(),
    ):
        return _fit_and_evaluate_registered()


def run_all() -> dict[str, Any]:
    """Register, resume both GPU extractions sequentially, then fit the fixed CPU probe."""
    register()
    for role in ("P", "H"):
        previous = None
        while True:
            result = extract(role)
            if result.get("status") == "COMPLETE":
                break
            cursor = result.get("next_row")
            if result.get("status") != "PARTIAL" or cursor == previous:
                raise RuntimeError(f"Sign-probe extraction made no verified progress: {role}")
            previous = cursor
    return fit_and_evaluate()


__all__ = [
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "LOGISTIC",
    "extract",
    "fit_and_evaluate",
    "register",
    "run_all",
    "verify_registration",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "extract", "fit", "runall"))
    parser.add_argument("--role", choices=tuple(POPULATIONS))
    arguments = parser.parse_args()
    if arguments.action == "register":
        value = register()
    elif arguments.action == "extract":
        if arguments.role is None:
            parser.error("extract requires --role P or H")
        value = extract(arguments.role)
    elif arguments.action == "fit":
        if arguments.role is not None:
            parser.error("fit does not accept --role")
        value = fit_and_evaluate()
    else:
        if arguments.role is not None:
            parser.error("runall does not accept --role")
        value = run_all()
    print(json.dumps(value, indent=2, sort_keys=True))

"""Fail closed on changed CPU identities and source evidence before GPU allocation."""

from dataclasses import asdict

import numpy as np
import pytest

from operational.phase_event_expansion.model import SolverConfig
from scripts import verify_phase_event_expansion_all_cuda as verifier
from scripts.verify_phase_event_expansion_all_cuda import (
    strict_compare_results,
    validate_cpu_metadata,
    validate_cpu_query,
    verify_guarded_evaluator,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("q", None),
        ("q", float("nan")),
        ("translation", None),
        ("translation", [0.0, float("inf")]),
        ("ttc_seconds", None),
        ("ttc_seconds", float("nan")),
        ("ttc_seconds", float("inf")),
        ("ttc_seconds", 0.6),
    ],
)
def test_strict_comparison_rejects_invalid_emitted_fields(field, value):
    from dataclasses import replace

    from operational.phase_event_expansion.model import ExpansionResult

    result = ExpansionResult(2.0, (0.1, -0.2), 0.5, True, "available", 0.1, {})
    assert strict_compare_results(result, result)["passed"]
    bad = replace(result, **{field: value})
    assert not strict_compare_results(result, bad)["passed"]


def test_strict_comparison_checks_invalid_ttc_and_derived_reciprocal_difference():
    from dataclasses import replace

    from operational.phase_event_expansion.model import ExpansionResult

    result = ExpansionResult(0.1, (0.0, 0.0), 10.0, True, "available", 0.1, {})
    changed = replace(result, q=0.100001, ttc_seconds=1 / 0.100001)
    comparison = strict_compare_results(result, changed)
    assert comparison["numeric_pass"]
    assert comparison["ttc_numeric_pass"]
    assert comparison["passed"]
    assert comparison["ttc_absolute_error"] <= comparison["ttc_derived_absolute_bound_seconds"]
    invalid = replace(result, valid=False, reason="flat_objective", ttc_seconds=None)
    assert strict_compare_results(invalid, invalid)["passed"]
    assert not strict_compare_results(invalid, replace(invalid, ttc_seconds=10.0))["passed"]


def metadata():
    recipe = asdict(SolverConfig(objective="temporal_ncc", reference="endpoint"))
    config = {
        "schema": "phase_event_expansion_real_development_v1",
        "solver": recipe,
        "cache_manifest_sha256": "cache",
        "synthetic_config_sha256": "synthetic_config",
        "synthetic_summary_sha256": "synthetic_summary",
    }
    seal = {"rows": 77, "targets_read": False, "config_sha256": "config", "sha256": "npz"}
    expected = {
        "solver": recipe,
        "cache_manifest_sha256": "cache",
        "synthetic_config_sha256": "synthetic_config",
        "synthetic_summary_sha256": "synthetic_summary",
        "config_sha256": "config",
        "predictions_sha256": "npz",
    }
    return config, seal, expected


@pytest.mark.parametrize(
    ("part", "key", "wrong"),
    [
        (0, "solver", {}),
        (0, "cache_manifest_sha256", "different"),
        (0, "synthetic_config_sha256", "different"),
        (0, "synthetic_summary_sha256", "different"),
        (1, "config_sha256", "different"),
        (1, "sha256", "different"),
        (1, "rows", 76),
        (1, "targets_read", True),
    ],
)
def test_cpu_metadata_rejects_changed_identity_or_seal(part, key, wrong):
    config, seal, expected = metadata()
    validate_cpu_metadata(config, seal, expected)
    (config, seal)[part][key] = wrong
    with pytest.raises(ValueError):
        validate_cpu_metadata(config, seal, expected)


def query():
    row = {"query_id": "Slider-500:0", "frame_index": 1, "query_us": 40081}
    record = {
        "token": "Slider-500:0",
        "index": 1,
        "query_us": 40081,
        "q": 2.0,
        "translation": [0.1, -0.2],
        "ttc_seconds": 0.5,
        "valid": True,
        "reason": "available",
        "elapsed_seconds": 1.0,
        "diagnostics": {},
    }
    return row, record


@pytest.mark.parametrize(
    ("key", "wrong"),
    [
        ("token", "Slider-500:1"),
        ("index", 2),
        ("query_us", 40082),
        ("ttc_seconds", 0.6),
        ("q", 3.0),
        ("translation", [float("nan"), 0]),
        ("valid", False),
        ("reason", "optimizer_failure"),
    ],
)
def test_cpu_query_rejects_wrong_clock_token_or_sealed_prediction(key, wrong):
    row, record = query()
    validate_cpu_query(record, row, row["query_id"], 0.5)
    record[key] = wrong
    with pytest.raises(ValueError):
        validate_cpu_query(record, row, row["query_id"], 0.5)


def test_invalid_cpu_query_requires_nan_in_sealed_prediction():
    row, record = query()
    record.update(valid=False, reason="optimizer_failure", ttc_seconds=None)
    validate_cpu_query(record, row, row["query_id"], np.nan)
    with pytest.raises(ValueError):
        validate_cpu_query(record, row, row["query_id"], 0.5)


def test_evaluator_allows_only_reviewed_guard_change(tmp_path, monkeypatch):
    # Tiny sources exercise the whitelist without depending on ignored run artifacts.
    archived, current = tmp_path / "legacy.py", tmp_path / "guarded.py"
    archived.write_text("def undistort_inputs(arrays, calibration):\n    return arrays\n")
    current.write_text(archived.read_text() + "\nGUARD_VERSION = 1\n")
    monkeypatch.setattr(verifier, "LEGACY_EVALUATOR_SHA", verifier.sha(archived))
    monkeypatch.setattr(verifier, "GUARDED_EVALUATOR_SHA", verifier.sha(current))
    verify_guarded_evaluator(archived, current)
    changed = tmp_path / "changed.py"
    changed.write_bytes(current.read_bytes() + b"\nUNREVIEWED = True\n")
    with pytest.raises(ValueError):
        verify_guarded_evaluator(archived, changed)


def test_reviewed_hash_does_not_override_calibration_ast_check(tmp_path, monkeypatch):
    archived, current = tmp_path / "legacy.py", tmp_path / "changed.py"
    archived.write_text("def undistort_inputs(arrays, calibration):\n    return arrays\n")
    current.write_text("def undistort_inputs(arrays, calibration):\n    return calibration\n")
    monkeypatch.setattr(verifier, "LEGACY_EVALUATOR_SHA", verifier.sha(archived))
    monkeypatch.setattr(verifier, "GUARDED_EVALUATOR_SHA", verifier.sha(current))
    with pytest.raises(ValueError, match="Calibration function"):
        verify_guarded_evaluator(archived, current)

"""Tests for V6 fold aggregation contracts."""

from __future__ import annotations

import pytest

from scripts.aggregate_v6_fold_results import (
    _validate_frozen_inputs,
    _validate_v6_summary,
)


def _inputs() -> tuple[dict, dict, dict]:
    grouped = {"status": "frozen_before_a8_results"}
    manifest = {
        "status": "frozen_before_v6_training",
        "contracts": {
            "a5_causal_is_diagnostic_geometry_unconstrained": True,
            "folds_unchanged_from_v5": True,
            "private_test_opened": False,
            "public_validation_opened": False,
            "v6_1_single_change_transport_radius_1_to_2": True,
        },
    }
    v5 = {
        "status": "completed_development_gate_evaluation",
        "contracts": {"private_test_opened": False},
    }
    return grouped, manifest, v5


def test_frozen_inputs_reject_open_private_test() -> None:
    grouped, manifest, v5 = _inputs()
    manifest["contracts"]["private_test_opened"] = True

    with pytest.raises(ValueError, match="contracts"):
        _validate_frozen_inputs(grouped, manifest, v5)


def test_v6_summary_requires_radius_two_and_frozen_geometry() -> None:
    manifest = {"configs": {"v6_1_r2_fold0": {"sha256": "frozen"}}}
    summary = {
        "config": {"sha256": "frozen"},
        "decision_contract": {
            "public_validation_used_for_selection": False,
            "private_test_remains_closed": True,
            "representation_change": {"transport_radius": 2},
            "dual_stream_contract": {"geometry_must_equal_parent_by_construction": True},
        },
    }

    _validate_v6_summary(summary, arm="v6_1", fold=0, manifest=manifest)
    summary["decision_contract"]["representation_change"]["transport_radius"] = 1
    with pytest.raises(ValueError, match="radius 2"):
        _validate_v6_summary(summary, arm="v6_1", fold=0, manifest=manifest)


def test_a5_summary_remains_diagnostic_only() -> None:
    manifest = {"configs": {"a5_causal_fold1": {"sha256": "frozen"}}}
    summary = {
        "config": {"sha256": "frozen"},
        "decision_contract": {
            "public_validation_used_for_selection": False,
            "private_test_remains_closed": True,
            "diagnostic_only_until_geometry_is_reassessed": True,
            "geometry_preservation_required": False,
        },
    }

    _validate_v6_summary(summary, arm="a5_causal", fold=1, manifest=manifest)

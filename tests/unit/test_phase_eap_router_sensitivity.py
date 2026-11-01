"""Router diagnostics preserve parent seals, missing predictions and exact native controls."""

import json

import numpy as np
import pytest

from operational.evttc_transfer_audit.common import file_sha
from scripts.audit_phase_eap_router_sensitivity import assert_f0, changes, verified_parent


def test_router_changes_keep_coverage_and_signs():
    row = changes([None, 1.0, -2.0, 4.0], [3.0, None, 2.0, 4.0])
    assert row["queries"] == 4
    assert row["paired"] == 2
    assert row["newly_missing"] == row["newly_available"] == row["sign_changes"] == 1
    assert row["exact"] == 1
    assert row["mean_abs_seconds"] == 2
    assert row["max_abs_seconds"] == 4


def test_no_overlap_is_not_zero_drift():
    result = changes([None], [None])
    assert result["paired"] == 0
    assert result["mean_abs_seconds"] is None


def test_native_prediction_mismatch_cannot_be_hidden_by_tolerance():
    vector = np.array([1.0, 2.0, 3.0], np.float32)
    assert_f0(vector, 1.0)
    with pytest.raises(ValueError):
        assert_f0(vector, 1.0000001)
    with pytest.raises(ValueError):
        assert_f0(vector, None)
    vector[0] = np.nan
    assert_f0(vector, None)
    with pytest.raises(ValueError):
        assert_f0(vector, 1.0)


def test_parent_must_seal_consumed_artifacts(tmp_path):
    (tmp_path / "FREEZE.json").write_text("{}")
    complete = {
        "status": "COMPLETE",
        "artifacts": {"FREEZE.json": file_sha(tmp_path / "FREEZE.json")},
    }
    (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    verified_parent(tmp_path, {"FREEZE.json"})
    with pytest.raises(ValueError, match="required artifacts"):
        verified_parent(tmp_path, {"PREDICTIONS.json"})
    (tmp_path / "FREEZE.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Changed"):
        verified_parent(tmp_path, {"FREEZE.json"})


def test_failed_parent_is_never_admitted(tmp_path):
    (tmp_path / "COMPLETE.json").write_text('{"status":"COMPLETE","artifacts":{}}')
    (tmp_path / "FAILED.json").write_text("{}")
    with pytest.raises(ValueError):
        verified_parent(tmp_path, set())


def test_parent_artifact_cannot_escape_its_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("unchanged")
    complete = {"status": "COMPLETE", "artifacts": {"../outside.txt": file_sha(outside)}}
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="escaping artifact"):
        verified_parent(root, set())


@pytest.mark.parametrize("old,new", [([], []), ([1], [1, 2]), ([[1]], [[1]])])
def test_invalid_prediction_alignment_is_rejected(old, new):
    with pytest.raises(ValueError):
        changes(old, new)

"""Scheduling never drops controls, changes scientific bindings or exposes H early."""

import copy

import pytest

from operational.joint_measurement import run as runner
from operational.joint_measurement.schedule import validate_order


def registration():
    return {
        "fits": ["DIRECT", "CORR", "FROZEN", "NATIVE"],
        "config_hash": "original-config",
        "scientific_source_hash": "original-source",
        "reason": "Cost-only scheduling decision, no outcome selection",
    }


def test_permutation_keeps_every_scientific_arm():
    record = registration()
    assert (
        validate_order(
            record, ["NATIVE", "DIRECT", "CORR", "FROZEN"], "original-config", "original-source"
        )
        == record["fits"]
    )


@pytest.mark.parametrize("change", ["omit", "duplicate", "foreign", "config", "source"])
def test_invalid_matrix_order_is_rejected(change):
    record = copy.deepcopy(registration())
    if change == "omit":
        record["fits"].pop()
    elif change == "duplicate":
        record["fits"][-1] = "DIRECT"
    elif change == "foreign":
        record["fits"][-1] = "NEW_UNREGISTERED_MODEL"
    elif change == "config":
        record["config_hash"] = "changed"
    else:
        record["scientific_source_hash"] = "changed"
    with pytest.raises(ValueError, match="complete admitted matrix"):
        validate_order(
            record, ["NATIVE", "DIRECT", "CORR", "FROZEN"], "original-config", "original-source"
        )


def test_reordered_matrix_cannot_evaluate_h_while_native_is_incomplete(tmp_path, monkeypatch):
    (tmp_path / "PREFLIGHT32.json").write_text("{}")
    (tmp_path / "COMPLETE.json").write_text("{}")
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner, "run_qa", lambda device: None)
    monkeypatch.setattr(runner, "cache_root", lambda role: tmp_path)
    monkeypatch.setattr(runner, "execution_order", lambda: registration()["fits"])
    calls = []

    def fit(name):
        calls.append(name)
        return {"status": "PAUSED" if name == "NATIVE" else "COMPLETE", "completed_updates": 3010}

    monkeypatch.setattr(runner, "fit", fit)
    monkeypatch.setattr(runner, "freeze_primary", lambda: pytest.fail("premature endpoint freeze"))
    monkeypatch.setattr(runner, "predict", lambda *args: pytest.fail("premature H evaluation"))
    result = runner.advance()
    assert calls == registration()["fits"]
    assert result["status"] == "TRAINING" and result["fit_id"] == "NATIVE"

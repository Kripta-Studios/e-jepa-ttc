"""C1 stage ordering must preserve endpoint barriers and the finite scope."""

import pytest

from operational.joint_measurement import geometry_run as runner


def prepare(monkeypatch, status):
    calls = []
    monkeypatch.setattr(runner, "require_authorization", lambda: None)
    monkeypatch.setattr(runner, "registered", lambda: {})
    monkeypatch.setattr(runner, "qa", lambda device: calls.append(("qa", device)))
    monkeypatch.setattr(runner, "fit", lambda sensors: {"status": status, "completed_updates": 100})
    monkeypatch.setattr(runner, "freeze_endpoint", lambda: calls.append("freeze"))
    monkeypatch.setattr(runner, "verify_endpoint", lambda: calls.append("verify"))
    return calls


def test_unfinished_geometry_never_opens_h(monkeypatch):
    calls = prepare(monkeypatch, "PAUSED")
    monkeypatch.setattr(runner, "predict", lambda role: pytest.fail("Early evaluation"))
    result = runner.advance(None)
    assert result == {"status": "TRAINING", "fit_id": runner.FIT, "updates": 100}
    assert calls == [("qa", "cpu"), ("qa", "cuda")]


def test_failed_cuda_proof_blocks_training(monkeypatch):
    prepare(monkeypatch, "COMPLETE")

    def qa(device):
        if device == "cuda":
            raise ValueError("CUDA resume failed")

    monkeypatch.setattr(runner, "qa", qa)
    monkeypatch.setattr(runner, "fit", lambda sensors: pytest.fail("Unverified training"))
    with pytest.raises(ValueError, match="CUDA resume"):
        runner.advance(None)


def test_completed_endpoint_is_verified_before_p_h_and_never_opens_v(monkeypatch):
    calls = prepare(monkeypatch, "COMPLETE")

    def predict(role):
        assert calls[:4] == [("qa", "cpu"), ("qa", "cuda"), "freeze", "verify"]
        calls.append(role)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(runner, "predict", predict)
    monkeypatch.setattr(
        runner, "complete_evidence", lambda: {"status": "C1_READY_FOR_INTERPRETATION"}
    )
    assert runner.advance(None)["status"] == "C1_READY_FOR_INTERPRETATION"
    assert calls[4:] == ["H", "P"]


def test_partial_h_evaluation_resumes_before_selection(monkeypatch):
    prepare(monkeypatch, "COMPLETE")
    monkeypatch.setattr(runner, "predict", lambda role: {"status": "PARTIAL", "next_row": 256})
    monkeypatch.setattr(runner, "complete_evidence", lambda: pytest.fail("Partial cohort selected"))
    assert runner.advance(None) == {
        "status": "EVALUATING",
        "role": "H",
        "detail": {"status": "PARTIAL", "next_row": 256},
    }

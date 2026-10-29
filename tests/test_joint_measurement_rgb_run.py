"""The RGB driver must finish both fixed endpoints before opening H."""

import pytest

from operational.joint_measurement import rgb_run as runner


@pytest.mark.parametrize("unfinished", ["FT", "FROZEN"])
def test_no_evaluation_before_both_endpoints(monkeypatch, unfinished):
    monkeypatch.setattr(runner, "require_authorization", lambda: None)
    monkeypatch.setattr(runner, "registered", lambda: {"fits": ["FT", "FROZEN"]})
    monkeypatch.setattr(runner, "qa", lambda: None)
    monkeypatch.setattr(
        runner,
        "fit",
        lambda name: {
            "status": "PAUSED" if name == unfinished else "COMPLETE",
            "completed_updates": 100 if name == unfinished else 12000,
        },
    )

    def forbidden(*args):
        pytest.fail("H/evidence must not open before both endpoints finish")

    monkeypatch.setattr(runner, "freeze_endpoints", forbidden)
    monkeypatch.setattr(runner, "predict", forbidden)
    monkeypatch.setattr(runner, "complete_evidence", forbidden)
    assert runner.advance() == {"status": "TRAINING", "fit_id": unfinished, "updates": 100}


def test_complete_queue_evaluates_only_p_h_after_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "require_authorization", lambda: None)
    monkeypatch.setattr(runner, "registered", lambda: {"fits": ["FT", "FROZEN"]})
    monkeypatch.setattr(runner, "qa", lambda: None)
    monkeypatch.setattr(runner, "fit", lambda name: {"status": "COMPLETE"})
    monkeypatch.setattr(runner, "freeze_endpoints", lambda: calls.append("freeze"))

    def predict(name, role):
        assert calls[0] == "freeze"
        calls.append((name, role))
        return {"status": "COMPLETE"}

    monkeypatch.setattr(runner, "predict", predict)
    monkeypatch.setattr(
        runner, "complete_evidence", lambda: {"status": "RGB_READY_FOR_INTERPRETATION"}
    )
    assert runner.advance()["status"] == "RGB_READY_FOR_INTERPRETATION"
    assert calls[1:] == [
        (name, role) for role in ("H", "P") for name in (runner.BASELINE, "FT", "FROZEN")
    ]

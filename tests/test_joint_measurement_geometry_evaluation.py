"""Geometry evaluation must not open H before a complete, intact endpoint."""

import pytest

from operational.joint_measurement import geometry_evaluate as evaluation
from operational.range_transfer.common import read, sha, write


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    root = run / "adaptive/C1_geometry"
    registration = root / "REGISTRATION.json"
    write(registration, {"fixture": True})
    directory = root / "fits" / evaluation.FIT
    checkpoint = directory / "checkpoint_last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint bytes, not a trained model")
    version = directory / "checkpoint_versions/final.pt"
    version.parent.mkdir()
    version.write_bytes(checkpoint.read_bytes())
    write(
        directory / "CHECKPOINT_RECEIPT.json",
        {
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "completed_updates": 12000,
            "identity_sha256": "identity",
            "checkpoint_sha256": sha(checkpoint),
        },
    )
    write(
        directory / "UPDATE_JOURNAL.json",
        {
            "completed_updates": 12000,
            "durable_updates": 12000,
            "pending_update_upper": 0,
            "identity_sha256": "identity",
        },
    )
    write(
        directory / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": 12000,
            "checkpoint_sha256": sha(checkpoint),
            "version": "final.pt",
        },
    )
    write(directory / "IDENTITY.json", {"registration_sha256": sha(registration)})
    control_identity = run / "fits" / evaluation.CONTROL / "IDENTITY.json"
    write(control_identity, {"test_control": True})
    record = {
        "paired_control": {"checkpoint_sha256": "control", "identity_sha256": sha(control_identity)}
    }
    monkeypatch.setattr(evaluation, "DIRECTORY", root)
    monkeypatch.setattr(evaluation, "RUN", run)
    monkeypatch.setattr(evaluation, "REGISTRATION", registration)
    monkeypatch.setattr(evaluation, "registered", lambda: record)
    monkeypatch.setattr(
        evaluation,
        "freeze_primary",
        lambda: {"endpoints": {evaluation.CONTROL: "control", evaluation.BASELINE: "baseline"}},
    )
    return root, directory, record


def test_complete_endpoint_seals_actual_bytes_and_reused_control(sealed):
    root, directory, _ = sealed
    result = evaluation.freeze_endpoint()
    assert result["endpoints"][evaluation.FIT] == sha(directory / "checkpoint_last.pt")
    assert result["endpoints"][evaluation.CONTROL] == "control"
    assert (root / "ENDPOINTS.json").exists()


@pytest.mark.parametrize(
    "error", ["incomplete", "pending", "alias", "version", "pointer", "identity"]
)
def test_incomplete_or_changed_endpoint_blocks_h_before_loading_any_queries(
    sealed, monkeypatch, error
):
    root, directory, _ = sealed
    if error in {"alias", "version"}:
        path = directory / (
            "checkpoint_last.pt" if error == "alias" else "checkpoint_versions/final.pt"
        )
        path.write_bytes(b"changed")
    else:
        path, field, value = {
            "incomplete": ("CHECKPOINT_RECEIPT.json", "status", "PAUSED"),
            "pending": ("UPDATE_JOURNAL.json", "pending_update_upper", 1),
            "pointer": ("CHECKPOINT_POINTER.json", "completed_updates", 11900),
            "identity": ("IDENTITY.json", "registration_sha256", "changed"),
        }[error]
        payload = read(directory / path)
        payload[field] = value
        write(directory / path, payload)

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("H opened before endpoint admission")

    monkeypatch.setattr(evaluation, "SensorDataset", forbidden_loader)
    with pytest.raises(ValueError, match="not complete"):
        evaluation.predict("H")
    assert not (root / "ENDPOINTS.json").exists()


def test_reused_control_cannot_be_swapped_after_registration(sealed):
    root, _, record = sealed
    record["paired_control"]["checkpoint_sha256"] = "another-control"
    with pytest.raises(ValueError, match="control changed"):
        evaluation.freeze_endpoint()
    assert not (root / "ENDPOINTS.json").exists()


def test_no_v_or_external_selection_in_development_evaluator():
    for role in ("V", "Dev32", "FCWD"):
        with pytest.raises(ValueError, match="only P/H"):
            evaluation.predict(role)

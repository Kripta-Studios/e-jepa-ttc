"""Endpoint and fragment gates for RGB evaluation, without looking at real outcomes."""

import pytest

from operational.joint_measurement import rgb_evaluate as evaluation
from operational.range_transfer.common import read, sha, write


@pytest.fixture
def endpoints(tmp_path, monkeypatch):
    run = tmp_path / "run"
    parent = tmp_path / "parent"
    registration = run / "rgb/REGISTRATION.json"
    write(registration, {"fixture": True})
    baseline = parent / "fits/R_A5/checkpoint_last.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"parent-checkpoint-fixture")
    record = {
        "fits": ["RGB-FT", "RGB-FROZEN"],
        "updates_per_fit": 12000,
        "parent": {"parent_sha256": sha(baseline)},
    }
    for name in record["fits"]:
        directory = run / "rgb/fits" / name
        directory.mkdir(parents=True)
        checkpoint = directory / "checkpoint_last.pt"
        checkpoint.write_bytes(name.encode())
        write(
            directory / "CHECKPOINT_RECEIPT.json",
            {
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "completed_updates": 12000,
                "checkpoint_sha256": sha(checkpoint),
            },
        )
        write(
            directory / "UPDATE_JOURNAL.json",
            {
                "completed_updates": 12000,
                "durable_updates": 12000,
                "pending_update_upper": 0,
            },
        )
        write(directory / "IDENTITY.json", {"registration_sha256": sha(registration)})
    monkeypatch.setattr(evaluation, "RUN", run)
    monkeypatch.setattr(evaluation, "REGISTRATION", registration)
    monkeypatch.setattr(evaluation, "registered", lambda: record)
    monkeypatch.setattr(evaluation, "owner", lambda: parent)
    return run


def test_both_complete_rgb_endpoints_are_required(endpoints):
    path = endpoints / "rgb/fits/RGB-FROZEN/CHECKPOINT_RECEIPT.json"
    receipt = read(path)
    receipt["status"] = "PAUSED"
    write(path, receipt)
    with pytest.raises(ValueError, match="not complete and durable"):
        evaluation.freeze_endpoints()
    assert not (endpoints / "rgb/ENDPOINTS.json").exists()


def test_pending_update_prevents_rgb_freeze(endpoints):
    path = endpoints / "rgb/fits/RGB-FT/UPDATE_JOURNAL.json"
    journal = read(path)
    journal["pending_update_upper"] = 1
    write(path, journal)
    with pytest.raises(ValueError, match="not complete and durable"):
        evaluation.freeze_endpoints()


def test_complete_rgb_freeze_includes_native_rgb_reference(endpoints):
    result = evaluation.freeze_endpoints()
    assert set(result["endpoints"]) == {"RGB-FT", "RGB-FROZEN", evaluation.BASELINE}
    assert read(endpoints / "rgb/ENDPOINTS.json") == result


def test_v_waits_for_metrics_and_cost_even_when_metric_selection_exists(endpoints):
    write(endpoints / "rgb/H_METRIC_SELECTION.json", {"status": "DEVELOPMENT_METRIC_CANDIDATE"})
    with pytest.raises(ValueError, match="metrics and paired cost before V"):
        evaluation.predict("RGB-FT", "V")


@pytest.mark.parametrize("change", ["binding", "interval", "bytes"])
def test_foreign_or_modified_rgb_fragment_rejected(tmp_path, change):
    path = tmp_path / "chunk.parquet"
    path.write_bytes(b"prediction-fragment-fixture")
    receipt = {"sha256": sha(path), "binding": "own", "start": 0, "stop": 256}
    write(path.with_suffix(".json"), receipt)
    evaluation.verify_chunk(path, "own", 0, 256)
    if change == "binding":
        receipt["binding"] = "another-producer"
    elif change == "interval":
        receipt["stop"] = 255
    else:
        path.write_bytes(b"changed")
    write(path.with_suffix(".json"), receipt)
    with pytest.raises(ValueError, match="fragment identity differs"):
        evaluation.verify_chunk(path, "own", 0, 256)

"""RGB system comparisons cannot open V without H admission or mix endpoint provenance."""

import shutil

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement import rgb_system_compare as module
from operational.range_transfer.common import digest, read, sha, write


def test_no_h_candidate_does_not_open_v(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "RUN", tmp_path)
    monkeypatch.setattr(module, "DIRECTORY", tmp_path / "comparison")
    write(module.DIRECTORY / "REGISTRATION.json", {})
    write(tmp_path / "rgb/SELECTION_FINAL.json", {"ranked": []})
    monkeypatch.setattr(module, "register", lambda: {})
    monkeypatch.setattr(module, "finalize_selection", lambda: {"ranked": []})

    def forbidden(*args, **kwargs):
        pytest.fail("No H candidate must not open V or load historical model weights")

    monkeypatch.setattr(module, "admit", forbidden)
    monkeypatch.setattr(module, "role_rows", forbidden)
    monkeypatch.setattr(module, "freeze_endpoints", forbidden)
    result = module.compare()
    assert result["status"] == "NO_H_ADMITTED_RGB_CANDIDATE"
    assert result["V_outcomes_opened"] is False
    assert read(module.DIRECTORY / "RESULT.json") == result


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    write(parent / "V_MANIFEST.json", {"rows_sha256": "rows"})
    registration = tmp_path / "rgb_registration.json"
    write(registration, {"fit": "fixed"})
    monkeypatch.setattr(module, "owner", lambda: parent)
    monkeypatch.setattr(module, "REGISTRATION", registration)
    rows = pd.DataFrame(
        {"sample_token": ["a", "b"], "group_id": ["g", "h"], "target_ttc": [0.7, -3.0]}
    )
    directory = tmp_path / "CANDIDATE"
    directory.mkdir()
    rows.assign(prediction_ttc=[np.nan, -2.0]).to_parquet(
        directory / "predictions.parquet", index=False
    )
    identity = {
        "fit_id": directory.name,
        "role": "V",
        "endpoint": "fixed-weights",
        "registration_sha256": sha(registration),
        "source": sha(module.ROOT / "operational/joint_measurement/rgb_evaluate.py"),
        "scorer_source": sha(module.ROOT / "operational/range_transfer/analyze.py"),
        "rgb_model_source": sha(module.ROOT / "operational/joint_measurement/rgb_model.py"),
        "role_manifest_sha256": sha(parent / "V_MANIFEST.json"),
        "rows_sha256": "rows",
    }
    write(directory / "IDENTITY.json", identity)
    receipt = {
        "status": "COMPLETE",
        "binding": digest(identity),
        "sha256": sha(directory / "predictions.parquet"),
        "rows": len(rows),
    }
    write(directory / "COMPLETE.json", receipt)
    return directory, rows


def test_admitted_candidate_retains_invalid_predictions(candidate):
    directory, rows = candidate
    frame = module.candidate_frame(directory, "fixed-weights", rows)
    assert len(frame) == 2 and np.isnan(frame.prediction_ttc.iloc[0])


@pytest.mark.parametrize(
    "field", ["role", "endpoint", "fit_id", "registration_sha256", "source", "rows_sha256"]
)
def test_rebound_wrong_identity_is_rejected(candidate, field):
    directory, rows = candidate
    identity = read(directory / "IDENTITY.json")
    identity[field] = "changed"
    write(directory / "IDENTITY.json", identity)
    receipt = read(directory / "COMPLETE.json")
    receipt["binding"] = digest(identity)
    write(directory / "COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="identity or coverage"):
        module.candidate_frame(directory, "fixed-weights", rows)


def test_rehashed_changed_target_rejected(candidate):
    directory, rows = candidate
    path = directory / "predictions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "target_ttc"] = 0.8
    frame.to_parquet(path, index=False)
    receipt = read(directory / "COMPLETE.json")
    receipt["sha256"] = sha(path)
    write(directory / "COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="cohort differs"):
        module.candidate_frame(directory, "fixed-weights", rows)


def test_changed_prediction_bytes_rejected(candidate):
    directory, rows = candidate
    path = directory / "predictions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "prediction_ttc"] = 0.8
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="identity or coverage"):
        module.candidate_frame(directory, "fixed-weights", rows)


def test_two_references_full_report_without_reselection(candidate, tmp_path, monkeypatch):
    directory, rows = candidate
    monkeypatch.setattr(module, "RUN", tmp_path)
    monkeypatch.setattr(module, "DIRECTORY", tmp_path / "comparison")
    destination = tmp_path / "rgb/predictions/V" / directory.name
    shutil.copytree(directory, destination)
    selection = {"ranked": [directory.name], "decisions": {directory.name: {"eligible": True}}}
    write(tmp_path / "rgb/SELECTION_FINAL.json", selection)
    selection_hash = sha(tmp_path / "rgb/SELECTION_FINAL.json")
    references = {}
    for reference in module.REFERENCES:
        path = module.owner() / "predictions/V" / f"{reference}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.assign(prediction_ttc=[0.9, -2.5]).to_csv(path, index=False)
        references[reference] = {"V_prediction_sha256": sha(path)}
    contract = {
        "references": references,
        "population": "fixture V",
        "cost_scope": "accuracy only",
        "modalities": {"R_CTX": "RGB", "F_TRUE": "RGB+event"},
    }
    write(module.DIRECTORY / "REGISTRATION.json", contract)
    monkeypatch.setattr(module, "register", lambda: contract)
    monkeypatch.setattr(module, "finalize_selection", lambda: selection)
    monkeypatch.setattr(module, "admit", lambda: {"heads": references})
    monkeypatch.setattr(
        module, "freeze_endpoints", lambda: {"endpoints": {directory.name: "fixed-weights"}}
    )
    monkeypatch.setattr(module, "role_rows", lambda role: rows if role == "V" else None)
    result = module.compare()
    assert set(result["references"]) == {"R_CTX", "F_TRUE"}
    assert set(result["candidates"]) == {directory.name}
    for reference in module.REFERENCES:
        entry = result["comparisons"][reference][directory.name]
        assert entry["gates"]["eligible"] is False
        assert entry["group_bootstrap"]["status"] == "INVALID_PREDICTION_COVERAGE"
    assert result["system_promotion"] is False
    assert result["candidate_reselection"] is False
    assert result["official_comparison"] is False
    assert sha(tmp_path / "rgb/SELECTION_FINAL.json") == selection_hash

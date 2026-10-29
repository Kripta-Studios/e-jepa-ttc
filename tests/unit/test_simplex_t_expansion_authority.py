"""Supplementary authority requires exact ACK and actual reference bytes."""

import json

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import expansion_authority as module


@pytest.mark.parametrize(
    "fault", ["none", "ack", "reference", "size", "pause", "end_pause", "semantics"]
)
def test_verified_reference_boundary(tmp_path, monkeypatch, fault):
    reference = tmp_path / "input.json"
    reference.write_text("fixture")
    entry = {"path": str(reference), "bytes": reference.stat().st_size, "sha256": sha256(reference)}
    ack = tmp_path / "SIMPLEX_T_STAGE70_EXPANSION_ACK.json"
    record = {
        "request_id": "SIMPLEX_T_2026-09-07_EXPANSION_TIME_ROI",
        "request": {"sha256": module.REQUEST_SHA256},
        "scope": {"D1_EXPANSION": {"recognized": True}, "DENSE_OLD": {"recognized": True}},
        "historical_experts": {"preprocessing_replaced": fault == "semantics"},
        "time_and_roi_contract": {"online_causal_availability_at_sensor_anchor_accredited": False},
        "evidence": [entry] * 29,
        "remaining_execution_obligations": ["real replay"],
    }
    ack.write_text(json.dumps(record))
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"shared_coordination": str(tmp_path)}))
    monkeypatch.setattr(
        module, "sha256", lambda p: module.ACK_SHA256 if p == ack and fault != "ack" else sha256(p)
    )
    if fault == "reference":
        reference.write_text("changed")
    elif fault == "size":
        reference.write_text("larger fixture")
    resource_calls = []

    def resources():
        resource_calls.append(True)
        return fault != "pause" and not (fault == "end_pause" and len(resource_calls) == 2)

    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            module.verify_expansion_authority(local, resource_ok=resources)
    else:
        result = module.verify_expansion_authority(local, resource_ok=resources)
        assert result["bytes_hashed"] == 29 * len("fixture")
        assert result["optimizer_updates"] == 0
        assert len(resource_calls) == 2


def test_missing_ack_remains_missing_evidence(tmp_path):
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"shared_coordination": str(tmp_path)}))
    with pytest.raises(ValueError, match="WAITING_EXPANDED_TIME_RECOGNITION"):
        module.verify_expansion_authority(local, resource_ok=lambda: True)

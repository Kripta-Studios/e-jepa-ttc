"""Pinned resource attempts retain pauses and errors without implying extra fits."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.resource_bundle import ResourceAttempt, resource_bundle_members
from e_jepa_ttc.simplex_t.resource_observations import ResourceObservations


@pytest.mark.parametrize(
    "fault", ["none", "bytes", "stage", "freeze", "duplicate", "nan", "count", "scope", "pause"]
)
def test_resource_transport_preserves_all_declared_attempts(tmp_path, fault):
    attempts = []
    for i, status in enumerate(("PAUSED_RESOURCE", "ERROR_BEFORE_RETURN", "PUBLISHED")):
        launch = tmp_path / f"launch{i}.json"
        receipt = tmp_path / f"receipt{i}.json"
        launch.write_text(
            json.dumps(
                dict(
                    schema="simplex_t_frozen_phase_launch_v2",
                    stage="T2",
                    freeze_sha256="a" * 64,
                    resource_receipt=str(receipt),
                )
            )
        )
        launch_hash = hashlib.sha256(launch.read_bytes()).hexdigest()
        observer = ResourceObservations(clock=lambda: 0.0)
        observer.observe(
            dict(
                process_tree_rss_bytes=10, host_available_bytes=20, written_volume_free_bytes=[30]
            ),
            allowed=False,
        )
        record = dict(
            **observer.summary(),
            launch_sha256=launch_hash,
            freeze_sha256="a" * 64,
            stage="T2",
            campaign_complete=False,
            resume_requested=i > 0,
            execution_result_status=status,
            other_reserved_bytes=0,
            own_reserved_bytes=65536,
            disk_floor_after_reservations_bytes=20000000000,
        )
        if i == 0:
            if fault == "freeze":
                record["freeze_sha256"] = "b" * 64
            elif fault == "nan":
                record["elapsed_seconds"] = float("nan")
            elif fault == "count":
                record["denied_admission_samples"] = 2
            elif fault == "scope":
                record["continuous_peak_measurement"] = True
        receipt.write_text(json.dumps(record))
        attempts.append(
            ResourceAttempt(
                launch, launch_hash, receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()
            )
        )
    if fault == "bytes":
        attempts[0].receipt.write_text("{}")
    elif fault == "duplicate":
        attempts.append(attempts[0])
    kwargs = dict(
        work_root=tmp_path,
        freeze_sha256="a" * 64,
        completed_stages={"T2", "T4"} if fault == "stage" else {"T2"},
        resource_ok=lambda: fault != "pause",
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            resource_bundle_members(attempts, **kwargs)
        return
    members = resource_bundle_members(attempts, **kwargs)
    assert len(members) == 6
    for attempt in attempts:
        assert any(m.path == attempt.receipt for m in members.values())
    assert "ERROR_BEFORE_RETURN" in attempts[1].receipt.read_text()

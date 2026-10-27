from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from operational.runpod_migration.pod2_rgb_handoff import (
    Pod2Handoff,
    quota_free_after_reserve,
)


@pytest.fixture(autouse=True)
def fixed_handoff_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deadline tests independent of the day the suite is executed."""
    from operational.runpod_migration import pod2_rgb_handoff

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 27, 12, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(pod2_rgb_handoff, "datetime", FixedDateTime)


def _config(tmp_path: Path) -> dict[str, Any]:
    archive = tmp_path / "global.tar"
    archive.write_bytes(b"x" * (1024 * 1024 + 7))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_receipt = tmp_path / "global.tar.json"
    archive_receipt.write_text(
        json.dumps(
            {
                "status": "READY",
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": archive_sha,
            }
        ),
        encoding="utf-8",
    )
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "ejepa_local_rgb_scientific_reusability_v1",
                "status": "PASS",
                "archive_sha256": archive_sha,
            }
        ),
        encoding="utf-8",
    )
    supplement = tmp_path / "supplement.tar"
    supplement.write_bytes(b"supplement")
    supplement_receipt = tmp_path / "supplement.tar.json"
    supplement_receipt.write_text(
        json.dumps(
            {
                "schema": "ejepa_pod1_operational_supplement_v1",
                "status": "READY",
                "archive_bytes": supplement.stat().st_size,
                "archive_sha256": hashlib.sha256(supplement.read_bytes()).hexdigest(),
                "primary_archive_sha256": archive_sha,
            }
        ),
        encoding="utf-8",
    )
    supplement_verification = tmp_path / "supplement_verification.json"
    supplement_verification.write_text(
        json.dumps(
            {
                "schema": "ejepa_pod1_supplement_local_verification_v1",
                "status": "PASS",
                "archive_sha256": hashlib.sha256(supplement.read_bytes()).hexdigest(),
                "archives": [
                    {
                        "status": "PASS",
                        "archive_sha256": hashlib.sha256(supplement.read_bytes()).hexdigest(),
                        "archive_bytes": supplement.stat().st_size,
                        "all_member_sha256_verified": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    coordinator = tmp_path / "coordinator.json"
    coordinator.write_text(
        json.dumps({"stage": "COMPLETE", "archive_sha256": archive_sha}), encoding="utf-8"
    )
    return {
        "schema": "ejepa_rgb_pod2_handoff_config_v1",
        "max_retries": 2,
        "retry_seconds": 0,
        "remote": {
            "host": "213.173.98.75",
            "user": "root",
            "port": 24247,
            "identity_file": str(tmp_path / "key"),
            "known_hosts_file": str(tmp_path / "known"),
            "root": "/workspace/ejepa_rgb_handoff",
            "deadline_utc": "2026-10-27T14:09:00+00:00",
            "quota_snapshot": "/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json",
            "quota_max_age_seconds": 120,
        },
        "local": {
            "coordinator_status": str(coordinator),
            "scientific_proof": str(proof),
            "supplement_receipt": str(supplement_receipt),
            "supplement_verification": str(supplement_verification),
            "status_path": str(tmp_path / "status.json"),
            "lock_path": str(tmp_path / "lock"),
        },
        "files": [
            {
                "local_path": str(archive),
                "remote_name": "global.tar",
                "identity_receipt": str(archive_receipt),
            },
            {"local_path": str(proof), "remote_name": "proof.json"},
            {
                "local_path": str(supplement),
                "remote_name": "supplement.tar",
                "identity_receipt": str(supplement_receipt),
            },
        ],
    }


def test_handoff_resumes_offset_and_finishes_without_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    worker = Pod2Handoff(config)
    offsets: dict[str, int] = {"global.tar": 13}
    identities: dict[str, str] = {}

    def remote(
        action: str,
        name: str,
        offset: int,
        total: int,
        file_sha: str,
        chunk_sha: str = "-",
        payload: bytes = b"",
    ) -> dict[str, Any]:
        if action == "probe":
            return {"bytes": offsets.get(name, 0), "final": False}
        if action == "write":
            assert len(payload) <= 1024 * 1024
            assert hashlib.sha256(payload).hexdigest() == chunk_sha
            offsets[name] = offset + len(payload)
            identities[name] = file_sha
            return {"bytes": offsets[name]}
        assert action == "finalize" and offsets[name] == total
        return {"status": "COMPLETE", "sha256": identities[name]}

    monkeypatch.setattr(worker, "_remote_call", remote)
    result = worker.run()
    assert result["stage"] == "COMPLETE"
    assert result["extracted"] is False
    assert len(result["files"]) == 3


def test_handoff_fails_closed_before_remote_when_science_is_not_pass(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proof = Path(config["local"]["scientific_proof"])
    value = json.loads(proof.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    proof.write_text(json.dumps(value), encoding="utf-8")
    worker = Pod2Handoff(config)
    with pytest.raises(ValueError, match="gates failed"):
        worker.run()


def test_exact_quota_formula_and_authority() -> None:
    now = 1_800_000_000.0
    snapshot = {
        "schema": "ejepa_event_volume_snapshot_v1",
        "run_root": "/workspace/ejepa_event_runs",
        "volume_root": "/workspace",
        "observed_timestamp": now - 10,
        "quota_bytes": 600_000_000_000,
        "used_bytes": 517_653_755_142,
    }
    assert quota_free_after_reserve(snapshot, now) == 24_364_186_362
    for changed in (
        {**snapshot, "observed_timestamp": now - 121},
        {**snapshot, "observed_timestamp": now + 1},
        {**snapshot, "run_root": "/workspace/wrong"},
        {**snapshot, "volume_root": "/tmp"},
        {**snapshot, "schema": "unknown"},
    ):
        with pytest.raises(ValueError, match="authority/freshness"):
            quota_free_after_reserve(changed, now)

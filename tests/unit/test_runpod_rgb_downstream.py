from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from operational.final_refit.recipe import canonical_sha256
from operational.runpod_migration.rgb_queue import (
    _needs_context_replay,
    rgb_dependency_state,
    run_rgb_downstream,
)
from operational.runpod_migration.rgb_queue_endpoints import route_endpoint


def test_rgb_queue_waits_for_admitted_a5_without_loading_science(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    result = run_rgb_downstream(
        run,
        2,
        deadline_utc=datetime.now(UTC) + timedelta(minutes=1),
    )
    assert result["status"] == "WAITING_DEPENDENCY"
    assert result["stage"] == "WAIT_R_A5"
    assert result["event_branch_executed"] is False
    assert result["event_inputs_read"] is False
    assert rgb_dependency_state(run, 2) == {
        "R_A5": False,
        "R_C2F": False,
        "PAIR_R": False,
        "R_CTX": False,
    }


def test_rgb_queue_rejects_naive_deadline(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="timezone-aware"):
        run_rgb_downstream(run, 2, deadline_utc=datetime.now())


def test_rgb_queue_source_has_no_event_fit_or_context_calls() -> None:
    source = inspect.getsource(run_rgb_downstream)
    assert '"PAIR_R"' in source
    assert '"R_CTX"' in source
    assert "PAIR_E" not in source
    assert "E_CTX" not in source


def test_real_rgb_coverage_receipt_is_complete_and_self_hashed() -> None:
    path = Path("artifacts/runpod_migration_20261026/RGB_DOWNSTREAM_COVERAGE.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.pop("receipt_sha256")
    assert canonical_sha256(receipt) == claimed
    assert receipt["status"] == "PASS"
    assert receipt["required_rgb_files"] > 0
    assert receipt["required_rgb_bytes"] > 9 * 1024**3
    assert set(receipt["rotations"]) == {"2", "3", "4"}
    for roles in receipt["rotations"].values():
        assert set(roles) == {"producer", "refiner", "generation"}
        assert all(role["sample_batch"] for role in roles.values())
    assert receipt["r_ctx"]["event_metadata_required"] is False
    assert receipt["complete_combined_finish_rotation"]["covered_by_rgb_pilot"] is False


def test_external_rgb_endpoint_is_verified_and_routed_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "downstream"
    producer = tmp_path / "producer"
    endpoint = producer / "rotations/2/fits/R_A5"
    endpoint.mkdir(parents=True)
    (run / "MIGRATION_BINDING.json").parent.mkdir(parents=True)
    (run / "MIGRATION_BINDING.json").write_text("{}\n", encoding="utf-8")
    admission = endpoint / "ENDPOINT_ADMISSION.json"
    receipt = endpoint / "CHECKPOINT_RECEIPT.json"
    checkpoint = endpoint / "endpoint.pt"
    lineage = endpoint / "STRUCTURAL_LINEAGE.json"
    for path in (admission, receipt, checkpoint, lineage):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(
        "operational.final_refit.scientific_fold_heads.verify_scientific_admission",
        lambda _path: {
            "rotation": 2,
            "fit_id": "R_A5",
            "checkpoint_receipt_path": str(receipt),
            "checkpoint_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        "operational.runpod_migration.rgb_queue_endpoints._verify_cloud_endpoint",
        lambda *_args: {
            "producer_migration_binding_sha256": "b" * 64,
            "producer_scientific_cloud_admission_sha256": "c" * 64,
            "producer_assignment_sha256": "d" * 64,
        },
    )
    try:
        assert route_endpoint(run, producer, 2, "R_A5") is True
    except OSError as error:
        pytest.skip(f"Windows symlink privilege unavailable: {error}")
    destination = run / "rotations/2/fits/R_A5"
    assert destination.is_symlink()
    assert destination.resolve() == endpoint.resolve()
    record = json.loads((run / "RGB_ENDPOINT_R_A5.json").read_text(encoding="utf-8"))
    assert record["routing_method"] == "DIRECTORY_SYMLINK_NO_CHECKPOINT_COPY"
    assert record["checkpoint_bytes_rewritten"] is False
    assert route_endpoint(run, producer, 2, "R_A5") is True
    other = tmp_path / "other/rotations/2/fits/R_A5"
    other.mkdir(parents=True)
    for path in (
        "ENDPOINT_ADMISSION.json",
        "CHECKPOINT_RECEIPT.json",
        "endpoint.pt",
        "STRUCTURAL_LINEAGE.json",
    ):
        (other / path).write_bytes(path.encode())
    with pytest.raises(ValueError, match="occupied"):
        route_endpoint(run, tmp_path / "other", 2, "R_A5")


def test_endpoint_route_ignores_progress_and_requires_admission(tmp_path: Path) -> None:
    run = tmp_path / "downstream"
    producer = tmp_path / "producer"
    endpoint = producer / "rotations/2/fits/R_C2F"
    endpoint.mkdir(parents=True)
    (endpoint / "PROGRESS.json").write_text("{}\n", encoding="utf-8")
    assert route_endpoint(run, producer, 2, "R_C2F") is False
    assert not (run / "rotations/2/fits/R_C2F").exists()


def test_endpoint_route_rejects_wrong_verified_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "downstream"
    producer = tmp_path / "producer"
    endpoint = producer / "rotations/2/fits/R_A5"
    endpoint.mkdir(parents=True)
    (endpoint / "ENDPOINT_ADMISSION.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "operational.final_refit.scientific_fold_heads.verify_scientific_admission",
        lambda _path: {"rotation": 3, "fit_id": "R_A5"},
    )
    with pytest.raises(ValueError, match="another rotation or fit"):
        route_endpoint(run, producer, 2, "R_A5")


def test_pair_r_readiness_survives_while_c2f_is_still_absent(tmp_path: Path) -> None:
    run = tmp_path / "downstream"
    root = run / "rotations/2/fits"
    for fit in ("R_A5", "PAIR_R"):
        directory = root / fit
        directory.mkdir(parents=True)
        (directory / "ENDPOINT_ADMISSION.json").write_text("{}\n", encoding="utf-8")
    assert rgb_dependency_state(run, 2) == {
        "R_A5": True,
        "R_C2F": False,
        "PAIR_R": True,
        "R_CTX": False,
    }


def test_completed_r_ctx_replays_until_generation_fragment_is_durable(tmp_path: Path) -> None:
    state = {"R_CTX": True}
    assert _needs_context_replay(tmp_path, state) is True
    fragment = tmp_path / "generation/R_CTX/fragment.npz"
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"fragment")
    assert _needs_context_replay(tmp_path, state) is True
    fragment.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    assert _needs_context_replay(tmp_path, state) is False

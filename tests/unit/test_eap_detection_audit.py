"""G0 invariants, failure cases and independent original-format corner fixtures."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.detection.data_audit import (
    DETECTION_FIELDS,
    EAP_CLASSES,
    EAP_REPOSITORY,
    EAP_REVISION,
    audit_sample_submission,
    download_recovery_plan,
    read_pinned_manifest,
    recovery_plan,
    safe_relative_path,
    validate_label_rows,
    validate_media_rows,
)
from e_jepa_ttc.detection.geometry_contract import (
    check_causal_availability,
    geometry_blockers,
    original_bundle_corners,
    validate_calibration,
)


def media_row(split="train", sequence="seq1", timestamp=100):
    return {
        "sample_token": f"{sequence}:{timestamp}",
        "sequence_id": sequence,
        "split": split,
        "rgb_shard_path": f"data/{split}/{sequence}/rgb_shards/rgb-00000.tar",
        "rgb_member_path": f"rgb/{timestamp}.png",
        "events_path": f"data/{split}/{sequence}/events.h5",
        "labels_path": f"data/train/{sequence}/labels.parquet" if split == "train" else None,
        "rgb_exposure_start_timestamp_us": timestamp,
        "rgb_exposure_end_timestamp_us": timestamp + 10,
        "K_event": [[100, 0, 20], [0, 100, 10], [0, 0, 1]],
        "T_event_ego": np.eye(4).tolist(),
    }


def label_row():
    return {
        "sample_token": "seq1:100",
        "sequence_id": "seq1",
        "split": "train",
        "track_id": "0001",
        "category": "car",
        "bbox_3d_ego": [10, 3, 2, 6, 2, 4, 0.37],
        "size": [4, 6, 2],
        "translation": [10, 3, 2],
        "ego_translation": [10, 3, 2],
        "yaw": 0.37,
        "rotation": [np.cos(0.37 / 2), 0, 0, np.sin(0.37 / 2)],
        "velocity": [1, -2],
        "velocity_3d": [1, -2, 0],
    }


def test_original_bundle_matches_independently_executed_upstream_fixture():
    fixture_path = (
        Path(__file__).parents[1] / "fixtures/detection/eap_original_visualizer_corners.json"
    )
    fixture = json.loads(fixture_path.read_text())
    assert fixture["source_commit"] == "2c5f3a74ba5b693b746820a0a355684e3597dc2d"
    assert fixture["scope"] == "original_bundle_visualizer_only_NOT_HF_conversion"
    for case in fixture["cases"]:
        corners = original_bundle_corners(case["bbox_bundle_xyzlhw_yaw"])
        np.testing.assert_allclose(corners, case["corners_ego_m"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(corners.mean(axis=0), [10, 3, 2], atol=1e-12)


def test_non_cubic_fixture_detects_wrong_order_and_yaw_sign():
    actual = original_bundle_corners([10, 3, 2, 6, 2, 4, np.pi / 2])
    # A long local Y edge rotates toward +X under the upstream clockwise convention.
    np.testing.assert_allclose(actual[2] - actual[1], [6, 0, 0], atol=1e-10)
    np.testing.assert_allclose(np.ptp(actual, axis=0), [6, 4, 2], atol=1e-10)


@pytest.mark.parametrize("box", [[1, 2], [0, 0, 0, -1, 2, 3, 0], [0, 0, 0, 1, 2, 3, float("nan")]])
def test_invalid_bundle_box_rejected(box):
    with pytest.raises(ValueError):
        original_bundle_corners(box)


def test_calibration_structure_and_transform_direction_fixture():
    # Independent hand fixture: camera X=-ego Y, camera Y=-ego Z, camera Z=ego X.
    transform = [[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3], [0, 0, 0, 1]]
    intrinsic = [[100, 0, 20], [0, 100, 10], [0, 0, 1]]
    validate_calibration(intrinsic, transform)
    camera = np.asarray(transform) @ [10, 2, 1, 1]
    np.testing.assert_array_equal(camera, [-1, 1, 13, 1])
    projected = np.asarray(intrinsic) @ camera[:3]
    np.testing.assert_allclose(projected[:2] / projected[2], [160 / 13, 230 / 13])
    reflected = np.asarray(transform, dtype=float)
    reflected[0, :3] *= -1
    with pytest.raises(ValueError, match="reflection"):
        validate_calibration(intrinsic, reflected)


def test_future_and_late_arrivals_are_distinct_failures():
    valid = dict(
        exposure_start_us=100,
        exposure_end_us=110,
        max_input_us=115,
        input_available_us=120,
        decision_us=130,
    )
    check_causal_availability(**valid)
    with pytest.raises(ValueError, match="Future or late"):
        check_causal_availability(**{**valid, "max_input_us": 140, "input_available_us": 140})
    with pytest.raises(ValueError, match="Future or late"):
        check_causal_availability(**{**valid, "input_available_us": 140})
    with pytest.raises(ValueError, match="arrive before"):
        check_causal_availability(**{**valid, "input_available_us": 112})


def test_metadata_validation_does_not_certify_rectification_or_availability():
    result = validate_media_rows([media_row(), media_row(timestamp=200)], "train")
    assert result["rows"] == 2
    assert result["structural_check"] == "PASS"
    assert not result["rectification_verified"] and not result["physical_availability_verified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels_path", "data/train/seq2/labels.parquet"),
        ("rgb_shard_path", "../escape.tar"),
        ("rgb_exposure_end_timestamp_us", 90),
        ("rgb_exposure_start_timestamp_us", 100.5),
    ],
)
def test_bad_media_contract_rejected(field, value):
    row = media_row()
    row[field] = value
    with pytest.raises(ValueError):
        validate_media_rows([row], "train")


def test_duplicate_and_rollback_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_media_rows([media_row(), media_row()], "train")
    with pytest.raises(ValueError, match="non-increasing"):
        validate_media_rows([media_row(timestamp=200), media_row()], "train")


def test_test_labels_never_admitted():
    row = media_row(split="test")
    row["labels_path"] = "data/test/seq1/labels.parquet"
    with pytest.raises(ValueError, match="Public test"):
        validate_media_rows([row], "test")


def test_label_relations_are_observations_not_geometry_approval():
    result = validate_label_rows([label_row()], sequence="seq1", media_tokens={"seq1:100"})
    assert result["structural_check"] == "PASS"
    assert result["relations_are_not_a_verified_geometry_conversion"]
    assert result["numeric_relations"]["size_matches_bbox_relation"] == 1
    wrong = label_row()
    wrong["size"] = [6, 4, 2]
    result = validate_label_rows([wrong], sequence="seq1", media_tokens={"seq1:100"})
    assert result["failures"]["size_relation_mismatch"] == 1


def test_bad_labels_count_missing_velocity_and_identity_duplicates():
    row = label_row()
    row["velocity"] = None
    result = validate_label_rows([row, row], sequence="seq1", media_tokens={"seq1:100"})
    assert result["failures"]["invalid_velocity"] == 2
    assert result["failures"]["duplicate_sample_track"] == 1


def test_geometry_gate_cannot_be_opened_with_a_boolean():
    assert len(geometry_blockers({"training_geometry_approved": True})) == 5
    contract_path = Path(__file__).parents[2] / "configs/detection/eap_contract_v1.json"
    contract = json.loads(contract_path.read_text())
    assert geometry_blockers(contract)


def test_geometry_evidence_is_hash_and_claim_bound(tmp_path):
    key = "hf_to_bundle_equivalence"
    contract = {
        "dataset_revision": EAP_REVISION,
        "claims": {
            key: {
                "status": "VERIFIED",
                "evidence_sha256": "a" * 64,
                "evidence_path": "nonexistent.json",
            },
        },
    }
    assert any(key in b for b in geometry_blockers(contract, evidence_root=tmp_path))
    path = tmp_path / "nonexistent.json"
    path.write_text("{}")
    assert any("SHA-256" in b for b in geometry_blockers(contract, evidence_root=tmp_path))
    contract["claims"][key]["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert any("claim/schema" in b for b in geometry_blockers(contract, evidence_root=tmp_path))
    evidence = {
        "artifact_type": "eap_geometry_evidence_v1",
        "claim_id": key,
        "dataset_revision": EAP_REVISION,
        "assertions_passed": True,
        "method": "unit-test synthetic receipt only",
        "source_refs": ["fixture"],
        "limitations": ["not proof of real sensor alignment"],
    }
    path.write_text(json.dumps(evidence))
    contract["claims"][key]["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(key not in b for b in geometry_blockers(contract, evidence_root=tmp_path))
    assert any(key in b for b in geometry_blockers(contract))


def test_foreign_sequence_token_cannot_hide_in_global_train_tokens():
    row = label_row()
    row["sample_token"] = "seq2:100"
    result = validate_label_rows([row], sequence="seq1", media_tokens={"seq1:100", "seq2:100"})
    assert result["failures"]["foreign_sequence_split_or_token"] == 1


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.json",
        "/tmp/outside.json",
        "evidence\\claim.json",
        "evidence/../claim.json",
        "evidence//claim.json",
    ],
)
def test_geometry_evidence_rejects_nonportable_or_escaping_paths(tmp_path, relative_path):
    key = "hf_to_bundle_equivalence"
    contract = {
        "dataset_revision": EAP_REVISION,
        "claims": {
            key: {
                "status": "VERIFIED",
                "evidence_path": relative_path,
                "evidence_sha256": "a" * 64,
            },
        },
    }
    assert any(
        "non-canonical evidence path" in b
        for b in geometry_blockers(contract, evidence_root=tmp_path)
    )


def test_geometry_evidence_requires_receipt_inside_explicit_root(tmp_path):
    key = "hf_to_bundle_equivalence"
    root = tmp_path / "contract_directory"
    (root / "evidence").mkdir(parents=True)
    receipt = {
        "artifact_type": "eap_geometry_evidence_v1",
        "claim_id": key,
        "dataset_revision": EAP_REVISION,
        "assertions_passed": True,
        "method": "unit-test synthetic receipt only",
        "source_refs": ["fixture"],
        "limitations": ["not proof of real sensor alignment"],
    }
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(receipt))
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    contract = {
        "dataset_revision": EAP_REVISION,
        "claims": {
            key: {
                "status": "VERIFIED",
                "evidence_path": "evidence/claim.json",
                "evidence_sha256": digest,
            },
        },
    }
    candidate = root / "evidence/claim.json"
    candidate.symlink_to(outside)
    assert any(
        "outside its declared root" in b for b in geometry_blockers(contract, evidence_root=root)
    )
    candidate.unlink()
    candidate.write_bytes(outside.read_bytes())
    assert all(key not in b for b in geometry_blockers(contract, evidence_root=root))


def sample_zip(path, tokens):
    payload = {
        "meta": {"classes": list(EAP_CLASSES), "fields": list(DETECTION_FIELDS)},
        "results": {token: [] for token in tokens},
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("submission.json", json.dumps(payload))


def test_template_checks_tokens_not_inferred_box_semantics(tmp_path):
    path = tmp_path / "sample.zip"
    sample_zip(path, ["a", "b"])
    result = audit_sample_submission(path, {"a", "b"})
    assert result["check"] == "PASS" and result["all_empty"]
    assert not result["per_object_conventions_verified_by_empty_template"]
    with pytest.raises(ValueError, match="tokens"):
        audit_sample_submission(path, {"a", "c"})


@pytest.mark.parametrize("path", ["../x", "/root/x", "C:/x", "a\\x", "a/../x", "a//x"])
def test_unsafe_relative_paths(path):
    with pytest.raises(ValueError):
        safe_relative_path(path)


def pinned_manifest():
    return {
        "repository": EAP_REPOSITORY,
        "revision": EAP_REVISION,
        "files": {
            "data/train/seq1/labels.parquet": {
                "size_bytes": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            }
        },
    }


def test_revision_must_be_pinned(tmp_path):
    manifest = pinned_manifest()
    manifest["revision"] = "main"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact HF revision"):
        read_pinned_manifest(path)


def test_recovery_is_dry_plan_and_never_requests_rgb_or_events(tmp_path):
    manifest = pinned_manifest()
    manifest["files"]["data/test/seq1/rgb_shards/rgb-00000.tar"] = {"size_bytes": 100}
    manifest["files"]["data/train/seq1/events.h5"] = {"size_bytes": 100}
    plan = recovery_plan(manifest, tmp_path)
    assert [x["path"] for x in plan["files"]] == ["data/train/seq1/labels.parquet"]
    assert plan["download_bytes"] == 3 and not (tmp_path / "data").exists()


def test_recovery_verifies_hash_and_preserves_existing_data(tmp_path, monkeypatch):
    plan = recovery_plan(pinned_manifest(), tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"abc"))
    receipts = download_recovery_plan(plan, tmp_path)
    assert receipts[0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    path = tmp_path / "data/train/seq1/labels.parquet"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Existing file differs"):
        download_recovery_plan(plan, tmp_path)
    assert path.read_bytes() == b"bad"


@pytest.mark.skipif(os.name != "nt", reason="Windows rename no-replace semantics")
def test_windows_recovery_does_not_require_hard_links(tmp_path, monkeypatch):
    """An exFAT destination accepts verified downloads without hard-link support."""
    plan = recovery_plan(pinned_manifest(), tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"abc"))

    def unsupported_link(*args, **kwargs):
        raise OSError("hard links unsupported by destination filesystem")

    monkeypatch.setattr(os, "link", unsupported_link)
    receipts = download_recovery_plan(plan, tmp_path)
    assert receipts[0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert (tmp_path / "data/train/seq1/labels.parquet").read_bytes() == b"abc"
    assert not list(tmp_path.rglob("*.g0-part"))


def test_recovery_never_overwrites_destination_created_at_publish(tmp_path, monkeypatch):
    """The filesystem commit must reject a concurrent writer after the existence check."""
    plan = recovery_plan(pinned_manifest(), tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"abc"))
    operation_name = "rename" if os.name == "nt" else "link"
    original = getattr(os, operation_name)

    def concurrent_writer(source, destination):
        Path(destination).write_bytes(b"other writer")
        return original(source, destination)

    monkeypatch.setattr(os, operation_name, concurrent_writer)
    with pytest.raises(FileExistsError):
        download_recovery_plan(plan, tmp_path)
    assert (tmp_path / "data/train/seq1/labels.parquet").read_bytes() == b"other writer"
    assert not list(tmp_path.rglob("*.g0-part"))


def test_bad_download_does_not_leave_accepted_file(tmp_path, monkeypatch):
    plan = recovery_plan(pinned_manifest(), tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="pinned metadata"):
        download_recovery_plan(plan, tmp_path)
    assert not (tmp_path / "data/train/seq1/labels.parquet").exists()
    assert not list(tmp_path.rglob("*.g0-part"))


def test_preexisting_partial_is_preserved(tmp_path, monkeypatch):
    plan = recovery_plan(pinned_manifest(), tmp_path)
    partial = tmp_path / "data/train/seq1/labels.parquet.g0-part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"preexisting")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"abc"))
    with pytest.raises(FileExistsError):
        download_recovery_plan(plan, tmp_path)
    assert partial.read_bytes() == b"preexisting"


def test_mutated_recovery_plan_rejects_test_media(tmp_path):
    plan = recovery_plan(pinned_manifest(), tmp_path)
    bad = copy.deepcopy(plan)
    bad["files"][0]["path"] = "data/test/seq1/rgb_shards/rgb-00000.tar"
    with pytest.raises(ValueError, match="not allowlisted"):
        download_recovery_plan(bad, tmp_path)


def test_recovery_rejects_parent_symlink_outside_root(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside selected root"):
        recovery_plan(pinned_manifest(), root)
    plan = recovery_plan(pinned_manifest(), tmp_path / "clean")
    with pytest.raises(ValueError, match="outside selected root"):
        download_recovery_plan(plan, root)
    assert not list(outside.iterdir())

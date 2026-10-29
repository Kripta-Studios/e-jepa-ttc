"""Complete RGB return archives fail closed on identity and path violations."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from operational.runpod_migration.rgb_return_archive import (
    ARCHIVE_SCHEMA,
    PILOT_SHA256,
    _add_record,
    _nested_manifest,
    _validate_optional_rgb_operational_recovery_v3,
    _validate_result,
    _validate_rgb_operational_recovery_v2,
    _validate_rgb_operational_recovery_v5,
    _validate_rgb_operational_recovery_v6,
    _validate_rgb_operational_recovery_v7,
    _validate_rgb_path_compatibility,
    _verified_manifest,
    admit_local_rgb_return,
    publish_rgb_return,
    sha256,
)


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _archive(path: Path, *, unsafe: bool = False, rotations: tuple[int, ...] = (2, 3, 4)) -> str:
    payload = b"rgb-output"
    safe_relative = (
        "roots/r4_downstream/output.bin" if rotations == (4,) else "r2/downstream/output.bin"
    )
    relative = "../escape.bin" if unsafe else safe_relative
    files = {
        relative: {
            "source_path": "/workspace/ejepa_technical/r2_rgb_downstream_v1/output.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "status": "READY",
        "rotations": [2, 3, 4],
        "campaign_complete": rotations == (2, 3, 4),
        "individual_rotation_return": rotations == (4,),
        "progress_presented_as_endpoint": False,
        "input_bindings": {"RGB_PILOT.tar": {"sha256": PILOT_SHA256, "included": False}},
        "active_rgb_context_chain": ["V5", "V6", "V7", "V8"],
        "compositional_v8_rotations": [4],
        "sparse_rgb_tar_payloads_included": False,
        "endpoints": {
            f"R{rotation}_{fit}": {}
            for rotation in rotations
            for fit in ("R_A5", "R_C2F", "PAIR_R", "R_CTX")
        },
        "rgb_media_scope_evidence": {
            f"R{rotation}_{role}": {
                "plan_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "physical_root": "/opt/ejepa_rgb_frames",
                "tar_payloads_included": False,
                "tar_receipts": [],
            }
            for rotation in rotations
            for role in ("refiner", "generation")
        },
        "content_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": files,
    }
    manifest["rotations"] = list(rotations)
    manifest_bytes = (json.dumps(manifest) + "\n").encode()
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
        info = tarfile.TarInfo(relative)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return sha256(path)


def test_archive_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "rgb.tar"
    _archive(archive)
    with pytest.raises(ValueError, match="SHA-256"):
        _verified_manifest(archive, "0" * 64)


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "rgb.tar"
    digest = _archive(archive, unsafe=True)
    with pytest.raises(ValueError, match="Unsafe relative"):
        _verified_manifest(archive, digest)


def test_individual_r4_archive_cannot_be_claimed_as_campaign_complete(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "r4.tar"
    digest = _archive(archive, rotations=(4,))
    manifest, _ = _verified_manifest(archive, digest, expected_rotations=(4,))
    assert manifest["campaign_complete"] is False
    assert manifest["individual_rotation_return"] is True
    with pytest.raises(ValueError, match="requested closure"):
        _verified_manifest(archive, digest, expected_rotations=(2, 3, 4))


def test_individual_r4_eviction_requires_byte_and_genealogy_pass(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / "artifacts/runpod_returns/irtz8y3sigd621/individual_r4"
    archive = tmp_path / "r4.tar"
    digest = _archive(archive, rotations=(4,))
    incomplete = admit_local_rgb_return(
        str(archive),
        expected_sha256=digest,
        destination_root=destination,
        repository=repository,
        expected_rotations=(4,),
    )
    assert incomplete["eviction"] is None
    proof = tmp_path / "r4_scientific.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "ejepa_local_rgb_scientific_reusability_v1",
                "status": "PASS",
                "archive_sha256": digest,
                "scientific_reusability_verified": True,
                "selected_root_path_maps_verified": True,
                "three_root_path_maps_verified": False,
                "active_fits_modified": False,
                "rotations": {
                    "4": {
                        "endpoints": {
                            fit: {
                                "original_verify_scientific_admission": "PASS",
                                "admission_sha256": "a" * 64,
                                "checkpoint_sha256": "b" * 64,
                            }
                            for fit in ("PAIR_R", "R_CTX")
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    complete = admit_local_rgb_return(
        str(archive),
        expected_sha256=digest,
        destination_root=destination,
        repository=repository,
        reuse_verification=proof,
        expected_rotations=(4,),
    )
    assert complete["eviction"]["status"] == "PASS"
    assert complete["eviction"]["scope"] == "RAW_R4_REPRODUCIBLE"
    assert complete["eviction"]["remote_deletion_performed"] is False


def test_individual_r4_tampered_genealogy_never_emits_pass(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / "artifacts/runpod_returns/irtz8y3sigd621/individual_r4"
    archive = tmp_path / "r4.tar"
    digest = _archive(archive, rotations=(4,))
    proof = tmp_path / "bad.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "ejepa_local_rgb_scientific_reusability_v1",
                "status": "PASS",
                "archive_sha256": "0" * 64,
                "scientific_reusability_verified": True,
                "selected_root_path_maps_verified": True,
            }
        ),
        encoding="utf-8",
    )
    result = admit_local_rgb_return(
        str(archive),
        expected_sha256=digest,
        destination_root=destination,
        repository=repository,
        reuse_verification=proof,
        expected_rotations=(4,),
    )
    assert result["eviction"]["status"] == "NOT_ELIGIBLE"
    assert result["eviction"]["external_uploader_may_use_as_eviction_authority"] is False


def test_result_missing_required_artifacts_is_not_complete(tmp_path: Path) -> None:
    run = tmp_path / "r2_rgb_downstream_v1"
    run.mkdir()
    (run / "only.bin").write_bytes(b"one")
    result = {
        "schema": "ejepa_cloud_rgb_downstream_queue_v1",
        "stage": "RGB_OOF",
        "status": "COMPLETE",
        "rotation": 2,
        "event_branch_executed": False,
        "event_inputs_read": False,
        "emitted_interface": "PHASE17_RAW_RGB",
        "artifacts": [
            {
                "path": "only.bin",
                "bytes": 3,
                "sha256": hashlib.sha256(b"one").hexdigest(),
            }
        ],
    }
    (run / "RGB_DOWNSTREAM_RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact closure is missing"):
        _validate_result(run, 2)


def test_local_admission_is_isolated_and_never_authorizes_cleanup_without_reuse(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / "artifacts/runpod_returns/irtz8y3sigd621/final_rgb"
    archive = tmp_path / "rgb.tar"
    digest = _archive(archive)
    result = admit_local_rgb_return(
        str(archive),
        expected_sha256=digest,
        destination_root=destination,
        repository=repository,
    )
    assert result["status"] == "PASS"
    assert result["active_fits_replaced"] is False
    assert result["cleanup"]["status"] == "NOT_EVALUATED"
    assert result["cleanup"]["remote_deletion_performed"] is False
    root = Path(result["snapshot_root"])
    assert (root / "r2/downstream/output.bin").read_bytes() == b"rgb-output"
    assert not (root / "CLEANUP_ELIGIBILITY.json").exists()


def test_local_admission_rejects_destination_outside_dedicated_tree(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    archive = tmp_path / "rgb.tar"
    digest = _archive(archive)
    with pytest.raises(ValueError, match="dedicated final_rgb"):
        admit_local_rgb_return(
            str(archive),
            expected_sha256=digest,
            destination_root=tmp_path / "elsewhere",
            repository=repository,
        )


def test_cleanup_is_not_eligible_from_hash_and_boolean_alone(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / "artifacts/runpod_returns/irtz8y3sigd621/final_rgb"
    archive = tmp_path / "rgb.tar"
    digest = _archive(archive)
    reuse = tmp_path / "reuse.json"
    reuse.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scientific_reusability_verified": True,
                "archive_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    result = admit_local_rgb_return(
        str(archive),
        expected_sha256=digest,
        destination_root=destination,
        repository=repository,
        reuse_verification=reuse,
    )
    assert result["cleanup"]["status"] == "NOT_ELIGIBLE"
    assert result["cleanup"]["remote_deletion_performed"] is False


def test_nested_producer_snapshot_manifest_is_packed_as_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint_last.pt"
    source.write_bytes(b"checkpoint")
    records: dict[str, dict[str, object]] = {}
    _add_record(
        records,
        "roots/r2_a5/rotations/2/fits/R_A5/checkpoint_last.pt",
        source,
        tmp_path,
    )
    checkpoint_sha = sha256(source)
    _nested_manifest(
        records,
        prefix="roots/r2_a5",
        rotation=2,
        fit="R_A5",
        endpoint={"completed_updates": 100, "checkpoint_sha256": checkpoint_sha},
    )
    manifest = {
        "content_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    }
    archive = tmp_path / "complete.tar"
    result = publish_rgb_return(manifest, records, archive)
    assert result["status"] == "READY"
    with tarfile.open(archive) as stream:
        nested = stream.getmember("roots/r2_a5/MANIFEST.json")
        assert nested.isfile()
        value = json.load(stream.extractfile(nested))  # type: ignore[arg-type]
    assert value["schema"] == "ejepa_cloud_checkpoint_snapshot_v1"
    assert value["files"]["rotations/2/fits/R_A5/checkpoint_last.pt"]["sha256"] == checkpoint_sha


def test_rgb_path_compatibility_sidecar_is_source_and_migration_bound(tmp_path: Path) -> None:
    downstream = tmp_path / "r2_rgb_downstream_v1"
    downstream.mkdir()
    migration = downstream / "MIGRATION_BINDING.json"
    migration.write_text("{}\n", encoding="utf-8")
    overlay = tmp_path / "runpod_rgb_path_compat_v1.py"
    bootstrap = tmp_path / "runpod_rgb_bootstrap_v1.py"
    overlay.write_text("# overlay\n", encoding="utf-8")
    bootstrap.write_text("# bootstrap\n", encoding="utf-8")
    sidecar = {
        "schema": "ejepa_rgb_native_path_compatibility_v1",
        "status": "PASS",
        "overlay_sha256": sha256(overlay),
        "original_bootstrap_sha256": sha256(bootstrap),
        "migration_binding_sha256": sha256(migration),
        "native_path_isinstance_preserved": True,
        "translation_constructor_unchanged": True,
        "scientific_sources_rewritten": False,
        "checkpoint_bytes_rewritten": False,
    }
    path = downstream / "RGB_NATIVE_PATH_COMPATIBILITY.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    sources = {overlay.name: overlay, bootstrap.name: bootstrap}
    assert _validate_rgb_path_compatibility(downstream, sources) == path
    sidecar["overlay_sha256"] = "0" * 64
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar/source binding"):
        _validate_rgb_path_compatibility(downstream, sources)


def test_rgb_operational_recovery_v2_binds_only_exact_wrapper_dependencies(
    tmp_path: Path,
) -> None:
    downstream = tmp_path / "r4_rgb_downstream_v1"
    downstream.mkdir()
    migration = downstream / "MIGRATION_BINDING.json"
    migration.write_text("{}\n", encoding="utf-8")
    sources = {}
    for name in (
        "runpod_rgb_recovery_v2.py",
        "runpod_rgb_path_compat_v1.py",
        "runpod_quota_race_overlay_v1.py",
    ):
        path = tmp_path / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        sources[name] = path
    sidecar = {
        "schema": "ejepa_rgb_operational_recovery_v2",
        "status": "PASS",
        "source_sha256": sha256(sources["runpod_rgb_recovery_v2.py"]),
        "dependencies": {
            f"/workspace/ops/{name}": sha256(sources[name])
            for name in (
                "runpod_rgb_path_compat_v1.py",
                "runpod_quota_race_overlay_v1.py",
            )
        },
        "migration_binding_sha256": sha256(migration),
        "heartbeat_emitted_by_active_worker": True,
        "heartbeat_interval_seconds": 30,
        "quota_retry_scope": "GNU-du status1 ENOENT only",
        "scientific_sources_rewritten": False,
    }
    path = downstream / "RGB_OPERATIONAL_RECOVERY_V2.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert _validate_rgb_operational_recovery_v2(downstream, sources) == path
    sidecar["heartbeat_interval_seconds"] = 31
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="recovery v2 sidecar/source binding"):
        _validate_rgb_operational_recovery_v2(downstream, sources)


def test_rgb_operational_recovery_v3_is_optional_but_strict_when_present(
    tmp_path: Path,
) -> None:
    downstream = tmp_path / "r4_rgb_downstream_v1"
    downstream.mkdir()
    migration = downstream / "MIGRATION_BINDING.json"
    migration.write_text("{}\n", encoding="utf-8")
    v2 = tmp_path / "runpod_rgb_recovery_v2.py"
    v3 = tmp_path / "runpod_rgb_recovery_v3.py"
    v2.write_text("# frozen v2\n", encoding="utf-8")
    v3.write_text("# recovery v3\n", encoding="utf-8")
    sources = {v2.name: v2, v3.name: v3}
    assert _validate_optional_rgb_operational_recovery_v3(downstream, sources) is None
    sidecar = {
        "schema": "ejepa_rgb_operational_recovery_v3",
        "status": "PASS",
        "source_sha256": sha256(v3),
        "frozen_v2_sha256": sha256(v2),
        "migration_binding_sha256": sha256(migration),
        "heartbeat_receipt_precedes_optional_external_guard": True,
        "external_guard_success_inferred": False,
        "optimizer_updates_inferred": False,
        "frozen_v2_main_delegated": True,
        "scientific_sources_rewritten": False,
    }
    path = downstream / "RGB_OPERATIONAL_RECOVERY_V3.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert _validate_optional_rgb_operational_recovery_v3(downstream, sources) == path
    sidecar["frozen_v2_sha256"] = "0" * 64
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="recovery v3 sidecar/source binding"):
        _validate_optional_rgb_operational_recovery_v3(downstream, sources)


def test_rgb_operational_recovery_v5_requires_exact_audit_and_sources(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    downstream = tmp_path / "r2_rgb_downstream_v1"
    downstream.mkdir()
    migration = downstream / "MIGRATION_BINDING.json"
    migration.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "RGB_OBSERVATION_METADATA_STABILITY_AUDIT.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "ejepa_rgb_observation_metadata_stability_audit_v1",
                "status": "PASS",
                "invariants": {
                    "all_observed_duplicate_differences_are_exact_three_query_fields": True,
                    "sensor_roi_group_role_conflicts": 0,
                    "model_forward_calls": 0,
                    "remote_updates": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    sources = [
        repository / "scripts/runpod_rgb_context_v5.py",
        repository / "scripts/runpod_rgb_recovery_v3.py",
        repository / "operational/oof_rgb_metadata_fix.py",
        audit,
    ]
    value = {
        "schema": "ejepa_rgb_context_metadata_recovery_v5",
        "status": "PASS",
        "source_sha256": sha256(sources[0]),
        "recovery_v3_sha256": sha256(sources[1]),
        "metadata_repair_sha256": sha256(sources[2]),
        "metadata_stability_audit_sha256": sha256(sources[3]),
        "migration_binding_sha256": sha256(migration),
        "excluded_query_reference_fields": [
            "query_age_us",
            "query_time_us",
            "sample_token",
        ],
        "materializer_ast_comparisons_replaced": 1,
        "materializer_and_imported_alias_scoped_to_queue_call": True,
        "rgb_pixels_changed": False,
        "scientific_recipes_changed": False,
        "scientific_inputs_changed": False,
        "scientific_sources_rewritten": False,
    }
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V5.json"
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    assert _validate_rgb_operational_recovery_v5(downstream, sources)[0] == sidecar
    value["excluded_query_reference_fields"] = ["sample_token"]
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="V5 sidecar/source binding"):
        _validate_rgb_operational_recovery_v5(downstream, sources)


def test_rgb_operational_recovery_v6_requires_both_scope_evidence_sets(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    downstream = tmp_path / "r2_rgb_downstream_v1"
    downstream.mkdir()
    migration = downstream / "MIGRATION_BINDING.json"
    migration.write_text("{}\n", encoding="utf-8")
    media_root = tmp_path / "media"
    media_root.mkdir()
    scopes = []
    extra_sources: list[Path] = []
    for role in ("refiner", "generation"):
        scope_root = tmp_path / role
        relative = f"data/train/sample/rgb_shards/{role}.tar"
        receipt_path = media_root / f"{relative}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "ejepa_rgb_sparse_tar_receipt_v1",
            "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
            "relative": relative,
            "members": [{"member": "rgb/frame.png", "compressed_payload_sha256": "3" * 64}],
        }
        receipt["receipt_sha256"] = _canonical(receipt)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        plan = {
            "schema": "ejepa_rgb_frame_supplement_plan_v1",
            "status": "PLANNED_HEADERS_VERIFIED",
            "rotations": [2],
            "roles": [role],
            "tars": [{"relative": relative, "members": receipt["members"]}],
        }
        plan["receipt_sha256"] = _canonical(plan)
        plan_path = scope_root / "plan.json"
        plan_path.parent.mkdir()
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        manifest = {
            "schema": "ejepa_rgb_frame_supplement_v1",
            "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
            "plan_sha256": sha256(plan_path),
            "payload_recompressed": False,
            "decoded_pixel_parity": True,
            "tars": [receipt],
        }
        manifest["receipt_sha256"] = _canonical(manifest)
        manifest_path = scope_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        scope = {
            "rotation": 2,
            "role": role,
            "plan_path": str(plan_path),
            "plan_sha256": sha256(plan_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        }
        scopes.append(scope)
        extra_sources.extend((plan_path, manifest_path))
    config = {
        "schema": "ejepa_rgb_media_scope_config_v1",
        "media_root": str(media_root),
        "scope_evidence_append_only": True,
        "scopes": scopes,
    }
    header = {key: value for key, value in config.items() if key != "scopes"}
    authority = _canonical(header)
    config_path = tmp_path / "RGB_MEDIA_SCOPE_CONFIG.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    v5 = repository / "scripts/runpod_rgb_context_v5.py"
    v6 = repository / "scripts/runpod_rgb_context_v6.py"
    sources = [v5, v6, config_path, *extra_sources]
    sidecar_value = {
        "schema": "ejepa_rgb_frame_gate_routing_v6",
        "status": "PASS",
        "source_sha256": sha256(v6),
        "context_v5_sha256": sha256(v5),
        "media_authority_sha256": authority,
        "initial_media_config_sha256": sha256(config_path),
        "migration_binding_sha256": sha256(migration),
        "reader_routing_scope": "materializer_call_only",
        "original_frames_crops_clocks_changed": False,
        "model_inference_requires_scope_admission": True,
    }
    (downstream / "RGB_OPERATIONAL_RECOVERY_V6.json").write_text(
        json.dumps(sidecar_value), encoding="utf-8"
    )
    for scope in scopes:
        admission = {
            "schema": "ejepa_rgb_media_scope_admission_v1",
            "status": "PASS",
            "rotation": 2,
            "role": scope["role"],
            "scope_evidence_sha256": _canonical(scope),
            "plan_sha256": scope["plan_sha256"],
            "manifest_sha256": scope["manifest_sha256"],
            "authority_sha256": authority,
            "media_root": str(media_root),
            "model_inference_before_admission": False,
            "optimizer_updates_inferred": False,
        }
        (downstream / f"RGB_MEDIA_SCOPE_ADMISSION_R2_{scope['role']}.json").write_text(
            json.dumps(admission), encoding="utf-8"
        )
    _, observed_config, evidence, _ = _validate_rgb_operational_recovery_v6(downstream, 2, sources)
    assert observed_config == config_path
    assert [item[0] for item in evidence] == ["refiner", "generation"]
    (downstream / "RGB_MEDIA_SCOPE_ADMISSION_R2_generation.json").unlink()
    with pytest.raises(FileNotFoundError):
        _validate_rgb_operational_recovery_v6(downstream, 2, sources)


def test_rgb_operational_recovery_v7_is_final_and_source_hash_generic(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    downstream = tmp_path / "r2_rgb_downstream_v1"
    downstream.mkdir()
    migration = downstream / "MIGRATION_BINDING.json"
    migration.write_text("{}\n", encoding="utf-8")
    v6 = repository / "scripts/runpod_rgb_context_v6.py"
    v7 = repository / "scripts/runpod_rgb_context_v7.py"
    sources = [v6, v7]
    value = {
        "schema": "ejepa_rgb_media_stat_cache_v7",
        "status": "PASS",
        "source_sha256": sha256(v7),
        "context_v6_sha256": sha256(v6),
        "migration_binding_sha256": sha256(migration),
        "barrier_fields": [
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ],
        "barrier_scope": "plan_manifest_all_tars_all_sidecars",
        "cache_scope": "current_process_only",
        "missing_or_changed_after_admission": "FAIL_CLOSED",
        "missing_before_admission": "NO_CACHE_WAIT",
    }
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V7.json"
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    assert _validate_rgb_operational_recovery_v7(downstream, sources) == (sidecar, v7)
    value["barrier_fields"] = ["st_size"]
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="V7 sidecar/source binding"):
        _validate_rgb_operational_recovery_v7(downstream, sources)

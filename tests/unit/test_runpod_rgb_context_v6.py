"""V6 gates exact sparse RGB scopes and routes only materializer readers."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from operational import oof_rgb_metadata_fix as metadata_fix
from scripts import runpod_rgb_context_v6 as v6


def _seal(value: dict) -> dict:
    value["receipt_sha256"] = v6.canonical(value)
    return value


def _scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    media = tmp_path / "media"
    monkeypatch.setattr(v6, "MEDIA_ROOT", media)
    relative, member_name = "data/train/sample.tar", "rgb/frame.png"
    tar_path = media / relative
    tar_path.parent.mkdir(parents=True)
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    payload = encoded.getvalue()
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    planned_member = {
        "member": member_name,
        "payload_bytes": len(payload),
        "rotations": [3],
        "roles": ["refiner"],
    }
    plan = _seal(
        {
            "schema": "ejepa_rgb_frame_supplement_plan_v1",
            "status": "PLANNED_HEADERS_VERIFIED",
            "rotations": [3],
            "roles": ["refiner"],
            "tar_count": 1,
            "member_count": 1,
            "tars": [{"relative": relative, "members": [planned_member]}],
        }
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = v6.digest(plan_path)
    actual_member = {
        **planned_member,
        "roles": ["generation", "refiner"],
        "compressed_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "decoded_pixel_parity_sample": True,
        "decoded_rgb_shape": [2, 2, 3],
        "decoded_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
    record = _seal(
        {
            "schema": "ejepa_rgb_sparse_tar_receipt_v1",
            "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
            "plan_sha256": "mixed-plan-is-authorized-by-manifest",
            "relative": relative,
            "sparse_tar_bytes": tar_path.stat().st_size,
            "sparse_tar_sha256": v6.digest(tar_path),
            "members": [actual_member],
        }
    )
    tar_path.with_suffix(".tar.json").write_text(json.dumps(record))
    manifest = _seal(
        {
            "schema": "ejepa_rgb_frame_supplement_v1",
            "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
            "plan_sha256": plan_sha,
            "tar_count": 1,
            "member_count": 1,
            "payload_recompressed": False,
            "decoded_pixel_parity": True,
            "optimizer_updates": 0,
            "tars": [record],
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    scope = {
        "rotation": 3,
        "role": "refiner",
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": v6.digest(manifest_path),
    }
    config = {"scope_map": {"R3_refiner": scope}, "authority_sha256": "authority"}
    return config, tar_path


def test_exact_scope_accepts_manifest_authorized_mixed_plan_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _scope(tmp_path, monkeypatch)
    result = v6.verify_scope(config, 3, "refiner")
    assert result is not None and result["status"] == "PASS"
    assert result["model_inference_before_admission"] is False


def test_missing_scope_waits_but_wrong_manifest_hash_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _scope(tmp_path, monkeypatch)
    assert v6.verify_scope(config, 3, "generation") is None
    config["scope_map"]["R3_refiner"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes differ"):
        v6.verify_scope(config, 3, "refiner")


def test_partial_transfer_waits_before_hashing_any_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, tar_path = _scope(tmp_path, monkeypatch)
    scope = config["scope_map"]["R3_refiner"]
    plan_path, manifest_path = Path(scope["plan_path"]), Path(scope["manifest_path"])
    plan, manifest = json.loads(plan_path.read_text()), json.loads(manifest_path.read_text())
    missing_relative = "data/train/z_last_missing.tar"
    planned_copy = {**plan["tars"][0], "relative": missing_relative}
    plan.update(tar_count=2, member_count=2, tars=[plan["tars"][0], planned_copy])
    plan.pop("receipt_sha256")
    plan_path.write_text(json.dumps(_seal(plan)))
    scope["plan_sha256"] = v6.digest(plan_path)
    record_copy = {**manifest["tars"][0], "relative": missing_relative}
    manifest.update(
        plan_sha256=scope["plan_sha256"],
        tar_count=2,
        member_count=2,
        tars=[manifest["tars"][0], record_copy],
    )
    manifest.pop("receipt_sha256")
    manifest_path.write_text(json.dumps(_seal(manifest)))
    scope["manifest_sha256"] = v6.digest(manifest_path)
    original_digest = v6.digest
    hashed_tars: list[Path] = []

    def observed_digest(path: Path) -> str:
        if path.suffix == ".tar":
            hashed_tars.append(path)
        return original_digest(path)

    monkeypatch.setattr(v6, "digest", observed_digest)
    assert v6.verify_scope(config, 3, "refiner") is None
    assert tar_path.is_file()
    assert hashed_tars == []


def test_missing_tar_sidecar_waits_before_hashing_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, tar_path = _scope(tmp_path, monkeypatch)
    marker = tar_path.with_suffix(".tar.json")
    marker.unlink()
    original_digest = v6.digest
    hashed_tars: list[Path] = []

    def observed_digest(path: Path) -> str:
        if path.suffix == ".tar":
            hashed_tars.append(path)
        return original_digest(path)

    monkeypatch.setattr(v6, "digest", observed_digest)
    assert v6.verify_scope(config, 3, "refiner") is None
    assert hashed_tars == []


def _modules(monkeypatch: pytest.MonkeyPatch):
    materialize = ModuleType("operational.final_refit.materialize")
    pipeline = ModuleType("operational.final_refit.scientific_oof_pipeline")

    class Reader:
        def __init__(self, root, *args, **kwargs):
            self.root = Path(root)

    materialize.TarFrameReader = Reader  # type: ignore[attr-defined]
    pipeline.rgb_raw = lambda *args, **kwargs: "raw"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, materialize.__name__, materialize)
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)
    import operational.final_refit as package

    monkeypatch.setattr(package, "materialize", materialize, raising=False)
    monkeypatch.setattr(package, "scientific_oof_pipeline", pipeline, raising=False)
    return materialize, pipeline, Reader


def test_reader_routes_only_exact_old_root_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialize, _, reader = _modules(monkeypatch)
    monkeypatch.setattr(v6, "MEDIA_ROOT", tmp_path / "media")
    with v6.routed_runtime(tmp_path, 3, tmp_path / "config.json"):
        assert materialize.TarFrameReader(v6.HISTORICAL_ROOT).root == tmp_path / "media"
        assert materialize.TarFrameReader(tmp_path / "current").root == tmp_path / "current"
    assert materialize.TarFrameReader is reader


def test_rgb_raw_gate_precedes_original_for_output_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pipeline, _ = _modules(monkeypatch)
    order: list[str] = []
    pipeline.rgb_raw = lambda *args, **kwargs: order.append("raw") or "ok"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        v6,
        "wait_scope",
        lambda run, config, rotation, role: order.append(f"gate:{rotation}:{role}"),
    )
    with v6.routed_runtime(tmp_path, 3, tmp_path / "config.json"):
        result = pipeline.rgb_raw(None, None, None, tmp_path / "refiner/R_CTX")
    assert order == ["gate:3:refiner", "raw"]
    assert result == "ok"


def test_v6_composition_captures_and_restores_v5_hook() -> None:
    def original(module, clock):
        return None

    v5 = SimpleNamespace(install_queue_heartbeat=original)
    with v6.composed_v5_hook(v5, Path("/config.json")):
        assert v5.install_queue_heartbeat is not original
    assert v5.install_queue_heartbeat is original


def test_shared_phase17_observation_keeps_each_query_clock_unchanged() -> None:
    rows = [
        {"sample_token": "q100", "query_time_us": 100, "query_age_us": 0},
        {"sample_token": "q200", "query_time_us": 200, "query_age_us": 100},
    ]
    stable = {
        "observation_id": "shared",
        "sensor": "front",
        "roi": np.array([1, 2, 3, 4], dtype=np.int64),
        "group_id": "g1",
        "historical_role": "P",
    }
    observations: dict[str, dict] = {}
    query_rows: list[dict] = []
    for row in rows:
        enriched = {**stable, **row}
        old = observations.setdefault("shared", enriched)
        assert metadata_fix.observation_metadata_equal(old, enriched)
        query_rows.append(row)

    assert len(observations) == 1
    assert query_rows == rows
    assert [row["query_time_us"] for row in query_rows] == [100, 200]
    assert [row["query_age_us"] for row in query_rows] == [0, 100]

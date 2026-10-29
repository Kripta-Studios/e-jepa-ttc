from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from operational.independent_refit import producer_rgb_inputs as subject
from operational.independent_refit.population import file_sha256


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _rgb_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(subject, "ROLE_ROWS", 4)
    monkeypatch.setattr(subject, "ROLE_VALID", 3)
    monkeypatch.setattr(subject, "ROLE_INVALID", 1)
    eap = tmp_path / "eap"
    manifests: dict[str, Path] = {}
    public_files: dict[str, object] = {}
    rows = []
    for ordinal, role in enumerate(("P", "H", "V")):
        count = 2 if role != "V" else 3
        sequence = f"seq{role}"
        relative = f"data/train/{sequence}/rgb_shards/rgb-00000.tar"
        tar_path = eap / relative
        tar_path.parent.mkdir(parents=True)
        members = [f"rgb/{index:03d}.png" for index in range(count)]
        with tarfile.open(tar_path, "w") as archive:
            for index, member in enumerate(members):
                buffer = io.BytesIO()
                Image.fromarray(np.full((8, 8, 3), 10 + index, dtype=np.uint8)).save(
                    buffer, format="PNG"
                )
                info = tarfile.TarInfo(member)
                info.size = len(buffer.getvalue())
                archive.addfile(info, io.BytesIO(buffer.getvalue()))
        public_files[relative] = {"sha256": "a" * 64, "size_bytes": tar_path.stat().st_size}
        token = f"token-{role}"
        row = {
            "ordinal": ordinal,
            "source_ordinal": 100 + ordinal,
            "sample_token": token,
            "sequence_id": sequence,
            "group_id": sequence,
            "track_id": f"track-{role}",
            "public_track_id": f"track-{role}",
            "role": role,
            "query_time_us": 1_000_000,
            "rgb_anchor_us": 1_000_000,
            "query_delay_us": 0,
            "producer_frame_count": count,
            "producer_shards": [relative] * count,
            "producer_members": members,
            "producer_times_us": np.arange(count, dtype=np.int64) * 100_000 + 800_000,
            "producer_sensor_times_us": np.arange(count, dtype=np.int64) * 100_000 + 800_000,
            "producer_boxes_xyxy": [np.array([1, 1, 6, 6], dtype=np.float32)] * count,
            "roi_xyxy": np.array([0, 0, 8, 8], dtype=np.float32),
            "target_ttc": 2.0,
            "target_phase": 0.5,
            "mass": 1.0,
            "dino_teacher_available": True,
        }
        rows.append(row)
        rows_path = tmp_path / f"{role}_ROWS.parquet"
        pd.DataFrame([row]).to_parquet(rows_path, index=False)
        manifest_path = tmp_path / f"{role}_MANIFEST.json"
        _json(
            manifest_path,
            {
                "schema": "rgb_port_role_manifest_v1",
                "status": "COMPLETE",
                "role": role,
                "population_size": 1,
                "rows_path": str(rows_path),
                "rows_sha256": file_sha256(rows_path),
                "eap_root": str(eap),
            },
        )
        manifests[role] = manifest_path
    monkeypatch.setattr(
        subject,
        "ROLE_MANIFEST_SHA256",
        {role: file_sha256(path) for role, path in manifests.items()},
    )
    population = tmp_path / "population.parquet"
    independent = [
        {
            "fit_ordinal": index,
            "sample_token": row["sample_token"],
            "sequence_id": row["sequence_id"],
            "track_id": row["track_id"],
            "public_track_id": row["public_track_id"],
            "timestamp_us": row["query_time_us"],
        }
        for index, row in enumerate(rows)
    ]
    independent.append(
        {
            "fit_ordinal": 3,
            "sample_token": "invalid",
            "sequence_id": "bad",
            "track_id": "bad",
            "public_track_id": "bad",
            "timestamp_us": 1,
        }
    )
    pd.DataFrame(independent).to_parquet(population, index=False)
    mask = tmp_path / "role.parquet"
    pd.DataFrame(
        {
            "fit_ordinal": range(4),
            "sample_token": ["token-P", "token-H", "token-V", "invalid"],
            "history_valid": [True, True, True, False],
            "invalid_reason": ["", "", "", "prefix"],
        }
    ).to_parquet(mask, index=False)
    public = tmp_path / "public.json"
    _json(public, {"files": public_files})
    teacher = tmp_path / "teacher.json"
    _json(
        teacher,
        {
            "schema": "independent_phase_train24_teacher_cache_v2",
            "status": "COMPLETE_VERIFIED_CONVERSION",
            "rows": 53_047,
        },
    )
    teacher_role = tmp_path / "teacher_role.json"
    _json(
        teacher_role,
        {
            "schema": "independent_phase_teacher_role_receipt_v1",
            "status": "PASS_EXACT_ROLE_FILTER",
            "rotation": 0,
            "role": "producer",
            "queries": 4,
            "cache_manifest_sha256": file_sha256(teacher),
            "full_train24_exposed": False,
            "generation_excluded": True,
        },
    )
    return {
        **manifests,
        "population": population,
        "mask": mask,
        "public": public,
        "teacher": teacher,
        "teacher_role": teacher_role,
    }


def _build(tmp_path: Path, fixture: dict[str, Path]) -> tuple[Path, dict[str, object]]:
    output = tmp_path / "source"
    value = subject.build_rgb_source_descriptor(
        population_path=fixture["population"],
        role_mask_path=fixture["mask"],
        role_manifests={role: fixture[role] for role in ("P", "H", "V")},
        public_manifest_path=fixture["public"],
        teacher_manifest_path=fixture["teacher"],
        teacher_role_receipt_path=fixture["teacher_role"],
        output_dir=output,
    )
    return output, value


def test_descriptor_strict_join_and_lazy_exact_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _rgb_fixture(tmp_path, monkeypatch)
    output, value = _build(tmp_path, fixture)
    assert value["rows"] == 3
    assert value["role_counts"] == {"P": 1, "H": 1, "V": 1}
    assert value["frame_counts"] == {"2": 2, "3": 1}
    assert value["learned_features_or_normalizers"] is False
    source = subject.IndependentRGBProducerSource(output / "SOURCE.json", max_frame_cache_bytes=1)
    try:
        sample = source.record(0)
        assert sample["rgb"].dtype == np.float32
        assert sample["rgb"].shape == (2, 3, 128, 128)
        assert float(sample["rgb"][0, 0, 0, 0]) == pytest.approx(10 / 255)
        with pytest.raises(ValueError, match="homogeneous"):
            source.batch([0, 2], object())
    finally:
        source.close()


def test_missing_rgb_token_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _rgb_fixture(tmp_path, monkeypatch)
    population = pd.read_parquet(fixture["population"])
    population.loc[0, "sample_token"] = "missing"
    population.to_parquet(fixture["population"], index=False)
    mask = pd.read_parquet(fixture["mask"])
    mask.loc[0, "sample_token"] = "missing"
    mask.to_parquet(fixture["mask"], index=False)
    with pytest.raises(ValueError, match="covered"):
        _build(tmp_path, fixture)


def test_duplicate_or_substituted_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _rgb_fixture(tmp_path, monkeypatch)
    fixture["P"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        _build(tmp_path, fixture)


def test_descriptor_tamper_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _rgb_fixture(tmp_path, monkeypatch)
    output, _ = _build(tmp_path, fixture)
    value = json.loads((output / "SOURCE.json").read_text(encoding="utf-8"))
    value["training_admitted"] = True
    (output / "SOURCE.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        subject.IndependentRGBProducerSource(output / "SOURCE.json")

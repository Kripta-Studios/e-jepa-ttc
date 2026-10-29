from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from operational.independent_refit import producer_inputs as subject


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(path: Path) -> str:
    value = {
        "schema": "train40_lossless_producer_input_prepare_v2_resources",
        "config": {
            "store_full_frame_events": False,
            "store_garl_event_roi": False,
            "store_jepa_event_roi": False,
            "store_event_v4_common_roi": True,
            "event_v4_margin_fraction": 0.25,
            "event_v4_require_precontext": True,
            "event_v4_precontext_fallback": "shifted_event_window",
            "event_v4_bins_per_polarity": 5,
            "event_v4_storage_dtype": "float32",
            "materialize_splits": ["train"],
            "include_rgb": False,
            "include_masks": False,
            "roi_size": 128,
            "event_pixel_diff": 5,
            "preprocessing_device": "cpu",
            "expected_train_rows": subject.SOURCE_ROWS,
        },
        "files": [
            {"path": name, "sha256": digest}
            for name, digest in subject.CRITICAL_ROW_LOCAL_SOURCES.items()
        ],
        "optimizer_updates": 0,
        "lossless_fp32": True,
    }
    return _write_json(path, value)


def _payload(start: int, stop: int, *, token_prefix: str = "token") -> dict[str, np.ndarray]:
    rows = stop - start
    return {
        "ordinals": np.arange(start, stop, dtype=np.int64),
        "events": np.ones((rows, *subject.EVENT_SHAPE), dtype=np.float32),
        "tokens": np.asarray([f"{token_prefix}-{value}" for value in range(start, stop)]),
        "target_ttc": np.ones(rows, dtype=np.float32),
        "visible_heights": np.ones((rows, 2), dtype=np.float32),
        "boxes": np.ones((rows, 3, 4), dtype=np.float32),
        "square": np.ones((rows, 4), dtype=np.float32),
        "motion": np.ones((rows, 18), dtype=np.float32),
        "delta": np.full(rows, 0.1, dtype=np.float32),
    }


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(subject, "SOURCE_ROWS", 3)
    monkeypatch.setattr(subject, "SOURCE_SHARDS", 2)
    monkeypatch.setattr(subject, "SHARD_ROWS", 2)
    root = tmp_path / "producer_inputs"
    root.mkdir()
    references = []
    for index, (start, stop) in enumerate(((0, 2), (2, 3))):
        name = f"shard_{index:05d}.npz"
        path = root / name
        arrays: Any = _payload(start, stop)
        np.savez_compressed(path, **arrays)
        digest = subject.file_sha256(path)
        receipt = {
            "status": "VERIFIED",
            "ordinals": list(range(start, stop)),
            "rows": stop - start,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "freeze_sha256": "pending",
            "exact_compression_parity": True,
            "optimizer_updates": 0,
        }
        references.append((name, start, stop, digest, receipt))
    freeze_path = tmp_path / "PREPARE_FREEZE.json"
    freeze_sha = _freeze(freeze_path)
    monkeypatch.setattr(subject, "PREPARE_FREEZE_SHA256", freeze_sha)
    ordered = []
    for name, start, stop, digest, receipt in references:
        receipt["freeze_sha256"] = freeze_sha
        receipt_sha = _write_json(root / name.replace(".npz", ".json"), receipt)
        ordered.append(
            {
                "name": name,
                "sha256": digest,
                "receipt_sha256": receipt_sha,
                "start": start,
                "stop": stop,
                "bytes": (root / name).stat().st_size,
            }
        )
    manifest_path = tmp_path / "INPUT_MANIFEST.json"
    manifest_sha = _write_json(
        manifest_path,
        {
            "schema": "train40_complete_fragment_manifest_v1",
            "status": "COMPLETE_VERIFIED",
            "kind": "INPUT",
            "row_count": 3,
            "sequence_count": 40,
            "freeze_sha256": freeze_sha,
            "teacher_weights_sha256": None,
            "optimizer_updates": 0,
            "ordered_shards": ordered,
        },
    )
    monkeypatch.setattr(subject, "INPUT_MANIFEST_SHA256", manifest_sha)
    return manifest_path, root, freeze_path


def test_exact_input_manifest_and_record_are_role_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, root, freeze = _source_fixture(tmp_path, monkeypatch)
    shards = subject.verify_input_manifest(manifest, root, freeze)
    index = pd.DataFrame(
        {
            "fit_ordinal": [7, 9],
            "sample_token": ["token-0", "token-2"],
            "group_id": ["g", "g"],
            "sequence_id": ["g", "g"],
            "track_id": ["a", "b"],
            "public_track_id": ["a", "b"],
            "teacher_ordinal": [0, 2],
            "teacher_shard": ["shard_00000.npz", "shard_00001.npz"],
            "teacher_local_row": [0, 0],
            "source_shard": ["shard_00000.npz", "shard_00001.npz"],
            "source_row": [0, 0],
        }
    )
    source = subject.IndependentEventProducerSource(
            source_root=root,
            shards=shards,
            index=index,
            history_invalid_tokens=frozenset(),
            teacher_manifest_sha256="1" * 64,
            teacher_role_receipt_sha256="2" * 64,
            max_cached_shards=1,
    )
    assert len(source) == 2
    assert source.record(1)["sample_token"] == "token-2"
    assert source.record(1)["event_v4_common_roi"].shape == subject.EVENT_SHAPE
    with pytest.raises(IndexError):
        source.record(2)


def test_payload_with_learned_field_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, root, freeze = _source_fixture(tmp_path, monkeypatch)
    shards = subject.verify_input_manifest(manifest, root, freeze)
    payload = _payload(0, 2)
    payload["h8_features"] = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected or learned"):
        subject._validate_payload(payload, shards[0])


def test_history_invalid_rows_are_removed_before_coordinate_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crosswalk = pd.DataFrame(
        {
            "fit_ordinal": [0, 1],
            "sample_token": ["keep", "drop"],
            "group_id": ["g", "g"],
            "sequence_id": ["g", "g"],
            "track_id": ["a", "b"],
            "public_track_id": ["a", "b"],
            "teacher_ordinal": [64, 65],
            "teacher_shard": ["shard_00002.npz", "shard_00002.npz"],
            "teacher_local_row": [0, 1],
        }
    )
    role = pd.DataFrame(
        {
            "rotation": [0, 0],
            "rotation_role": ["producer", "producer"],
            "fit_ordinal": [0, 1],
            "sample_token": ["keep", "drop"],
            "history_valid": [True, False],
            "invalid_reason": ["", "prefix"],
        }
    )
    crosswalk_path = tmp_path / "crosswalk.parquet"
    role_path = tmp_path / "role.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)
    role.to_parquet(role_path, index=False)
    receipt_path = tmp_path / "RECEIPT.json"
    _write_json(receipt_path, {"status": "PASS"})
    monkeypatch.setattr(subject, "CROSSWALK_RECEIPT_SHA256", subject.file_sha256(receipt_path))
    monkeypatch.setattr(subject, "CROSSWALK_PARQUET_SHA256", subject.file_sha256(crosswalk_path))
    selected, invalid_tokens = subject._load_selected_index(
        crosswalk_receipt_path=receipt_path,
        crosswalk_path=crosswalk_path,
        role_mask_path=role_path,
        role_mask_sha256=subject.file_sha256(role_path),
        rotation=0,
    )
    assert selected["sample_token"].tolist() == ["keep"]
    assert invalid_tokens == frozenset({"drop"})


def test_fit_open_fails_before_any_source_use_without_h5_admission(tmp_path: Path) -> None:
    missing = tmp_path / "ADMISSION.json"
    with pytest.raises(ValueError, match="canonical identity"):
        subject.open_event_producer_source(
            source_manifest_path=tmp_path / "manifest.json",
            source_root=tmp_path,
            prepare_freeze_path=tmp_path / "freeze.json",
            crosswalk_receipt_path=tmp_path / "crosswalk.json",
            crosswalk_path=tmp_path / "crosswalk.parquet",
            h5_admission_path=missing,
            plan={},
            teacher_manifest_path=tmp_path / "teacher.json",
            teacher_role_receipt_path=tmp_path / "role.json",
            rotation=0,
        )


def test_batch_rejects_non_boolean_relation_valid(tmp_path: Path) -> None:
    from operational.independent_refit.teacher_cache import TeacherRoleCache

    receipt_path = tmp_path / "ROLE.json"
    receipt = {
        "schema": "independent_phase_teacher_role_receipt_v1",
        "status": "PASS_EXACT_ROLE_FILTER",
        "rotation": 0,
        "role": "producer",
        "cache_manifest_sha256": "1" * 64,
        "queries": 1,
        "full_train24_exposed": False,
        "generation_excluded": True,
    }
    _write_json(receipt_path, receipt)

    class InvalidValidityTeacher(TeacherRoleCache):
        def get(self, sample_token: str) -> tuple[np.ndarray, np.ndarray]:
            assert sample_token == "keep"
            shape = (2, 6, 32, 32)
            return np.zeros(shape, dtype=np.float16), np.ones(shape, dtype=np.uint8)

    teacher = InvalidValidityTeacher(
        root=tmp_path,
        index=pd.DataFrame({"sample_token": ["keep"]}),
        allowed_tokens=frozenset({"keep"}),
        role_receipt_path=receipt_path,
        role_receipt_sha256=subject.file_sha256(receipt_path),
    )
    source = object.__new__(subject.IndependentEventProducerSource)
    source.index = pd.DataFrame({"sample_token": ["keep"]})
    source.population_size = 1
    source.history_invalid_tokens = frozenset()
    source.teacher_manifest_sha256 = "1" * 64
    source.teacher_role_receipt_sha256 = subject.file_sha256(receipt_path)
    source.records = lambda indices: [{"sample_token": "keep"}]  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="relation target contract"):
        source.batch([0], teacher)


def test_real_teacher_role_equals_history_valid_plus_exact_invalid_mask() -> None:
    from operational.independent_refit.lineage import load_rotation_role

    root = Path(__file__).resolve().parents[1]
    fold_path = (
        root / "output/independent_phase/train24_population_v2/FOLD_CONTRACT.json"
    )
    preflight_root = (
        root / "output/independent_phase/raw_history_h5_metadata_preflight_v4"
    )
    role_mask = pd.read_parquet(preflight_root / "roles/rotation_0_producer.parquet")
    nominal = load_rotation_role(fold_path, rotation=0, role="producer")
    nominal_tokens = frozenset(nominal["sample_token"].astype(str))
    valid_tokens = frozenset(
        role_mask.loc[role_mask["history_valid"], "sample_token"].astype(str)
    )
    invalid_tokens = frozenset(
        role_mask.loc[~role_mask["history_valid"], "sample_token"].astype(str)
    )
    assert len(nominal_tokens) == 35_116
    assert len(valid_tokens) == 35_048
    assert len(invalid_tokens) == 68
    assert not valid_tokens & invalid_tokens
    assert nominal_tokens == valid_tokens | invalid_tokens


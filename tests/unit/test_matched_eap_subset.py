from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from e_jepa_ttc.data.matched_eap_subset import (
    ALLOWED_PARQUET_COLUMNS,
    MatchedSubsetConfig,
    build_matched_eap_subset,
    load_matched_manifest,
    validate_code_commit,
)

TEST_COMMIT = "a" * 40


def _write_fixture(root: Path, *, rows: int = 16) -> Path:
    (root / "data").mkdir(parents=True)
    values = [
        {
            "sequence_id": "seq-a",
            "sample_token": f"token-{index}",
            "track_id": "track-a",
            "public_track_id": "public-a",
            "timestamp_us": index * 100_000,
            "frame_timestamps_us": [[index * 100_000 - 100_000, index * 100_000]],
            "events_path": "events/a.h5",
            "event_windows_us": [
                [index * 100_000 - 200_000, index * 100_000 - 100_000],
                [index * 100_000 - 100_000, index * 100_000],
            ],
            "ttc": 99.0,
            "boxes_xyxy": [[0, 0, 1, 1]],
        }
        for index in range(rows)
    ]
    source = root / "data" / "train.parquet"
    pd.DataFrame(values).to_parquet(source)
    return source


def _write_split(root: Path) -> Path:
    split = root / "split.json"
    split.write_text(
        json.dumps({"assignments": {"train": ["seq-a"], "validation": ["seq-b"]}}),
        encoding="utf-8",
    )
    return split


def test_projection_is_exact_and_forbidden_mutation_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_fixture(tmp_path)
    split = _write_split(tmp_path)
    observed: list[list[str]] = []
    original = pd.read_parquet

    def spy(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        observed.append(list(kwargs.get("columns", [])))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    first = build_matched_eap_subset(
        tmp_path,
        split,
        config=MatchedSubsetConfig(stage_sizes=(4, 8, 12, 16)),
        code_commit=TEST_COMMIT,
    )
    assert observed == [list(ALLOWED_PARQUET_COLUMNS)]
    observed.clear()
    source_mutated = pd.read_parquet(source)
    source_mutated["ttc"] = -1000.0
    source_mutated["boxes_xyxy"] = [[[10, 10, 11, 11]]] * len(source_mutated)
    source_mutated.to_parquet(source)
    second = build_matched_eap_subset(
        tmp_path,
        split,
        config=MatchedSubsetConfig(stage_sizes=(4, 8, 12, 16)),
        code_commit=TEST_COMMIT,
    )
    assert observed[-1] == list(ALLOWED_PARQUET_COLUMNS)
    assert first["sampler_order"] == second["sampler_order"]
    assert first["signature"] == second["signature"]
    assert first["source"]["projected_columns"] == list(ALLOWED_PARQUET_COLUMNS)


def test_nested_whole_track_stages_and_signature_mutation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    split = _write_split(tmp_path)
    manifest = build_matched_eap_subset(
        tmp_path,
        split,
        config=MatchedSubsetConfig(stage_sizes=(4, 8, 12, 16)),
        code_commit=TEST_COMMIT,
    )
    assert manifest["selection_report"]["nce_anchor_coverage"] >= 0.8
    assert [stage["actual_row_count"] for stage in manifest["stages"]] == [4, 8, 12]
    assert [
        item["nominal_row_count"] for item in manifest["selection_report"]["unavailable_stages"]
    ] == [16]
    assert len(manifest["rows"]) == manifest["stages"][-1]["nominal_row_count"]
    stage_sets = [set(stage["row_ids"]) for stage in manifest["stages"]]
    assert all(left <= right for left, right in zip(stage_sets, stage_sets[1:], strict=False))
    assert (
        load_matched_manifest(_write_manifest(tmp_path, manifest))["signature"]
        == manifest["signature"]
    )
    tampered = dict(manifest)
    tampered["sampler_order"] = list(reversed(manifest["sampler_order"]))
    with pytest.raises(ValueError, match="signature mismatch"):
        load_matched_manifest(_write_manifest(tmp_path, tampered))


def _write_manifest(root: Path, value: dict[str, object]) -> Path:
    path = (
        root
        / f"manifest-{hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()}.json"
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_annotations_source_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    with pytest.raises(FileNotFoundError):
        build_matched_eap_subset(
            tmp_path / "annotations",
            _write_split(tmp_path),
            code_commit=TEST_COMMIT,
        )


def test_manifest_requires_exact_commit_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40-hex"):
        validate_code_commit("working_tree")
    _write_fixture(tmp_path)
    split = _write_split(tmp_path)
    with pytest.raises(ValueError, match="code_commit"):
        build_matched_eap_subset(tmp_path, split, code_commit="abc")

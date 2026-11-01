from __future__ import annotations

import json
import math
import zipfile

import pytest

from scripts.infer_detection_benchmark import (
    native_box,
    read_template,
    seal_submission,
    validate_media_rows,
)
from scripts.infer_detection_release import digest, write_json_bytes


def row(split="test", labels=None):
    return {"sample_token": "s:1", "split": split, "sequence_id": "s", "labels_path": labels}


def box():
    return dict(
        class_name="car",
        translation_m=[1, 2, 3],
        size_lwh_m=[4, 2, 1],
        yaw_rad=-1.2,
        velocity_mps=[-2, 0.1],
        score=0.9,
        query_index=3,
    )


def test_test_population_rejects_labels_and_wrong_split():
    validate_media_rows([row()], ["s:1"], "test")
    for invalid in ([row(labels="labels.parquet")], [row("train")], [row(), row()]):
        with pytest.raises(ValueError):
            validate_media_rows(invalid, ["s:1"], "test")
    with pytest.raises(ValueError):
        validate_media_rows([row()], ["s:2"], "test")


def test_native_export_keeps_geometry_velocity_and_strips_diagnostics():
    original = box()
    exported = native_box(original)
    assert set(exported) == {
        "class_name",
        "translation_m",
        "size_lwh_m",
        "yaw_rad",
        "velocity_mps",
        "score",
    }
    assert exported["velocity_mps"] == [-2, 0.1]
    assert exported["translation_m"] == [1, 2, 3]
    assert original["query_index"] == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("score", math.nan),
        ("score", 1.2),
        ("velocity_mps", [1]),
        ("size_lwh_m", [1, -2, 3]),
        ("class_name", "unknown"),
        ("translation_m", [math.inf, 0, 1]),
    ],
)
def test_invalid_native_boxes_are_rejected(key, value):
    original = box()
    original[key] = value
    with pytest.raises(ValueError):
        native_box(original)


def test_template_requires_one_root_member_and_unique_tokens(tmp_path):
    p = tmp_path / "sample.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "submission.json",
            json.dumps({"meta": {"format": "eap_native_detection_result"}, "results": {"s:1": []}}),
        )
    assert read_template(p) == ["s:1"]
    with zipfile.ZipFile(p, "a") as z:
        z.writestr("extra.txt", "x")
    with pytest.raises(ValueError):
        read_template(p)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "submission.json",
            '{"meta":{"format":"eap_native_detection_result"},"results":{"s:1":[],"s:1":[]}}',
        )
    with pytest.raises(ValueError, match="Duplicate"):
        read_template(p)


def test_zip_sealing_preserves_empty_frames_and_exact_coverage(tmp_path):
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    write_json_bytes(tmp_path / "ADMISSION.json", {"split": "test"})
    write_json_bytes(chunk / "predictions.json", {"predictions": {"s:1": [box()], "s:2": []}})
    write_json_bytes(
        chunk / "COMPLETE.json", {"files": {"predictions.json": digest(chunk / "predictions.json")}}
    )
    seal_submission(tmp_path, [chunk], ["s:1", "s:2"])
    with zipfile.ZipFile(tmp_path / "submission.zip") as archive:
        assert archive.namelist() == ["submission.json"]
        value = json.loads(archive.read("submission.json"))
    assert value["results"]["s:2"] == []
    assert value["results"]["s:1"][0] == native_box(box())
    with pytest.raises(ValueError, match="Incomplete"):
        seal_submission(tmp_path, [chunk], ["s:1", "s:2", "s:3"])
    (chunk / "predictions.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed"):
        seal_submission(tmp_path, [chunk], ["s:1", "s:2"])

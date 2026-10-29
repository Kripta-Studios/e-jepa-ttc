"""Prove annotation paths are unreachable in the independent inference input route."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.infer_monodgp_eap import (
    AP_RUNTIME_MODULE,
    OPTIONAL_RUNTIME_MODULES,
    RUNTIME_MODULES,
    EAPRGBInputs,
    compare_saved_predictions,
    infer,
    load_runtime_modules,
    select_runtime_sources,
    sha256,
    verify_media_source,
)


def test_rgb_input_does_not_open_nonexistent_label_path(tmp_path, monkeypatch):
    directory = tmp_path / "data/train/a"
    directory.mkdir(parents=True)
    raw = io.BytesIO()
    Image.new("RGB", (32, 24), (3, 4, 5)).save(raw, format="PNG")
    with tarfile.open(directory / "rgb.tar", "w") as shard:
        info = tarfile.TarInfo("rgb/f.png")
        info.size = len(raw.getvalue())
        shard.addfile(info, io.BytesIO(raw.getvalue()))
    missing_labels = "data/train/a/labels_that_do_not_exist.parquet"
    manifest = tmp_path / "data/train.parquet"
    pd.DataFrame(
        [
            {
                "sample_token": "a:f",
                "sequence_id": "a",
                "split": "train",
                "rgb_shard_path": "data/train/a/rgb.tar",
                "rgb_member_path": "rgb/f.png",
                "labels_path": missing_labels,
                "rgb_exposure_start_timestamp_us": 1000,
                "rgb_exposure_end_timestamp_us": 1100,
                "K_event": np.eye(3).tolist(),
                "T_event_ego": np.eye(4).tolist(),
            }
        ]
    ).to_parquet(manifest, index=False)
    original = pd.read_parquet
    opened = []

    def guarded(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        assert Path(path).resolve() == manifest.resolve(), "Inference tried to open labels"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    with EAPRGBInputs(tmp_path, ["a:f"]) as inputs:
        sample = inputs[0]
    assert sample.annotations == []
    assert sample.image.shape == (24, 32, 3)
    assert sample.image[0, 0].tolist() == [3, 4, 5]
    assert not (tmp_path / missing_labels).exists()
    assert opened == [manifest.resolve()]


def prediction(score=0.5):
    return {
        "class_name": "car",
        "translation_m": [1, 2, 10],
        "size_lwh_m": [4, 2, 1],
        "yaw_rad": 0.1,
        "score": score,
    }


def test_prediction_parity_uses_absolute_tolerance():
    assert (
        compare_saved_predictions({"a": [prediction()]}, {"a": [prediction(0.50005)]})["status"]
        == "PASS"
    )
    result = compare_saved_predictions({"a": [prediction()]}, {"a": [prediction(0.5002)]})
    assert result["status"] == "FAIL"
    assert result["mismatches"][0]["reason"] == "score"


def test_prediction_parity_rejects_population_and_class_changes():
    with pytest.raises(ValueError, match="population"):
        compare_saved_predictions({"a": []}, {"b": []})
    other = prediction()
    other["class_name"] = "bus"
    assert compare_saved_predictions({"a": [prediction()]}, {"a": [other]})["status"] == "FAIL"
    assert compare_saved_predictions({"a": [prediction()]}, {"a": []})["status"] == "FAIL"


def test_prediction_parity_compares_optional_velocity_and_presence():
    first, second = prediction(), prediction()
    first["velocity_mps"] = [1, 2]
    result = compare_saved_predictions({"a": [first]}, {"a": [second]})
    assert result["mismatches"][0]["reason"] == "velocity_presence"
    second["velocity_mps"] = [1, 2.001]
    result = compare_saved_predictions({"a": [first]}, {"a": [second]})
    assert result["mismatches"][0]["reason"] == "velocity_mps"


def test_media_source_rejects_altered_calibration_manifest(tmp_path):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"manifest with original K/T")
    expected = sha256(path)
    assert (
        verify_media_source(path, expected)["source_media_verification"] == "VERIFIED_EXPECTED_HASH"
    )
    path.write_bytes(b"manifest with altered K/T")
    with pytest.raises(ValueError, match="media/calibration"):
        verify_media_source(path, expected)


def test_media_source_without_expected_hash_is_only_observed(tmp_path):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"manifest")
    result = verify_media_source(path, None)
    assert result["source_media_sha256"] == sha256(path)
    assert result["expected_source_media_sha256"] is None
    assert result["source_media_verification"] == "OBSERVED_HASH_ONLY"


def test_tokens_json_media_mismatch_fails_before_torch_or_checkpoint(tmp_path, monkeypatch):
    import json

    data = tmp_path / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"unexpected bytes")
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"sample_tokens": ["a:f"], "source_media_sha256": "0" * 64}))
    monkeypatch.setitem(sys.modules, "torch", None)
    args = SimpleNamespace(
        output=tmp_path / "out.json", manifest=None, tokens_json=tokens, data_root=tmp_path
    )
    with pytest.raises(ValueError, match="media/calibration"):
        infer(args)


def runtime_fixture(tmp_path):
    registered = {}
    for index, (_, relative) in enumerate(RUNTIME_MODULES):
        path = tmp_path / "code" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "ORDER = [0]\n"
            if index == 0
            else "from e_jepa_ttc.detection.development_metrics import ORDER\n"
            f"ORDER.append({index})\n"
        )
        if index == 1:
            text += (
                "class EAPRGBDataset:\n    def _frame_labels(self, row):\n"
                "        raise AssertionError('GT')\n"
            )
        path.write_text(text, encoding="utf-8")
        registered[relative.replace("/", "\\")] = sha256(path)
    return registered


def test_runtime_sources_require_all_hashes_and_fail_on_corrupt_snapshot(tmp_path):
    registered = runtime_fixture(tmp_path)
    plan = select_runtime_sources(registered, tmp_path, tmp_path / "unavailable_current")
    assert all(item["source"] == "checkpoint_snapshot" for item in plan)
    assert AP_RUNTIME_MODULE[0][0] not in {item["module"] for item in plan}
    Path(plan[-1]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        select_runtime_sources(registered, tmp_path)
    with pytest.raises(ValueError, match="Missing registered"):
        select_runtime_sources({}, tmp_path)


def test_new_development_snapshot_requires_registered_ap_helper(tmp_path):
    registered = runtime_fixture(tmp_path)
    relative = RUNTIME_MODULES[0][1]
    development = tmp_path / "code" / relative
    development.write_text(
        "from e_jepa_ttc.detection.evaluation_ap_fast import ap101_value\nORDER = [0]\n",
        encoding="utf-8",
    )
    registered[relative.replace("/", "\\")] = sha256(development)
    with pytest.raises(ValueError, match="evaluation_ap_fast"):
        select_runtime_sources(registered, tmp_path, tmp_path / "unavailable_current")


def test_new_development_snapshot_loads_hash_bound_ap_helper_first(tmp_path):
    registered = runtime_fixture(tmp_path)
    development_relative = RUNTIME_MODULES[0][1]
    development = tmp_path / "code" / development_relative
    development.write_text(
        "from e_jepa_ttc.detection.evaluation_ap_fast import ap101_value\nORDER = [0]\n",
        encoding="utf-8",
    )
    registered[development_relative.replace("/", "\\")] = sha256(development)
    _, helper_relative = AP_RUNTIME_MODULE[0]
    helper = tmp_path / "code" / helper_relative
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("def ap101_value(flags, count):\n    return None\n", encoding="utf-8")
    registered[helper_relative.replace("/", "\\")] = sha256(helper)

    plan = select_runtime_sources(registered, tmp_path, tmp_path / "unavailable_current")
    assert plan[0]["module"] == AP_RUNTIME_MODULE[0][0]
    assert plan[0]["sha256"] == sha256(helper)
    assert plan[0]["source"] == "checkpoint_snapshot"


def test_velocity_runtime_requires_registered_extension_and_resolves_other_dependencies(tmp_path):
    registered = runtime_fixture(tmp_path)
    with pytest.raises(ValueError, match="velocity_extension"):
        select_runtime_sources(registered, tmp_path, require_velocity=True)
    for _, relative in OPTIONAL_RUNTIME_MODULES:
        path = tmp_path / "code" / relative
        path.write_text("value = 1\n", encoding="utf-8")
        registered[relative] = sha256(path)
    plan = select_runtime_sources(registered, tmp_path, require_velocity=True)
    assert len(plan) == 9
    assert all(item["source"] == "checkpoint_snapshot" for item in plan)
    names = [item["module"] for item in plan]
    assert names.index("e_jepa_ttc.detection.velocity_extension") < names.index(
        "smoke_monodgp_portable"
    )
    assert names.index("e_jepa_ttc.detection.geometry_contract") < names.index(
        "e_jepa_ttc.detection.monodgp_eap_adapter"
    )


def test_runtime_loader_rechecks_every_hash_before_execution(tmp_path):
    plan = select_runtime_sources(runtime_fixture(tmp_path), tmp_path)
    Path(plan[-1]["path"]).write_text("changed", encoding="utf-8")
    old = sys.modules.get(RUNTIME_MODULES[0][0])
    with pytest.raises(ValueError, match="changed before loading"):
        load_runtime_modules(plan)
    assert sys.modules.get(RUNTIME_MODULES[0][0]) is old


def test_runtime_loader_uses_canonical_dependency_order_and_label_override(tmp_path):
    plan = select_runtime_sources(runtime_fixture(tmp_path), tmp_path)
    previous_modules = {name: sys.modules.get(name) for name, _ in RUNTIME_MODULES}
    parent = sys.modules["e_jepa_ttc.detection"]
    previous_attributes = {
        name.rsplit(".", 1)[-1]: getattr(parent, name.rsplit(".", 1)[-1], None)
        for name, _ in RUNTIME_MODULES
        if "." in name
    }
    try:
        loaded = load_runtime_modules(plan)
        assert loaded[RUNTIME_MODULES[0][0]].ORDER == [0, 1, 2, 3, 4]
        assert parent.eap_rgb_data is loaded[RUNTIME_MODULES[1][0]]
        inputs = type(
            "LabelFree",
            (loaded[RUNTIME_MODULES[1][0]].EAPRGBDataset,),
            {"_frame_labels": EAPRGBInputs._frame_labels},
        )
        assert inputs()._frame_labels({}) == []
    finally:
        for name, value in previous_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for name, value in previous_attributes.items():
            if value is None:
                delattr(parent, name)
            else:
                setattr(parent, name, value)

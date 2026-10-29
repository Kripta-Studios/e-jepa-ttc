from __future__ import annotations

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.detection.diagnostic_export import (
    SCHEMA,
    build_sidecar,
    capture_exposed_outputs,
    validate_json_finite,
)
from scripts.infer_monodgp_eap import infer


def test_capture_is_detached_finite_and_does_not_mutate_outputs():
    logits = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    depth = torch.tensor([[[2.0, -0.5], [4.0, 0.1], [8.0, 0.3]]])
    depth_map = torch.zeros(1, 81, 12, 20)
    outputs = {
        "pred_logits": logits,
        "pred_depth": depth,
        "pred_depth_map_logits": depth_map,
        "aux_outputs": [{"x": 1}],
    }
    before_logits = logits.clone()
    before_depth = depth.clone()

    frame = capture_exposed_outputs(outputs, sample_token="sequence:frame")

    assert torch.equal(outputs["pred_logits"], before_logits)
    assert torch.equal(outputs["pred_depth"], before_depth)
    assert frame["tensors"]["pred_logits"]["selected_shape"] == [3, 4]
    assert frame["tensors"]["pred_logits"]["values"] == logits[0].tolist()
    assert frame["tensors"]["pred_depth"]["values"] == depth[0].tolist()
    assert "pred_depth_map_logits" not in frame["tensors"]
    assert frame["non_query_tensors"]["pred_depth_map_logits"] == {
        "source_shape": [1, 81, 12, 20],
        "dtype": "torch.float32",
        "classification": "known_spatial_output",
        "values_exported": False,
    }
    assert frame["unavailable"]["decoded_box_2d_per_query"].startswith("not_exposed")
    assert frame["unavailable"]["output:aux_outputs"] == "unsupported_exposed_type:list"


def test_capture_rejects_nonfinite_and_bad_batch():
    with pytest.raises(ValueError, match="Non-finite"):
        capture_exposed_outputs(
            {"pred_logits": torch.tensor([[[float("nan")]]])}, sample_token="s:f"
        )
    with pytest.raises(IndexError, match="batch index"):
        capture_exposed_outputs(
            {"pred_logits": torch.zeros(1, 2, 3)}, sample_token="s:f", batch_index=1
        )
    with pytest.raises(ValueError, match=r"\[B,Q,D\]"):
        capture_exposed_outputs({"pred_logits": torch.zeros(2, 1, 2, 3)}, sample_token="s:f")


def test_sidecar_binds_observer_checkpoint_and_prediction(tmp_path: Path):
    observer = tmp_path / "observer.py"
    observer.write_text("# observer\n", encoding="utf-8")
    frame = capture_exposed_outputs({"pred_logits": torch.zeros(1, 2, 3)}, sample_token="s:f")
    sidecar = build_sidecar(
        [frame],
        checkpoint_sha256="a" * 64,
        prediction_output_sha256="b" * 64,
        observer_path=observer,
    )
    validate_json_finite(sidecar)
    assert sidecar["schema"] == SCHEMA
    assert sidecar["sample_count"] == 1
    assert sidecar["claims"] == {
        "changes_predictions": False,
        "ground_truth_access": False,
        "internal_unexposed_values_reconstructed": False,
    }
    json.dumps(sidecar, allow_nan=False)


def test_sidecar_rejects_duplicate_tokens_and_nonfinite_json(tmp_path: Path):
    observer = tmp_path / "observer.py"
    observer.write_text("# observer\n", encoding="utf-8")
    frame = capture_exposed_outputs({"pred_logits": torch.zeros(1, 1, 1)}, sample_token="x")
    with pytest.raises(ValueError, match="unique"):
        build_sidecar(
            [frame, frame],
            checkpoint_sha256="a" * 64,
            prediction_output_sha256="b" * 64,
            observer_path=observer,
        )
    with pytest.raises(ValueError, match="non-finite"):
        validate_json_finite({"bad": float("inf")})


def test_inference_rejects_prediction_sidecar_path_alias(tmp_path: Path):
    shared = tmp_path / "result.json"
    with pytest.raises(ValueError, match="different paths"):
        infer(Namespace(output=shared, diagnostic_sidecar=shared.parent / "." / shared.name))


def _profile_module():
    path = Path(__file__).resolve().parents[2] / "scripts/profile_detection_campaign.py"
    spec = importlib.util.spec_from_file_location("profile_detection_campaign_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiler_manifest_supports_pilot_and_campaign_schema(tmp_path: Path):
    module = _profile_module()
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"train": {"sample_tokens": ["s:a", "s:b"]}}))
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"roles": {"train": {"sample_tokens": ["s:c"]}}}))
    assert module._tokens(pilot, "train") == ["s:a", "s:b"]
    assert module._tokens(campaign, "train") == ["s:c"]


def test_profiler_admission_binds_media_split_revision_and_role(tmp_path: Path):
    module = _profile_module()
    data_root = tmp_path / "data_root"
    media = data_root / "data/train.parquet"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"canonical metadata")
    split = tmp_path / "split.json"
    revision = "a" * 40
    split_payload = {
        "source_split": "train",
        "dataset_revision": revision,
        "assignments": {"train": ["s"], "selection": ["v"], "calibration": ["c"]},
    }
    split.write_text(json.dumps(split_payload), encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_split": "train",
                "dataset_revision": revision,
                "source_media_sha256": digest(media),
                "candidate_split_sha256": digest(split),
                "train": {"sequences": ["s"], "sample_tokens": ["s:f"]},
            }
        ),
        encoding="utf-8",
    )
    tokens, receipt = module._validate_manifest_admission(manifest, split, data_root, "train")
    assert tokens == ["s:f"]
    assert receipt["source_media_sha256"] == digest(media)
    media.write_bytes(b"changed")
    with pytest.raises(ValueError, match="media metadata differs"):
        module._validate_manifest_admission(manifest, split, data_root, "train")


def test_profiler_weighted_loss_requires_positive_finite_scalar():
    module = _profile_module()

    class Criterion:
        weight_dict = {"a": 2.0}

    value = module._weighted_loss(
        {"a": torch.tensor(1.5), "ignored": torch.tensor(9.0)}, Criterion()
    )
    assert value.item() == 3.0
    with pytest.raises(ValueError, match="positive finite"):
        module._weighted_loss({"a": torch.tensor(0.0)}, Criterion())

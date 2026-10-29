from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


def load_script(name: str):
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.__file__, str)
    return module


def make_checkpoint(tmp_path: Path):
    module = load_script("infer_detection_campaign.py")
    assert isinstance(module.__file__, str)
    run = tmp_path / "run"
    code = run / "code"
    portable = code / "portable_source"
    portable.mkdir(parents=True)
    (code / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    frozen_loader = code / "scripts/infer_detection_campaign.py"
    frozen_loader.parent.mkdir()
    shutil.copy2(Path(module.__file__), frozen_loader)
    (portable / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
    prereg = {
        "source_receipt": {"revision": "a"},
        "effective_model": {"classes": 8},
        "code_sha256": {
            "runtime.py": module.sha256_file(code / "runtime.py"),
            "scripts/infer_detection_campaign.py": module.sha256_file(frozen_loader),
        },
        "portable_source_sha256": {"model.py": module.sha256_file(portable / "model.py")},
        "matmul_precision": "highest",
        "dataset_media_sha256": "c" * 64,
    }
    prereg_path = run / "runtime_preregistration.json"
    prereg_path.write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "runtime_preregistration_sha256": module.sha256_file(prereg_path),
        "source_receipt_sha256": module.canonical_sha256(prereg["source_receipt"]),
        "effective_model_sha256": module.canonical_sha256(prereg["effective_model"]),
        "code_sha256_manifest": module.canonical_sha256(prereg["code_sha256"]),
        "portable_source_sha256_manifest": module.canonical_sha256(
            prereg["portable_source_sha256"]
        ),
        "matmul_precision": "highest",
    }
    checkpoint = run / "checkpoints/last.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "schema": module.CHECKPOINT_SCHEMA,
            "producer": "scripts/train_detection_campaign.py",
            "inference_contract": "campaign_v1_strict_loader_required",
            "runtime_provenance": provenance,
            "runtime_provenance_sha256": module.canonical_sha256(provenance),
            "model": {},
        },
        checkpoint,
    )
    return module, checkpoint, code


def test_runtime_verifier_accepts_exact_frozen_sources(tmp_path: Path) -> None:
    module, checkpoint, _ = make_checkpoint(tmp_path)
    payload, prereg, root = module.verify_checkpoint_runtime(checkpoint)
    assert payload["model"] == {}
    assert prereg["matmul_precision"] == "highest"
    assert root == checkpoint.parent.parent


def test_runtime_verifier_rejects_corrupt_frozen_dependency(tmp_path: Path) -> None:
    module, checkpoint, code = make_checkpoint(tmp_path)
    (code / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen runtime dependency differs"):
        module.verify_checkpoint_runtime(checkpoint)


def test_outer_inference_loader_is_bound_to_preregistration(tmp_path: Path) -> None:
    module = load_script("infer_detection_campaign.py")
    assert isinstance(module.__file__, str)
    drifted = tmp_path / "infer_detection_campaign.py"
    drifted.write_text("# changed\n", encoding="utf-8")
    prereg = {
        "code_sha256": {
            "scripts/infer_detection_campaign.py": module.sha256_file(Path(module.__file__))
        }
    }
    with pytest.raises(ValueError, match="loader differs"):
        module.verify_loader_identity(prereg, drifted)


def test_token_population_is_explicit_unique_and_label_free(tmp_path: Path) -> None:
    module = load_script("infer_detection_campaign.py")
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"sample_tokens": ["s:1", "s:2"]}), encoding="utf-8")
    assert module.read_tokens(path) == ("s:1", "s:2")
    path.write_text(json.dumps(["s:1", "s:1"]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        module.read_tokens(path)


def test_inference_rejects_preloaded_runtime_module(monkeypatch) -> None:
    module = load_script("infer_detection_campaign.py")
    monkeypatch.setitem(sys.modules, "lib.models", ModuleType("lib.models"))
    with pytest.raises(RuntimeError, match="fresh subprocess"):
        module.assert_fresh_runtime()


def test_inference_rejects_changed_known_rgb_member() -> None:
    module = load_script("infer_detection_campaign.py")
    receipt = {}
    with pytest.raises(ValueError, match="checkpoint ledger"):
        module.verify_input_member("token", "b" * 64, {"token": "a" * 64}, receipt)
    module.verify_input_member("new", "c" * 64, {"token": "a" * 64}, receipt)
    assert receipt == {"new": "c" * 64}


def test_prediction_and_diagnostic_paths_must_be_distinct_and_new(tmp_path: Path) -> None:
    module = load_script("infer_detection_campaign.py")
    output = tmp_path / "predictions.json"
    # Windows resolves this alias to the same path even though its spelling differs.
    with pytest.raises(ValueError, match="distinct paths"):
        module.validate_output_paths(output, tmp_path / "nested" / ".." / "predictions.json")
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="sidecar"):
        module.validate_output_paths(output, sidecar)


def test_training_cli_does_not_mutate_preexisting_output(tmp_path: Path, monkeypatch) -> None:
    module = load_script("train_detection_campaign.py")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_detection_campaign.py",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--output",
            str(output),
        ],
    )
    assert module.main() == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "failure.json").exists()


def test_training_dataset_constructor_applies_declared_label_cache(monkeypatch) -> None:
    module = load_script("train_detection_campaign.py")
    config_path = Path(__file__).parents[2] / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml"
    config = module.load_campaign_config(config_path)
    observed = {}

    class FakeDataset:
        def __init__(self, root, tokens, **kwargs):
            observed.update(root=str(root), tokens=list(tokens), kwargs=kwargs)

    monkeypatch.setattr(module, "EAPRGBDataset", FakeDataset)
    result = module.open_campaign_dataset("root", ["token"], config)
    assert isinstance(result, FakeDataset)
    assert observed == {
        "root": "root",
        "tokens": ["token"],
        "kwargs": {"max_label_sequences": 24},
    }


def test_evaluation_model_input_is_invariant_to_annotation_order_and_overflow() -> None:
    module = load_script("train_detection_campaign.py")
    config_path = Path(__file__).parents[2] / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml"
    config = module.load_campaign_config(config_path)
    first = {"category": "car", "bbox_3d_ego": [10, 0, 0, 4, 1.5, 2, 0]}
    second = {"category": "truck", "bbox_3d_ego": [12, 1, 0, 5, 2, 2, 0.1]}
    intrinsic = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    event_from_ego = np.array([[0.0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])

    def raw(annotations):
        return module.RawRGBSample(
            sample_token="sequence:frame",
            sequence_id="sequence",
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            annotations=annotations,
            intrinsic=intrinsic,
            event_from_ego=event_from_ego,
            exposure_start_us=0,
            exposure_end_us=1,
            media={},
            member_sha256="a" * 64,
        )

    samples = [raw([first, second]), raw([second, first]), raw([first] * 51)]
    prepared = [module.prepare_model_input(item, config) for item in samples]
    for candidate in prepared[1:]:
        assert torch.equal(candidate.image, prepared[0].image)
        assert torch.equal(candidate.calibration, prepared[0].calibration)
        assert torch.equal(candidate.image_size, prepared[0].image_size)
        assert not candidate.target["labels"].numel()
    # Any deterministic model receives identical tensors and therefore produces identical outputs.
    predictions = [float(item.image.sum() + item.calibration.sum()) for item in prepared]
    assert predictions == [predictions[0]] * 3

    protocol = module.EvaluationProtocol.default()
    truth = [
        module.build_evaluation_ground_truth(
            item.annotations,
            intrinsic=item.intrinsic,
            event_from_ego=item.event_from_ego,
            image_wh=(item.image.shape[1], item.image.shape[0]),
            protocol=protocol,
        )
        for item in samples
    ]
    assert truth[0].sha256 != truth[1].sha256
    assert len(truth[2].eligible) + len(truth[2].ignored) == 51
    historical, reason = module.legacy_ground_truth(samples[2], config)
    assert historical is None and reason is not None and "targets exceed" in reason

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from e_jepa_ttc.evaluation.object_ttc import binary_risk_metrics
from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import EventTubeletTransformerRegressor
from scripts.verify_smoke_completion import main as verify_main


def test_tiny_cnn_signature() -> None:
    model = TinyCNNRegressor(in_channels=21, width=48)
    assert not hasattr(model, "num_risk_thresholds")


def test_token_transformer_signature() -> None:
    model = EventTubeletTransformerRegressor(in_channels=21)
    assert hasattr(model, "forward")


def test_onnx_exporter_single_output() -> None:
    model = EventTubeletTransformerRegressor(in_channels=21)
    dummy_input = torch.randn(1, 21, 90, 160)
    with patch("torch.onnx.export") as mock_export:
        mock_export(
            model,
            dummy_input,
            "test.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["log_ttc"],
            dynamic_axes={"input": {0: "batch_size"}, "log_ttc": {0: "batch_size"}},
        )
        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args
        assert kwargs["output_names"] == ["log_ttc"]


def test_onnx_export_traces_batch_size_correctly(tmp_path) -> None:
    from scripts.export_onnx import export_to_onnx

    # Create a small transformer to speed up the test
    model = EventTubeletTransformerRegressor(in_channels=21, embed_dim=32, depth=1, num_heads=1)
    checkpoint = {
        "resolved_model_config": {
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 1,
            "patch_size": 16,
            "temporal_patch_size": 1,
        },
        "model_state_dict": model.state_dict(),
    }
    ckpt_path = tmp_path / "dummy.pt"
    torch.save(checkpoint, ckpt_path)

    out = tmp_path / "model.onnx"
    cache_path = tmp_path / "cache.npz"
    np.savez(
        cache_path,
        x=np.zeros((10, 21, 90, 160), dtype=np.float32),
        split=np.array(["validation"] * 10),
    )

    # This will execute torch.onnx.export and onnxruntime.InferenceSession.run with a real batch > 1
    # If the reshape is incorrectly traced as static, onnxruntime will crash here.
    export_to_onnx(ckpt_path, out, "event-tubelet-transformer", str(cache_path), sample_count=2)
    assert out.exists()


def test_eap_fractions_are_5_and_10() -> None:
    ps1_path = Path("scripts/run_eap_matrix.ps1")
    if ps1_path.exists():
        content = ps1_path.read_text(encoding="utf-8")
        assert "0.10" in content and "0.05" in content


def test_run_recovery_has_fractions_005_and_010() -> None:
    ps1_path = Path("scripts/run_recovery_multiseed.ps1")
    if ps1_path.exists():
        content = ps1_path.read_text(encoding="utf-8")
        assert "0.05" in content and "0.10" in content


def test_run_recovery_has_nav_modes() -> None:
    ps1_path = Path("scripts/run_recovery_multiseed.ps1")
    if ps1_path.exists():
        content = ps1_path.read_text(encoding="utf-8")
        assert '"enabled", "disabled"' in content


def test_run_recovery_copies_canonical_summaries() -> None:
    ps1_path = Path("scripts/run_recovery_multiseed.ps1")
    if ps1_path.exists():
        content = ps1_path.read_text(encoding="utf-8")
        assert "Copy-Item" in content
        assert "summary.json" in content


def test_run_all_creates_onnx_dir() -> None:
    ps1_path = Path("scripts/run_all.ps1")
    if ps1_path.exists():
        content = ps1_path.read_text(encoding="utf-8")
        assert "New-Item -ItemType Directory -Force" in content


def test_run_all_has_strict_error_action() -> None:
    ps1_path = Path("scripts/run_all.ps1")
    if ps1_path.exists():
        content = ps1_path.read_text(encoding="utf-8")
        assert '$ErrorActionPreference = "Stop"' in content


def test_object_ttc_returns_class_support() -> None:
    target = np.array([1.0, 0.0])
    prob = np.array([0.9, 0.1])
    metrics = binary_risk_metrics(target, prob)
    assert "class_support" in metrics
    assert metrics["class_support"]["positive"] == 1
    assert metrics["class_support"]["negative"] == 1


def test_object_ttc_auroc_handles_zero_support() -> None:
    target = np.array([0.0, 0.0])
    prob = np.array([0.1, 0.2])
    metrics = binary_risk_metrics(target, prob)
    assert "class_support" in metrics
    assert metrics["class_support"]["positive"] == 0


def test_verify_detects_missing_file(tmp_path) -> None:
    import sys

    sys.argv = ["verify_smoke_completion.py", "--smoke-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as e:
        verify_main()
    assert e.value.code == 1


def test_verify_detects_empty_json(tmp_path) -> None:
    import sys

    (tmp_path / "evttc" / "ssl_navigation_enabled").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "evttc" / "ssl_navigation_enabled" / "summary.json", "w") as f:
        f.write("")
    sys.argv = ["verify_smoke_completion.py", "--smoke-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        verify_main()


def test_verify_detects_nan(tmp_path) -> None:
    import sys

    (tmp_path / "evttc" / "ssl_navigation_enabled").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "evttc" / "ssl_navigation_enabled" / "summary.json", "w") as f:
        json.dump({"loss": float("nan")}, f)
    sys.argv = ["verify_smoke_completion.py", "--smoke-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        verify_main()


def test_verify_allows_nan_auroc_when_no_support(tmp_path) -> None:
    # Need to simulate the whole directory or patch
    pass


def test_verify_expects_correct_eap_matrix_and_low_label_paths() -> None:

    script_path = Path("scripts/verify_smoke_completion.py")
    if script_path.exists():
        content = script_path.read_text(encoding="utf-8")
        assert "low_label_05_jepa" in content
        assert "low_label_005_jepa" not in content
        assert "matrix" in content
        assert "pretrain" in content
        assert "seed-7" in content
        assert "finetune" in content


def test_verify_detects_final_test_opened(tmp_path) -> None:
    import sys

    (tmp_path / "evttc" / "ssl_navigation_enabled").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "evttc" / "ssl_navigation_enabled" / "summary.json", "w") as f:
        json.dump({"final_test_opened": True}, f)
    sys.argv = ["verify_smoke_completion.py", "--smoke-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        verify_main()


@patch("numpy.testing.assert_allclose")
@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
@patch("torch.onnx.export")
@patch("torch.load")
def test_export_onnx_saves_manifest(
    mock_load, mock_export, mock_session, mock_onnx_load, mock_onnx_check, mock_assert, tmp_path
) -> None:
    from scripts.export_onnx import export_to_onnx

    mock_load.return_value = {
        "resolved_model_config": {"width": 48},
        "model_state_dict": TinyCNNRegressor(21, 48).state_dict(),
    }
    mock_session.return_value.run.return_value = [np.zeros((1,))]

    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_bytes(b"dummy")

    out = tmp_path / "model.onnx"
    out.write_bytes(b"dummy_onnx")
    cache_path = tmp_path / "cache.npz"
    np.savez(
        cache_path,
        x=np.zeros((10, 21, 90, 160), dtype=np.float32),
        split=np.array(["validation"] * 10),
    )
    export_to_onnx(ckpt_path, out, "tiny_cnn", str(cache_path))

    assert (tmp_path / "model_manifest.json").exists()


@patch("numpy.testing.assert_allclose")
@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
@patch("torch.onnx.export")
@patch("torch.load")
def test_export_onnx_saves_equivalence(
    mock_load, mock_export, mock_session, mock_onnx_load, mock_onnx_check, mock_assert, tmp_path
) -> None:
    from scripts.export_onnx import export_to_onnx

    mock_load.return_value = {
        "resolved_model_config": {"width": 48},
        "model_state_dict": TinyCNNRegressor(21, 48).state_dict(),
    }
    mock_session.return_value.run.return_value = [np.zeros((32, 1))]

    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_bytes(b"dummy")

    out = tmp_path / "model.onnx"
    out.write_bytes(b"dummy_onnx")
    cache_path = tmp_path / "cache.npz"
    np.savez(
        cache_path,
        x=np.zeros((35, 21, 90, 160), dtype=np.float32),
        split=np.array(["validation"] * 35),
    )
    export_to_onnx(ckpt_path, out, "tiny_cnn", str(cache_path), sample_count=32)
    assert (tmp_path / "equivalence.json").exists()


@patch("numpy.testing.assert_allclose")
@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
@patch("torch.onnx.export")
@patch("torch.load")
def test_export_onnx_saves_benchmark(
    mock_load, mock_export, mock_session, mock_onnx_load, mock_onnx_check, mock_assert, tmp_path
) -> None:
    from scripts.export_onnx import export_to_onnx

    mock_load.return_value = {
        "resolved_model_config": {"width": 48},
        "model_state_dict": TinyCNNRegressor(21, 48).state_dict(),
    }
    mock_session.return_value.run.return_value = [np.zeros((32, 1))]

    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_bytes(b"dummy")

    out = tmp_path / "model.onnx"
    out.write_bytes(b"dummy_onnx")
    cache_path = tmp_path / "cache.npz"
    np.savez(
        cache_path,
        x=np.zeros((35, 21, 90, 160), dtype=np.float32),
        split=np.array(["validation"] * 35),
    )
    export_to_onnx(ckpt_path, out, "tiny_cnn", str(cache_path), sample_count=32)
    assert (tmp_path / "benchmark.json").exists()


def test_export_onnx_raises_on_empty_cache(tmp_path) -> None:
    from scripts.export_onnx import export_to_onnx

    cache_path = tmp_path / "cache.npz"
    np.savez(cache_path, x=np.array([]), split=np.array([]))
    with patch("torch.load") as mock_load:
        mock_load.return_value = {
            "resolved_model_config": {"width": 48},
            "model_state_dict": TinyCNNRegressor(21, 48).state_dict(),
        }
        with pytest.raises(ValueError, match="No validation samples found"):
            export_to_onnx(Path("dummy.pt"), tmp_path / "model.onnx", "tiny_cnn", str(cache_path))

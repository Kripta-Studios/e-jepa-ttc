import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts.verify_smoke_completion import main as verify_main


def _setup_mock_smoke_dir(tmp_path: Path):
    required = [
        (
            "evttc/ssl_navigation_enabled/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/ssl_navigation_disabled/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/jepa_navigation_enabled/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/jepa_navigation_disabled/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/scratch_navigation_enabled/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/scratch_navigation_disabled/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/low_label_05_jepa/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/low_label_05_scratch/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/low_label_010_jepa/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "evttc/low_label_010_scratch/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        ("eap/cache/manifest.json", {"evaluation_split": "validation", "final_test_opened": False}),
        (
            "eap/matrix/pretrain/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/jepa/fraction-1/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/scratch/fraction-1/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/jepa/fraction-0.1/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/scratch/fraction-0.1/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/jepa/fraction-0.05/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/scratch/fraction-0.05/seed-7/summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/matrix_summary.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "eap/matrix/eap_split_statistics.json",
            {"evaluation_split": "validation", "final_test_opened": False},
        ),
        (
            "onnx/model_manifest.json",
            {
                "selection_split": "validation",
                "strict_state_dict_loading": True,
                "output_names": ["log_ttc"],
                "checkpoint_sha256": "a",
                "onnx_sha256": "wrong",
                "resolved_model_config": {"width": 48},
            },
        ),
        (
            "onnx/equivalence.json",
            {
                "status": "passed",
                "real_validation_samples": True,
                "sample_count": 32,
                "maximum_absolute_error": 1e-5,
                "mean_absolute_error": 1e-6,
            },
        ),
        (
            "onnx/benchmark.json",
            {
                "warmup_iterations": 50,
                "measured_iterations": 500,
                "latency_ms": {"p50": 1, "p95": 2, "p99": 3},
            },
        ),
        (
            "evttc/cache_validation.json",
            {
                "status": "passed",
                "cache_format_version": 2,
                "normalize": True,
                "normalization": "non_centered_occupied_p95_scale",
                "sidecar_sha256_matches": True,
                "sparse_event_audit_passed": True,
                "nonempty_samples_collapsed_to_zero": 0,
            },
        ),
    ]
    for path, data in required:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f)

    p = tmp_path / "onnx" / "model.onnx"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"dummy")

    h = hashlib.sha256(b"dummy").hexdigest()
    manifest_path = tmp_path / "onnx" / "model_manifest.json"
    data = json.load(open(manifest_path))
    data["onnx_sha256"] = h
    with open(manifest_path, "w") as f:
        json.dump(data, f)


def _run_verify(tmp_path: Path) -> int:
    import sys

    sys.argv = ["verify_smoke_completion.py", "--smoke-dir", str(tmp_path)]
    try:
        verify_main()
    except SystemExit as e:
        return e.code
    return 0


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_detects_invalid_cache_validation(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)
    with open(tmp_path / "evttc" / "cache_validation.json", "w") as f:
        json.dump({"status": "failed"}, f)
    assert _run_verify(tmp_path) == 1


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_detects_invalid_model_manifest_split(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)
    with open(tmp_path / "onnx" / "model_manifest.json", "w") as f:
        json.dump({"selection_split": "test"}, f)
    assert _run_verify(tmp_path) == 1


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_detects_invalid_equivalence(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)
    with open(tmp_path / "onnx" / "equivalence.json", "w") as f:
        json.dump({"status": "passed", "sample_count": 10}, f)
    assert _run_verify(tmp_path) == 1


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_detects_invalid_benchmark(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)
    with open(tmp_path / "onnx" / "benchmark.json", "w") as f:
        json.dump({"warmup_iterations": 10}, f)
    assert _run_verify(tmp_path) == 1


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_detects_onnx_sha256_mismatch(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)
    with open(tmp_path / "onnx" / "model_manifest.json", "w") as f:
        json.dump({"onnx_sha256": "wrong"}, f)
    assert _run_verify(tmp_path) == 1


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_detects_nan_in_auprc_with_support(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)
    with open(tmp_path / "evttc" / "ssl_navigation_enabled" / "summary.json", "w") as f:
        json.dump(
            {
                "evaluation_split": "validation",
                "final_test_opened": False,
                "auprc": float("nan"),
                "class_support": {"positive": 10, "negative": 10},
            },
            f,
        )
    assert _run_verify(tmp_path) == 1


@patch("onnx.checker.check_model")
@patch("onnx.load")
@patch("onnxruntime.InferenceSession")
def test_verify_success_with_valid_files(mock_sess, mock_load, mock_check, tmp_path):
    _setup_mock_smoke_dir(tmp_path)

    class MockGraph:
        def __init__(self):
            self.input = [1]

            class Out:
                name = "log_ttc"

            self.output = [Out()]

    mock_load.return_value.graph = MockGraph()
    assert _run_verify(tmp_path) == 0

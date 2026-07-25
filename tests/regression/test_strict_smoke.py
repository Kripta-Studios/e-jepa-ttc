import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


from e_jepa_ttc.artifacts.hashing import compute_artifact_hash
from scripts.verify_smoke_completion import main as verify_main


def _setup_mock_smoke_dir(tmp_path: Path):
    from e_jepa_ttc.artifacts.protocol import get_repo_root

    required = [
        (
            "evttc/ssl_navigation_enabled/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/ssl_navigation_disabled/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/jepa_navigation_enabled/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/jepa_navigation_disabled/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/scratch_navigation_enabled/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/scratch_navigation_disabled/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/low_label_05_jepa/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/low_label_05_scratch/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/low_label_010_jepa/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "evttc/low_label_010_scratch/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        ("eap/cache/manifest.json", {"evaluation_split": "validation", "final_test_opened": False}),
        (
            "eap/matrix/pretrain/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/jepa/fraction-1/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/scratch/fraction-1/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/jepa/fraction-0.1/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/scratch/fraction-0.1/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/jepa/fraction-0.05/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/finetune/scratch/fraction-0.05/seed-7/summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/matrix_summary.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "eap/matrix/eap_split_statistics.json",
            {"evaluation_splits": ["validation"], "final_test_opened": False},
        ),
        (
            "onnx/model_manifest.json",
            {
                "selection_split": "validation",
                "strict_state_dict_loading": True,
                "output_names": ["log_ttc"],
                "checkpoint_sha256": "c" * 64,
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
                "artifact_type": "architecture_parity_v3",
                "schema_version": "3.0",
                "sample_id_hash": "dummy_hash",
                "evidence_type": "onnx_equivalence",
                "code_commit": json.load(
                    open(
                        get_repo_root()
                        / "artifacts"
                        / "audit"
                        / "recovery_v3"
                        / "frozen_protocol.json"
                    )
                )["code_commit"],
                "protocol_version": "recovery_v3",
                "protocol_sha256": json.load(
                    open(
                        get_repo_root()
                        / "artifacts"
                        / "audit"
                        / "recovery_v3"
                        / "frozen_protocol.json"
                    )
                )["protocol_sha256"],
                "created_at": "2026-07-25",
            },
        ),
        (
            "onnx/benchmark.json",
            {
                "warmup_iterations": 50,
                "iterations": 500,
                "p50_ms": 1,
                "p95_ms": 2,
                "p99_ms": 3,
            },
        ),
        (
            "evttc/cache_validation.json",
            {
                "schema_version": "3.0",
                "status": "passed",
                "audit_mode": "exhaustive",
                "evidence_type": "real_smoke",
                "cache_format_version": 2,
                "cache_path": "cache.npz",
                "cache_sha256_computed": "b" * 64,
                "cache_sha256_declared": "b" * 64,
                "normalize": True,
                "normalization": "non_centered_occupied_p95_scale",
                "normalizer_source_split": "train",
                "normalizer_origins_verified": True,
                "sidecar_sha256_matches": True,
                "sample_count_total": 10,
                "sample_count_audited": 10,
                "nonempty_samples_collapsed_to_zero": 0,
                "checks": {},
                "failures": [],
                "warnings": [],
                "provenance": {},
            },
        ),
        ("onnx_selection.json", {"status": "passed", "checkpoint_sha256": "c" * 64}),
        ("phase_1_evttc.json", {"status": "passed"}),
        ("phase_2_eap.json", {"status": "passed"}),
        ("phase_4_onnx.json", {"status": "passed"}),
        ("phase_eap_cache.json", {"status": "passed"}),
        ("phase_eap_matrix_inner.json", {"status": "passed"}),
    ]
    from e_jepa_ttc.artifacts.protocol import get_repo_root

    frozen_protocol = json.load(
        open(get_repo_root() / "artifacts" / "audit" / "recovery_v3" / "frozen_protocol.json")
    )
    expected_commit = frozen_protocol["code_commit"]
    expected_protocol_hash = frozen_protocol["protocol_sha256"]

    for path, data in required:
        if "artifact_type" not in data:
            data["artifact_type"] = "stage_record_v3"  # Dummy for schema mock
        if "code_commit" not in data:
            data["code_commit"] = expected_commit
        if "protocol_sha256" not in data:
            data["protocol_sha256"] = expected_protocol_hash

        if "artifact_sha256" not in data:
            data["artifact_sha256"] = compute_artifact_hash(data)
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
    data["artifact_sha256"] = compute_artifact_hash(data)
    with open(manifest_path, "w") as f:
        json.dump(data, f)


def _run_verify(tmp_path: Path) -> int:
    import sys
    from unittest.mock import patch

    sys.argv = ["verify_smoke_completion.py", "--smoke-dir", str(tmp_path)]
    with patch("scripts.verify_smoke_completion.jsonschema.validate"):
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
                "evaluation_splits": ["validation"],
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

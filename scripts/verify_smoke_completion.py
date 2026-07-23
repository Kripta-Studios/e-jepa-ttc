import argparse
import json
import math
from pathlib import Path


def _check_nans(obj: dict | list | float | str, key: str = "", parent_dict: dict = None) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _check_nans(v, k, obj):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if _check_nans(v, key, parent_dict):
                return True
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            if key in ("auprc", "auroc"):
                if parent_dict and "class_support" in parent_dict:
                    pos = parent_dict["class_support"].get("positive", 1)
                    neg = parent_dict["class_support"].get("negative", 1)
                    if pos == 0 or neg == 0:
                        return False
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, default=Path("artifacts/smoke/current"))
    args = parser.parse_args()

    smoke_dir = args.smoke_dir

    required_files = [
        smoke_dir / "evttc" / "ssl_navigation_enabled" / "summary.json",
        smoke_dir / "evttc" / "ssl_navigation_disabled" / "summary.json",
        smoke_dir / "evttc" / "jepa_navigation_enabled" / "summary.json",
        smoke_dir / "evttc" / "jepa_navigation_disabled" / "summary.json",
        smoke_dir / "evttc" / "scratch_navigation_enabled" / "summary.json",
        smoke_dir / "evttc" / "scratch_navigation_disabled" / "summary.json",
        smoke_dir / "evttc" / "low_label_05_jepa" / "summary.json",
        smoke_dir / "evttc" / "low_label_05_scratch" / "summary.json",
        smoke_dir / "evttc" / "low_label_010_jepa" / "summary.json",
        smoke_dir / "evttc" / "low_label_010_scratch" / "summary.json",
        smoke_dir / "eap" / "cache" / "manifest.json",
        smoke_dir / "eap" / "matrix" / "pretrain" / "seed-7" / "summary.json",
        smoke_dir
        / "eap"
        / "matrix"
        / "finetune"
        / "jepa"
        / "fraction-1"
        / "seed-7"
        / "summary.json",
        smoke_dir
        / "eap"
        / "matrix"
        / "finetune"
        / "scratch"
        / "fraction-1"
        / "seed-7"
        / "summary.json",
        smoke_dir
        / "eap"
        / "matrix"
        / "finetune"
        / "jepa"
        / "fraction-0.1"
        / "seed-7"
        / "summary.json",
        smoke_dir
        / "eap"
        / "matrix"
        / "finetune"
        / "scratch"
        / "fraction-0.1"
        / "seed-7"
        / "summary.json",
        smoke_dir
        / "eap"
        / "matrix"
        / "finetune"
        / "jepa"
        / "fraction-0.05"
        / "seed-7"
        / "summary.json",
        smoke_dir
        / "eap"
        / "matrix"
        / "finetune"
        / "scratch"
        / "fraction-0.05"
        / "seed-7"
        / "summary.json",
        smoke_dir / "eap" / "matrix" / "matrix_summary.json",
        smoke_dir / "eap" / "matrix" / "eap_split_statistics.json",
        smoke_dir / "onnx" / "model.onnx",
        smoke_dir / "onnx" / "model_manifest.json",
        smoke_dir / "onnx" / "equivalence.json",
        smoke_dir / "onnx" / "benchmark.json",
    ]

    manifest = {
        "status": "failed",
        "exit_code": 1,
        "all_required_artifacts_exist": False,
        "pytorch_onnx_equivalence_passed": False,
        "final_test_opened": False,
        "required_artifact_count": len(required_files),
        "validated_artifact_count": 0,
        "completed_stages": [],
        "failed_stages": [],
        "commit": "unknown",
    }

    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        manifest["commit"] = commit
    except Exception:
        pass

    all_exist = True
    validated_count = 0
    for req in required_files:
        if not req.exists():
            manifest["failed_stages"].append(f"Missing required artifact: {req}")
            all_exist = False
            continue

        if req.suffix == ".json":
            with open(req, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not data:
                        manifest["failed_stages"].append(f"Empty JSON object: {req}")
                        all_exist = False
                        continue

                    # Summary validations
                    if req.name == "summary.json":
                        valid_splits = data.get("evaluation_splits") or data.get("validation_splits") or []
                        valid_split = data.get("evaluation_split")
                        has_validation_samples = data.get("validation_samples", 0) > 0
                        if "validation" not in valid_splits and valid_split != "validation" and not has_validation_samples:
                            manifest["failed_stages"].append(
                                f"Missing/invalid evaluation/validation splits in {req}"
                            )
                            all_exist = False
                            continue
                        if data.get("final_test_opened") is True:
                            manifest["final_test_opened"] = True
                            manifest["failed_stages"].append(f"Final test opened in {req}")
                            all_exist = False
                            continue

                    # General NaN checking
                    if _check_nans(data, parent_dict=data if isinstance(data, dict) else None):
                        manifest["failed_stages"].append(f"NaN or Inf found in {req}")
                        all_exist = False
                        continue

                    # ONNX model manifest validation
                    if req.name == "model_manifest.json":
                        split = data.get("selection_split")
                        if split in ("test", "CPLA-high"):
                            manifest["failed_stages"].append(
                                "ONNX model selected using forbidden split"
                            )
                            all_exist = False
                            continue
                        if not data.get("strict_state_dict_loading"):
                            manifest["failed_stages"].append("ONNX strict loading not true")
                            all_exist = False
                            continue
                        if data.get("output_names") != ["log_ttc"]:
                            manifest["failed_stages"].append("ONNX output names invalid")
                            all_exist = False
                            continue
                        if not data.get("checkpoint_sha256") or not data.get("onnx_sha256"):
                            manifest["failed_stages"].append("Missing hashes in model manifest")
                            all_exist = False
                            continue
                        if not data.get("resolved_model_config"):
                            manifest["failed_stages"].append("Missing model config in manifest")
                            all_exist = False
                            continue

                    # ONNX equivalence validation
                    if req.name == "equivalence.json":
                        if (
                            data.get("status") == "passed"
                            and data.get("real_validation_samples") is True
                            and data.get("sample_count", 0) >= 32
                            and data.get("maximum_absolute_error", 1.0) <= 1e-4
                            and data.get("mean_absolute_error", 1.0) <= 1e-5
                        ):
                            manifest["pytorch_onnx_equivalence_passed"] = True
                        else:
                            manifest["failed_stages"].append("ONNX equivalence failed rules")
                            all_exist = False
                            continue

                    # ONNX benchmark validation
                    if req.name == "benchmark.json":
                        if "p50_ms" not in data or "p95_ms" not in data or "p99_ms" not in data:
                            manifest["failed_stages"].append("Benchmark missing percentiles")
                            all_exist = False
                            continue
                        if data.get("iterations", 0) < 500:
                            manifest["failed_stages"].append("Benchmark insufficient iterations")
                            all_exist = False
                            continue

                    validated_count += 1
                except json.JSONDecodeError:
                    manifest["failed_stages"].append(f"Invalid JSON: {req}")
                    all_exist = False
        elif req.suffix == ".onnx":
            try:
                import onnx
                import onnxruntime

                onnx_model = onnx.load(req)
                onnx.checker.check_model(onnx_model)
                if len(onnx_model.graph.input) != 1:
                    manifest["failed_stages"].append("ONNX must have exactly 1 input")
                    all_exist = False
                    continue
                if (
                    len(onnx_model.graph.output) != 1
                    or onnx_model.graph.output[0].name != "log_ttc"
                ):
                    manifest["failed_stages"].append(
                        "ONNX must have exactly 1 output named log_ttc"
                    )
                    all_exist = False
                    continue
                if req.stat().st_size == 0:
                    manifest["failed_stages"].append("ONNX file is empty")
                    all_exist = False
                    continue

                # Check SHA256 matches manifest if manifest exists
                manifest_path = req.parent / "model_manifest.json"
                if manifest_path.exists():
                    import hashlib

                    h = hashlib.sha256(req.read_bytes()).hexdigest()
                    with open(manifest_path, encoding="utf-8") as fm:
                        if json.load(fm).get("onnx_sha256") != h:
                            manifest["failed_stages"].append("ONNX SHA256 mismatch")
                            all_exist = False
                            continue

                onnxruntime.InferenceSession(str(req))
                validated_count += 1
            except Exception as e:
                manifest["failed_stages"].append(f"ONNX loading/checking failed: {e}")
                all_exist = False
        else:
            validated_count += 1

    manifest["validated_artifact_count"] = validated_count

    if (
        all_exist
        and validated_count == len(required_files)
        and manifest["pytorch_onnx_equivalence_passed"]
    ):
        manifest["all_required_artifacts_exist"] = True
        manifest["status"] = "passed"
        manifest["exit_code"] = 0
        manifest["completed_stages"] = [str(p) for p in required_files]
        manifest["failed_stages"] = []

    with open(smoke_dir / "completion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if manifest["status"] == "failed":
        print(f"Smoke completion gate failed. Missing/invalid: {manifest['failed_stages']}")
        exit(1)


if __name__ == "__main__":
    main()

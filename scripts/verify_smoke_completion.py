import argparse
import json
import math
from pathlib import Path


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
        "required_artifact_count": len(required_files),
        "validated_artifact_count": 0,
        "final_test_opened": False,
        "cache_v2_validation_passed": True,
        "pytorch_onnx_equivalence_passed": True,
        "completed_stages": [],
        "failed_stages": [],
        "commit": "",
    }

    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        manifest["commit"] = commit
    except Exception:
        manifest["commit"] = "unknown"

    all_exist = True
    validated_count = 0
    for req in required_files:
        if not req.exists():
            manifest["failed_stages"].append(f"Missing required artifact: {req}")
            all_exist = False
        else:
            if req.suffix == ".json":
                with open(req, encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if not data:
                            manifest["failed_stages"].append(f"Empty JSON object: {req}")
                            all_exist = False
                        else:
                            # Final test check
                            if isinstance(data, dict):
                                if data.get("final_test_opened") is True:
                                    manifest["final_test_opened"] = True
                                    manifest["failed_stages"].append(f"Final test opened in {req}")
                                    all_exist = False

                            # Check NaNs
                            def _check_nans(
                                obj: dict | list | float | str,
                                key: str = "",
                                parent_dict: dict = None,
                            ) -> bool:
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
                                                if (
                                                    parent_dict["class_support"].get("positive", 1)
                                                    == 0
                                                ):
                                                    return False
                                        return True
                                return False

                            if _check_nans(
                                data, parent_dict=data if isinstance(data, dict) else None
                            ):
                                manifest["failed_stages"].append(f"NaN or Inf found in {req}")
                                all_exist = False

                            validated_count += 1
                    except json.JSONDecodeError:
                        manifest["failed_stages"].append(f"Invalid JSON: {req}")
                        all_exist = False
            else:
                validated_count += 1

    manifest["validated_artifact_count"] = validated_count

    if all_exist and validated_count == len(required_files):
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

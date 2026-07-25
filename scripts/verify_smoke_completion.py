import argparse
import json
import math
from pathlib import Path

import jsonschema


def _load_schema(name: str) -> dict:
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def _check_nans(obj, key: str = "", parent_dict: dict = None) -> bool:
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
        smoke_dir / "evttc" / "cache_validation.json",
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
        smoke_dir / "eap" / "matrix" / "finetune" / "jepa" / "fraction-1" / "seed-7" / "summary.json",
        smoke_dir / "eap" / "matrix" / "finetune" / "scratch" / "fraction-1" / "seed-7" / "summary.json",
        smoke_dir / "eap" / "matrix" / "finetune" / "jepa" / "fraction-0.1" / "seed-7" / "summary.json",
        smoke_dir / "eap" / "matrix" / "finetune" / "scratch" / "fraction-0.1" / "seed-7" / "summary.json",
        smoke_dir / "eap" / "matrix" / "finetune" / "jepa" / "fraction-0.05" / "seed-7" / "summary.json",
        smoke_dir / "eap" / "matrix" / "finetune" / "scratch" / "fraction-0.05" / "seed-7" / "summary.json",
        smoke_dir / "eap" / "matrix" / "matrix_summary.json",
        smoke_dir / "eap" / "matrix" / "eap_split_statistics.json",
        smoke_dir / "onnx" / "model.onnx",
        smoke_dir / "onnx" / "model_manifest.json",
        smoke_dir / "onnx" / "equivalence.json",
        smoke_dir / "onnx" / "benchmark.json",
        smoke_dir / "onnx_selection.json",
        smoke_dir / "phase_1_evttc.json",
        smoke_dir / "phase_2_eap.json",
        smoke_dir / "phase_4_onnx.json",
        smoke_dir / "phase_eap_cache.json",
        smoke_dir / "phase_eap_matrix_inner.json",
    ]

    manifest = {
        "status": "failed",
        "exit_code": 1,
        "all_required_artifacts_exist": False,
        "cache_v2_validation_passed": False,
        "pytorch_onnx_equivalence_passed": False,
        "final_test_opened": False,
        "required_artifact_count": len(required_files),
        "validated_artifact_count": 0,
        "completed_stages": [],
        "failed_stages": [],
        "commit": "unknown",
        "evidence_type": "real_smoke",
        "schema_version": "3.0",
    }

    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        manifest["commit"] = commit
        manifest["code_commit"] = commit
    except Exception:
        pass

    all_exist = True
    validated_count = 0

    # Store cross-artifact provenance
    provenance = {}

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
                    
                    artifact_type = data.get("artifact_type")
                    if artifact_type:
                        try:
                            schema = _load_schema(f"{artifact_type}.schema.json")
                            jsonschema.validate(instance=data, schema=schema)
                        except Exception as e:
                            manifest["failed_stages"].append(
                                f"Schema validation failed for {req} against {artifact_type}.schema.json: {e}"
                            )
                            all_exist = False
                            continue

                    # Summary validations (Training Run)
                    if artifact_type in ("supervised_run_v3", "jepa_pretrain_run_v3", "training_run_v3", "matrix_summary_v3"):
                        valid_splits = data.get("evaluation_splits") or data.get("validation_splits") or []
                        valid_split = data.get("evaluation_split")
                        has_validation_samples = data.get("validation_samples", 0) > 0
                        if (
                            "validation" not in valid_splits
                            and valid_split != "validation"
                            and not has_validation_samples
                        ):
                            manifest["failed_stages"].append(
                                f"Missing/invalid evaluation/validation splits in {req}"
                            )
                            all_exist = False
                            continue
                        if data.get("final_test_opened") is not False:
                            manifest["final_test_opened"] = data.get("final_test_opened")
                            manifest["failed_stages"].append(
                                f"final_test_opened is not explicitly False in {req}"
                            )
                            all_exist = False
                            continue
                        
                        run_type = "scratch" if "scratch" in req.parent.name else "jepa" if "jepa" in req.parent.name or "ssl" in req.parent.name else "unknown"
                        if run_type != "unknown":
                            resolved = data.get("run_fingerprint_payload", {}).get("resolved_model_config", {})
                            arch_fingerprint = {k: v for k, v in resolved.items() if k not in ("pretrained_encoder",)}
                            train_hash = data.get("split_manifest_sha256")
                            val_hash = data.get("subset_manifest_sha256")
                            
                            key = f"{run_type}_{req.parent.name}"
                            if "low_label_010" in req.parent.name:
                                key = f"{run_type}_low_label_010"
                            if "low_label_05" in req.parent.name:
                                key = f"{run_type}_low_label_05"

                            provenance[key] = {
                                "arch": arch_fingerprint,
                                "train_hash": train_hash,
                                "val_hash": val_hash
                            }

                    # General NaN checking
                    if _check_nans(data, parent_dict=data if isinstance(data, dict) else None):
                        manifest["failed_stages"].append(f"NaN or Inf found in {req}")
                        all_exist = False
                        continue

                    if req.name == "cache_validation.json":
                        if (
                            data.get("status") == "passed"
                            and data.get("cache_format_version") == 2
                            and data.get("normalize") is True
                            and data.get("normalization") == "non_centered_occupied_p95_scale"
                            and data.get("sidecar_sha256_matches") is True
                            and "checks" in data
                            and data.get("nonempty_samples_collapsed_to_zero") == 0
                            and data.get("sample_count_total", float("inf")) <= 200
                        ):
                            manifest["cache_v2_validation_passed"] = True
                        else:
                            manifest["failed_stages"].append(
                                f"Invalid cache validation payload: {data}"
                            )
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

                        provenance["model_manifest_checkpoint_hash"] = data.get("checkpoint_sha256")
                        provenance["model_manifest_protocol_hash"] = data.get("protocol_sha256")
                        provenance["model_manifest_git_commit"] = data.get("code_commit")
                        provenance["model_manifest_cache_sha256"] = data.get("cache_sha256")

                    # ONNX Selection JSON validation
                    if req.name == "onnx_selection.json":
                        if "checkpoint_sha256" not in data:
                            manifest["failed_stages"].append(
                                "Missing checkpoint_sha256 in onnx_selection"
                            )
                            all_exist = False
                            continue
                        provenance["selection_checkpoint_hash"] = data.get("checkpoint_sha256")
                        provenance["selection_protocol_hash"] = data.get("protocol_sha256")
                        provenance["selection_git_commit"] = data.get("code_commit")
                        provenance["selection_cache_sha256"] = data.get("cache_sha256")

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
        and manifest["cache_v2_validation_passed"]
        and manifest["pytorch_onnx_equivalence_passed"]
    ):
        # Cross-artifact provenance check
        if provenance.get("model_manifest_checkpoint_hash") != provenance.get("selection_checkpoint_hash"):
            manifest["failed_stages"].append(
                f"Cross-artifact provenance mismatch: Selection ({provenance.get('selection_checkpoint_hash')}) vs ONNX Export ({provenance.get('model_manifest_checkpoint_hash')})"
            )
        elif (
            provenance.get("model_manifest_protocol_hash") 
            and provenance.get("selection_protocol_hash")
            and provenance.get("model_manifest_protocol_hash") != provenance.get("selection_protocol_hash")
        ):
            manifest["failed_stages"].append("Protocol hash mismatch between Selection and ONNX manifest")
        elif (
            provenance.get("model_manifest_git_commit") 
            and provenance.get("selection_git_commit")
            and provenance.get("model_manifest_git_commit") != provenance.get("selection_git_commit")
        ):
            manifest["failed_stages"].append("Git commit mismatch between Selection and ONNX manifest")
        elif (
            provenance.get("model_manifest_cache_sha256")
            and provenance.get("selection_cache_sha256")
            and provenance.get("model_manifest_cache_sha256") != provenance.get("selection_cache_sha256")
        ):
            manifest["failed_stages"].append("Cache mismatch between Selection and ONNX manifest")
        else:
            # Check parity if we collected them
            parity_ok = True
            if "jepa_jepa_navigation_enabled" in provenance and "scratch_scratch_navigation_enabled" in provenance:
                j = provenance["jepa_jepa_navigation_enabled"]
                s = provenance["scratch_scratch_navigation_enabled"]
                if j["arch"] != s["arch"]:
                    manifest["failed_stages"].append("Architecture parity mismatch between JEPA and Scratch")
                    parity_ok = False
                
            if parity_ok:
                manifest["all_required_artifacts_exist"] = True
                manifest["status"] = "passed"
                manifest["exit_code"] = 0
                manifest["completed_stages"] = [str(p) for p in required_files]
                manifest["failed_stages"] = []

    manifest["artifact_type"] = "completion_manifest_v3"
    manifest["smoke_completed"] = (manifest["status"] == "passed")
    manifest["full_completed"] = False
    manifest["failures"] = manifest["failed_stages"]
    
    # Adding extra fields requested by instructions for completion_manifest
    manifest["protocol_version"] = "3.0"
    manifest["protocol_sha256"] = provenance.get("model_manifest_protocol_hash", "unknown")
    manifest["created_at"] = "2026-07-25"
    manifest["artifact_sha256"] = "pending"

    try:
        jsonschema.validate(instance=manifest, schema=_load_schema("completion_manifest_v3.schema.json"))
    except Exception as e:
        manifest["status"] = "failed"
        manifest["exit_code"] = 1
        manifest["failed_stages"].append(f"Internal error: Verification manifest failed schema validation: {e}")

    with open(smoke_dir / "completion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if manifest["status"] == "failed":
        print(f"Smoke completion gate failed. Missing/invalid: {manifest['failed_stages']}")
        exit(1)
    else:
        print("Smoke completion gate passed: all critical artifacts verified.")

if __name__ == "__main__":
    main()

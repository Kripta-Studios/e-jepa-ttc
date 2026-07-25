import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jsonschema

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.artifacts.protocol import load_frozen_protocol

SCHEMA_FILE_BY_ARTIFACT_TYPE = {
    "audit_cache": "cache_audit_v3.schema.json",
}


def _load_schema(name: str) -> dict:
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_nonnegative(data: dict, keys: tuple[str, ...]) -> bool:
    return all(
        isinstance(data.get(key), (int, float))
        and math.isfinite(float(data[key]))
        and float(data[key]) >= 0.0
        for key in keys
    )


def _check_nans(
    obj: object,
    key: str = "",
    parent_dict: dict[str, Any] | None = None,
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

    try:
        frozen_protocol = load_frozen_protocol()
    except Exception as e:
        print(f"Failed to load frozen protocol: {e}")
        exit(1)

    expected_commit = frozen_protocol["code_commit"]
    expected_protocol_hash = frozen_protocol["protocol_sha256"]
    evidence_type = "real_smoke"

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
        smoke_dir / "onnx_selection.json",
        smoke_dir / "phase_1_evttc.json",
        smoke_dir / "phase_2_eap.json",
        smoke_dir / "phase_4_onnx.json",
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
        "code_commit": expected_commit,
        "evidence_type": evidence_type,
        "schema_version": "3.0",
        "protocol_version": frozen_protocol["protocol_version"],
        "protocol_sha256": expected_protocol_hash,
    }

    all_exist = True
    validated_count = 0
    provenance = {}

    for req in required_files:
        if not req.exists():
            manifest["failed_stages"].append(f"Missing required artifact: {req}")
            all_exist = False
            continue

        if req.suffix == ".json":
            with open(req, encoding="utf-8-sig") as f:
                try:
                    data = json.load(f)
                    if not data:
                        manifest["failed_stages"].append(f"Empty JSON object: {req}")
                        all_exist = False
                        continue

                    artifact_type = data.get("artifact_type")
                    if not isinstance(artifact_type, str) or not artifact_type:
                        manifest["failed_stages"].append(
                            f"Missing artifact_type in required JSON: {req}"
                        )
                        all_exist = False
                        continue
                    if artifact_type:
                        try:
                            schema_file = SCHEMA_FILE_BY_ARTIFACT_TYPE.get(
                                artifact_type, f"{artifact_type}.schema.json"
                            )
                            schema = _load_schema(schema_file)
                            jsonschema.validate(instance=data, schema=schema)
                        except Exception as e:
                            manifest["failed_stages"].append(
                                f"Schema validation failed for {req}: {e}"
                            )
                            all_exist = False
                            continue

                        # Verify strict provenance fields on all artifacts
                        if data.get("code_commit") != expected_commit:
                            manifest["failed_stages"].append(
                                f"code_commit mismatch in {req}. Expected {expected_commit}, "
                                f"got {data.get('code_commit')}"
                            )
                            all_exist = False
                            continue

                        if data.get("protocol_sha256") != expected_protocol_hash:
                            manifest["failed_stages"].append(f"protocol_sha256 mismatch in {req}")
                            all_exist = False
                            continue

                        if not verify_artifact_hash(data):
                            manifest["failed_stages"].append(f"Self-hash invalid in {req}")
                            all_exist = False
                            continue

                    # Summary validations (Training Run)
                    if artifact_type in (
                        "supervised_run_v3",
                        "jepa_pretrain_run_v3",
                        "training_run_v3",
                        "matrix_summary_v3",
                    ):
                        if data.get("final_test_opened") is not False:
                            manifest["failed_stages"].append(
                                f"final_test_opened is not False in {req}"
                            )
                            all_exist = False
                            continue

                        run_type = (
                            "scratch"
                            if "scratch" in req.parent.name
                            else "jepa"
                            if "jepa" in req.parent.name or "ssl" in req.parent.name
                            else "unknown"
                        )
                        if run_type != "unknown":
                            resolved = data.get("run_fingerprint_payload", {}).get(
                                "resolved_model_config", {}
                            )
                            arch_fingerprint = {
                                k: v
                                for k, v in resolved.items()
                                if k not in ("pretrained_encoder",)
                            }
                            train_hash = data.get("split_manifest_sha256", data.get("cache_sha256"))
                            val_hash = data.get("subset_manifest_sha256", data.get("cache_sha256"))

                            key = f"{run_type}_{req.parent.name}"
                            if "low_label_010" in req.parent.name:
                                key = f"{run_type}_low_label_010"
                            if "low_label_05" in req.parent.name:
                                key = f"{run_type}_low_label_05"

                            provenance[key] = {
                                "arch": arch_fingerprint,
                                "train_hash": train_hash,
                                "val_hash": val_hash,
                                "params": data.get("parameter_count"),
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
                        ):
                            manifest["cache_v2_validation_passed"] = True

                    # ONNX Validation
                    if req.name == "model_manifest.json":
                        if (
                            data.get("selection_split") != "validation"
                            or data.get("diagnostic_split_consulted") is not False
                            or data.get("final_test_opened") is not False
                            or data.get("strict_state_dict_loading") is not True
                            or data.get("output_names") != ["log_ttc"]
                        ):
                            manifest["failed_stages"].append(
                                "ONNX model manifest violates selection/export policy"
                            )
                            all_exist = False
                            continue
                        onnx_path = smoke_dir / "onnx" / "model.onnx"
                        if data.get("onnx_sha256") != _sha256(onnx_path):
                            manifest["failed_stages"].append(
                                "ONNX model hash does not match model_manifest.json"
                            )
                            all_exist = False
                            continue
                        provenance["model_manifest"] = data

                    if req.name == "onnx_selection.json":
                        if (
                            data.get("selection_split") != "validation"
                            or data.get("final_test_opened") is not False
                        ):
                            manifest["failed_stages"].append(
                                "ONNX selection did not remain validation-only"
                            )
                            all_exist = False
                            continue
                        provenance["onnx_selection"] = data

                    if req.name == "equivalence.json":
                        if (
                            data.get("status") == "passed"
                            and data.get("real_validation_samples") is True
                            and "sample_id_hash" in data
                        ):
                            manifest["pytorch_onnx_equivalence_passed"] = True
                            provenance["equiv_sample_id_hash"] = data.get("sample_id_hash")
                        else:
                            manifest["failed_stages"].append("ONNX equivalence failed strict rules")
                            all_exist = False
                            continue

                    if req.name == "benchmark.json":
                        benchmark_keys = (
                            "mean_ms",
                            "p50_ms",
                            "p95_ms",
                            "p99_ms",
                        )
                        if (
                            not _finite_nonnegative(data, benchmark_keys)
                            or not isinstance(data.get("iterations"), int)
                            or data["iterations"] <= 0
                            or not isinstance(data.get("batch_size"), int)
                            or data["batch_size"] <= 0
                            or not (data["p50_ms"] <= data["p95_ms"] <= data["p99_ms"])
                        ):
                            manifest["failed_stages"].append(
                                "ONNX benchmark contains invalid latency/support fields"
                            )
                            all_exist = False
                            continue

                    if req.name.startswith("phase_") and data.get("status") != "passed":
                        manifest["failed_stages"].append(f"Stage record did not pass: {req}")
                        all_exist = False
                        continue

                    validated_count += 1
                except json.JSONDecodeError:
                    manifest["failed_stages"].append(f"Invalid JSON: {req}")
                    all_exist = False
        elif req.suffix == ".onnx":
            try:
                import onnx

                onnx_model = onnx.load(req)
                onnx.checker.check_model(onnx_model)
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
        # Strict ONNX matching
        sel = provenance.get("onnx_selection", {})
        man = provenance.get("model_manifest", {})
        if sel and man:
            if sel.get("checkpoint_sha256") != man.get("checkpoint_sha256"):
                manifest["failed_stages"].append(
                    "ONNX candidate and manifest mismatch on checkpoint"
                )
            if sel.get("protocol_sha256") != man.get("protocol_sha256"):
                manifest["failed_stages"].append("ONNX protocol hash mismatch")

        # Scratch vs JEPA parity
        if (
            "jepa_jepa_navigation_enabled" in provenance
            and "scratch_scratch_navigation_enabled" in provenance
        ):
            j = provenance["jepa_jepa_navigation_enabled"]
            s = provenance["scratch_scratch_navigation_enabled"]
            if j["arch"] != s["arch"]:
                manifest["failed_stages"].append(
                    "Architecture parity mismatch between JEPA and Scratch"
                )
            if j["train_hash"] != s["train_hash"] or j["val_hash"] != s["val_hash"]:
                manifest["failed_stages"].append(
                    "Dataset subset parity mismatch between JEPA and Scratch"
                )

        # Low label
        if "jepa_low_label_05" in provenance and "scratch_low_label_05" in provenance:
            j = provenance["jepa_low_label_05"]
            s = provenance["scratch_low_label_05"]
            if j["train_hash"] != s["train_hash"]:
                manifest["failed_stages"].append(
                    "Low label subset mismatch between JEPA and Scratch"
                )

        if not manifest["failed_stages"]:
            manifest["all_required_artifacts_exist"] = True
            manifest["status"] = "passed"
            manifest["exit_code"] = 0
            manifest["completed_stages"] = [str(p) for p in required_files]
            manifest["failed_stages"] = []

    manifest["artifact_type"] = "completion_manifest_v3"
    manifest["smoke_completed"] = manifest["status"] == "passed"
    manifest["full_completed"] = False
    manifest["failures"] = manifest["failed_stages"]

    manifest["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    manifest = sign_artifact(manifest)
    jsonschema.validate(
        instance=manifest,
        schema=_load_schema("completion_manifest_v3.schema.json"),
    )

    with open(smoke_dir / "completion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    if manifest["status"] == "failed":
        print(
            "Smoke completion gate failed. Missing/invalid: "
            f"{json.dumps(manifest['failed_stages'], indent=2)}"
        )
        exit(1)
    else:
        print("Smoke completion gate passed: all critical artifacts verified.")


if __name__ == "__main__":
    main()

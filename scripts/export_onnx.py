import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from e_jepa_ttc.artifacts.hashing import hash_dict, sign_artifact, verify_artifact_hash
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import EventTubeletTransformerRegressor


def _git_commit() -> str:
    """Return the exact source revision used for an export."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def hash_file(p: Path) -> str:
    if not p or not Path(p).exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as bf:
        for chunk in iter(lambda: bf.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_numpy_outputs(output: object) -> list[np.ndarray]:
    if isinstance(output, torch.Tensor):
        return [output.detach().cpu().numpy()]
    if isinstance(output, (tuple, list)):
        arrays: list[np.ndarray] = []
        for value in output:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unsupported PyTorch output type: {type(value)!r}")
            arrays.append(value.detach().cpu().numpy())
        return arrays
    raise TypeError(f"Unsupported PyTorch output type: {type(output)!r}")


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def export_to_onnx(
    checkpoint_path: Path,
    output_onnx: Path,
    model_type: str,
    record: dict,
    validation_split: str = "validation",
    sample_count: int = 32,
) -> None:
    print(f"Exporting {model_type} from {checkpoint_path} to {output_onnx}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("resolved_model_config", {})
    if not cfg:
        raise ValueError("Checkpoint lacks resolved_model_config")
    declared_model_config_hash = record.get("model_config_sha256")
    if declared_model_config_hash is not None and hash_dict(cfg) != declared_model_config_hash:
        raise ValueError("Checkpoint model config does not match the selection record.")

    in_channels = cfg.get("in_channels", checkpoint.get("in_channels", 21))

    if model_type == "tiny_cnn" or model_type == "tiny-cnn":
        width = cfg.get("width", 48)
        model = TinyCNNRegressor(in_channels=in_channels, width=width)
    elif model_type == "event-tubelet-transformer":
        embed_dim = cfg.get("embed_dim", checkpoint.get("embed_dim", 192))
        depth = cfg.get("depth", checkpoint.get("depth", 6))
        num_heads = cfg.get("num_heads", checkpoint.get("num_heads", 6))
        patch_size = cfg.get("patch_size", checkpoint.get("patch_size", 16))
        temporal_patch_size = cfg.get(
            "temporal_patch_size", checkpoint.get("temporal_patch_size", 1)
        )

        model = EventTubeletTransformerRegressor(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint lacks model_state_dict exactly")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cache_path = record.get("cache_path")
    if not cache_path or not Path(cache_path).exists():
        msg = (
            "Valid cache_path is strictly required to extract real validation samples "
            "for ONNX export"
        )
        raise ValueError(msg)

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    with np.load(cache_path, allow_pickle=False) as cache_data:
        validation_indices = np.flatnonzero(cache_data["split"].astype(str) == validation_split)
        if validation_indices.size == 0:
            raise ValueError("No validation samples found in cache")
        selected_indices = validation_indices[:sample_count]
        x = cache_data["x"][selected_indices]
        sample_ids = (
            cache_data["sample_id"][selected_indices]
            if "sample_id" in cache_data
            else selected_indices
        )

    validation_input = torch.from_numpy(x).float()
    export_input = validation_input

    sample_id_hash = hashlib.sha256(sample_ids.tobytes()).hexdigest()

    torch.onnx.export(
        model,
        export_input,
        output_onnx.as_posix(),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["log_ttc"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "log_ttc": {0: "batch_size"},
        },
    )
    print("ONNX export completed.")

    print("Validating ONNX model with onnxruntime...")
    onnx_model = onnx.load(output_onnx.as_posix())
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(output_onnx.as_posix(), providers=["CPUExecutionProvider"])
    ort_inputs = {"input": validation_input.numpy()}

    with torch.no_grad():
        pt_outputs = model(validation_input)
    ort_outputs = session.run(None, ort_inputs)

    print("Comparing PyTorch and ONNX outputs...")
    pt_arrays = _as_numpy_outputs(pt_outputs)
    if len(pt_arrays) != len(ort_outputs):
        raise ValueError(
            f"Output count mismatch: PyTorch={len(pt_arrays)}, ONNX={len(ort_outputs)}"
        )
    absolute_tolerance = 1e-5
    relative_tolerance = 1e-3
    absolute_differences: list[np.ndarray] = []
    relative_differences: list[np.ndarray] = []
    for pt_out, ort_out in zip(pt_arrays, ort_outputs, strict=True):
        np.testing.assert_allclose(
            pt_out,
            ort_out,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        absolute_difference = np.abs(pt_out - ort_out)
        absolute_differences.append(absolute_difference.reshape(-1))
        relative_differences.append(
            (absolute_difference / np.maximum(np.abs(pt_out), 1e-8)).reshape(-1)
        )
    all_absolute_differences = np.concatenate(absolute_differences)
    all_relative_differences = np.concatenate(relative_differences)
    max_diff = float(np.max(all_absolute_differences))
    max_rel_diff = float(np.max(all_relative_differences))
    mean_abs_error = float(np.mean(all_absolute_differences))

    print(f"PyTorch-ONNX comparison successful! Max diff: {max_diff}")

    actual_protocol_version, actual_protocol_hash = get_current_protocol_identity()
    record_protocol_hash = record.get("protocol_sha256", record.get("protocol_hash"))
    if record_protocol_hash not in (None, actual_protocol_hash):
        raise ValueError("Selection record protocol does not match the currently frozen protocol.")
    code_commit = _git_commit()
    selection_code_commit = record.get("git_commit", record.get("code_commit"))
    evidence_type = record.get("evidence_type", "diagnostic_export")
    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    with open(output_onnx.parent / "equivalence.json", "w", encoding="utf-8") as f:
        json.dump(
            sign_artifact(
                {
                    "artifact_type": "onnx_equivalence_v3",
                    "schema_version": "3.0",
                    "status": "passed",
                    "real_validation_samples": True,
                    "sample_count": int(validation_input.size(0)),
                    "sample_id_hash": sample_id_hash,
                    "maximum_absolute_error": max_diff,
                    "mean_absolute_error": mean_abs_error,
                    "maximum_relative_error": max_rel_diff,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "evidence_type": evidence_type,
                    "code_commit": code_commit,
                    "protocol_version": actual_protocol_version,
                    "protocol_sha256": actual_protocol_hash,
                    "created_at": created_at,
                }
            ),
            f,
            indent=2,
        )

    with open(output_onnx.parent / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            sign_artifact(
                _without_none(
                    {
                        "artifact_type": "onnx_manifest_v3",
                        "schema_version": "3.0",
                        "checkpoint_path": str(checkpoint_path.resolve().as_posix()),
                        "checkpoint_sha256": hash_file(checkpoint_path),
                        "onnx_path": str(output_onnx.resolve().as_posix()),
                        "onnx_sha256": hash_file(output_onnx),
                        "model_name": model_type,
                        "model_config_sha256": record.get("model_config_sha256"),
                        "cache_sha256": record.get("cache_sha256", hash_file(Path(cache_path))),
                        "navigation_mode": record.get("navigation_mode"),
                        "normalization_sha256": record.get("normalization_sha256"),
                        "in_channels": in_channels,
                        "resolved_model_config": cfg,
                        "selection_split": "validation",
                        "selection_metric": "validation_mae_s",
                        "selected_checkpoint_code_commit": selection_code_commit,
                        "diagnostic_split_consulted": False,
                        "strict_state_dict_loading": True,
                        "output_names": ["log_ttc"],
                        "code_commit": code_commit,
                        "final_test_opened": False,
                        "evidence_type": evidence_type,
                        "protocol_version": actual_protocol_version,
                        "protocol_sha256": actual_protocol_hash,
                        "created_at": created_at,
                    }
                )
            ),
            f,
            indent=2,
        )

    print("Benchmarking ONNX latency...")
    for _ in range(50):
        _ = session.run(None, ort_inputs)

    benchmark_iters = 500
    latencies = []
    for _ in range(benchmark_iters):
        start_time = time.perf_counter()
        _ = session.run(None, ort_inputs)
        latencies.append(time.perf_counter() - start_time)

    p50 = np.percentile(latencies, 50) * 1000
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    mean_lat = np.mean(latencies) * 1000

    print(
        f"ONNX Benchmark -> Mean: {mean_lat:.2f}ms, P50: {p50:.2f}ms, "
        f"P95: {p95:.2f}ms, P99: {p99:.2f}ms"
    )

    import psutil

    hardware_info = {
        "cpu_name": platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    if torch.cuda.is_available():
        hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        hardware_info["gpu_name"] = None

    with open(output_onnx.parent / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(
            sign_artifact(
                {
                    "artifact_type": "onnx_benchmark_v3",
                    "schema_version": "3.0",
                    "device": "CPU",
                    "batch_size": int(validation_input.size(0)),
                    "iterations": benchmark_iters,
                    "mean_ms": mean_lat,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "hardware": hardware_info,
                    "evidence_type": evidence_type,
                    "code_commit": code_commit,
                    "protocol_version": actual_protocol_version,
                    "protocol_sha256": actual_protocol_hash,
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                }
            ),
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--output", type=str, required=True, help="Path to save the ONNX model")
    parser.add_argument(
        "--selection-record", type=str, required=True, help="Path to the JSON selection record"
    )
    parser.add_argument(
        "--validation-split", type=str, default="validation", help="Name of the validation split"
    )
    parser.add_argument(
        "--sample-count", type=int, default=32, help="Number of real samples to use"
    )
    args = parser.parse_args()

    record_path = Path(args.selection_record)
    if not record_path.exists():
        sys.exit(f"Error: Selection record {record_path} does not exist.")

    with open(record_path, encoding="utf-8") as f:
        try:
            content = f.read().strip()
            if content.startswith("\ufeff"):
                content = content[1:]
            record = json.loads(content)
        except Exception as e:
            sys.exit(f"Error parsing selection record: {e}")

    # Explicit JSON schema check for the record
    try:
        import jsonschema

        schema_path = Path("schemas/onnx_candidate_v3.schema.json")
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as fs:
                schema = json.load(fs)
                jsonschema.validate(instance=record, schema=schema)
    except Exception as e:
        sys.exit(f"Error: Selection record schema validation failed: {e}")

    if not verify_artifact_hash(record):
        sys.exit("Error: Selection record artifact signature mismatch.")

    actual_protocol_version, actual_protocol_hash = get_current_protocol_identity()
    if (
        record.get("protocol_version") != actual_protocol_version
        or record.get("protocol_sha256") != actual_protocol_hash
    ):
        sys.exit("Error: Selection record does not match the frozen protocol.")

    checkpoint_path = Path(record["checkpoint_path"])
    if not checkpoint_path.exists():
        sys.exit(f"Error: Checkpoint file {checkpoint_path} does not exist.")

    actual_chk_hash = hash_file(checkpoint_path)
    if actual_chk_hash != record.get("checkpoint_sha256"):
        sys.exit("Error: Checkpoint hash mismatch.")

    cache_path = record.get("cache_path")
    if not cache_path or cache_path == "unknown" or not Path(cache_path).exists():
        sys.exit(
            "Error: Valid cache_path is strictly required to extract real validation "
            "samples for ONNX export"
        )

    actual_cache_hash = hash_file(Path(cache_path))
    if actual_cache_hash != record.get("cache_sha256"):
        sys.exit(
            f"Error: Cache hash mismatch. Expected {record.get('cache_sha256')} "
            f"got {actual_cache_hash}"
        )

    protocol_hash = record.get("protocol_sha256", record.get("protocol_hash"))
    if not protocol_hash or protocol_hash == "unknown":
        sys.exit("Error: Selection record missing protocol_hash.")

    code_commit = record.get("code_commit")
    if not code_commit or code_commit == "unknown":
        sys.exit("Error: Selection record missing code_commit.")

    metrics_path = checkpoint_path.parent / "metrics.json"
    model_type = None
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
            model_type = metrics.get("model_name")
            if metrics.get("final_test_opened") is True:
                sys.exit("Error: Selected checkpoint was exposed to final test.")

    if model_type is None:
        sys.exit(
            "Error: model_name not found in metrics, implicit fallback to tiny_cnn is prohibited."
        )

    if model_type == "tiny-cnn":
        model_type = "tiny_cnn"

    output_onnx = Path(args.output)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        checkpoint_path,
        output_onnx,
        model_type,
        record,
        args.validation_split,
        args.sample_count,
    )


if __name__ == "__main__":
    main()

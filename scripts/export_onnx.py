import argparse
import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import EventTubeletTransformerRegressor


def export_to_onnx(
    checkpoint_path: Path,
    output_onnx: Path,
    model_type: str,
    cache_path: str = None,
    validation_split: str = "validation",
    sample_count: int = 32,
) -> None:
    print(f"Exporting {model_type} from {checkpoint_path} to {output_onnx}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("resolved_model_config", {})
    if not cfg:
        raise ValueError("Checkpoint lacks resolved_model_config")
    
    in_channels = cfg.get("in_channels", checkpoint.get("in_channels", 21))

    if model_type == "tiny_cnn" or model_type == "tiny-cnn":
        width = cfg.get("width", 48)
        model = TinyCNNRegressor(in_channels=in_channels, width=width)
    elif model_type == "event-tubelet-transformer":
        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
        patch_size = cfg["patch_size"]
        temporal_patch_size = cfg.get("temporal_patch_size", 1)

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

    if not cache_path or not Path(cache_path).exists():
        raise ValueError("Valid cache_path is strictly required to extract real validation samples for ONNX export")
        
    cache_data = np.load(cache_path)
    mask = cache_data["split"] == validation_split
    x = cache_data["x"][mask]
    if len(x) > sample_count:
        x = x[:sample_count]
    elif len(x) == 0:
        raise ValueError("No validation samples found in cache")
    dummy_input = torch.from_numpy(x).float()

    # Export with full batch to avoid static shape tracing issues
    export_input = dummy_input

    torch.onnx.export(
        model,
        export_input,
        output_onnx.as_posix(),
        export_params=True,
        opset_version=17,
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
    ort_inputs = {"input": dummy_input.numpy()}

    with torch.no_grad():
        pt_outputs = model(dummy_input)
    ort_outputs = session.run(None, ort_inputs)

    print("Comparing PyTorch and ONNX outputs...")
    if isinstance(pt_outputs, torch.Tensor):
        np.testing.assert_allclose(pt_outputs.numpy(), ort_outputs[0], rtol=1e-3, atol=1e-5)
        max_diff = float(np.max(np.abs(pt_outputs.numpy() - ort_outputs[0])))
    else:
        max_diff = 0.0
        for pt_out, ort_out in zip(pt_outputs, ort_outputs, strict=False):
            np.testing.assert_allclose(pt_out.numpy(), ort_out, rtol=1e-3, atol=1e-5)
            diff = float(np.max(np.abs(pt_out.numpy() - ort_out)))
            if diff > max_diff:
                max_diff = diff
    print(f"PyTorch-ONNX comparison successful! Max diff: {max_diff}")

    mean_abs_error = float(np.mean(np.abs(pt_outputs.numpy() - ort_outputs[0])))

    with open(output_onnx.parent / "equivalence.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "passed",
                "max_absolute_difference": max_diff,
                "mean_absolute_difference": mean_abs_error,
                "rtol": 1e-3,
                "atol": 1e-5,
                "real_samples_used": True,
                "sample_count": int(dummy_input.size(0))
            },
            f,
            indent=2,
        )

    def file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as bf:
            for chunk in iter(lambda: bf.read(8192 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    with open(output_onnx.parent / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "checkpoint_source": str(checkpoint_path),
                "onnx_path": str(output_onnx),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "onnx_sha256": file_sha256(output_onnx),
            },
            f,
            indent=2,
        )

    print("Benchmarking ONNX latency...")
    # Warmup
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
        f"ONNX Benchmark -> Mean: {mean_lat:.2f}ms, "
        f"P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms"
    )

    with open(output_onnx.parent / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": "CPU",
                "batch_size": sample_count,
                "iterations": benchmark_iters,
                "mean_ms": mean_lat,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save the ONNX model")
    parser.add_argument(
        "--model-type",
        type=str,
        required=False,
        choices=["tiny_cnn", "event-tubelet-transformer"],
        help="Type of the model to export",
    )
    parser.add_argument("--cache", type=str, required=False, help="Path to the features cache NPZ")
    parser.add_argument(
        "--validation-split", type=str, default="validation", help="Name of the validation split"
    )
    parser.add_argument(
        "--sample-count", type=int, default=32, help="Number of real samples to use"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Error: Checkpoint file {checkpoint_path} does not exist.")

    model_type = args.model_type
    if model_type is None:
        metrics_path = checkpoint_path.parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
                if "model_name" in metrics:
                    model_type = metrics["model_name"]

        if model_type is None:
            raise ValueError(
                "--model-type is required if metrics.json is not found in the checkpoint directory"
            )

    if model_type == "tiny_cnn" or model_type == "tiny-cnn":
        model_type = "tiny_cnn"

    output_onnx = Path(args.output)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        checkpoint_path,
        output_onnx,
        model_type,
        args.cache,
        args.validation_split,
        args.sample_count,
    )


if __name__ == "__main__":
    main()

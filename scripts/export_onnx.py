import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor
from e_jepa_ttc.models.token_transformer import EventTubeletTransformerRegressor


def export_to_onnx(checkpoint_path: Path, output_onnx: Path, model_type: str) -> None:
    print(f"Exporting {model_type} from {checkpoint_path} to {output_onnx}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    in_channels = checkpoint.get("in_channels", 21)

    if model_type == "tiny_cnn" or model_type == "tiny-cnn":
        width = checkpoint.get("resolved_model_config", {}).get("width", 48)
        model = TinyCNNRegressor(in_channels=in_channels, width=width)
        dummy_input = torch.randn(1, in_channels, 90, 160)
    elif model_type == "event-tubelet-transformer":
        cfg = checkpoint.get("resolved_model_config", {})
        embed_dim = cfg.get("embed_dim", 192)
        depth = cfg.get("depth", 6)
        num_heads = cfg.get("num_heads", 6)
        patch_size = cfg.get("patch_size", 16)
        temporal_patch_size = cfg.get("temporal_patch_size", 1)

        model = EventTubeletTransformerRegressor(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        dummy_input = torch.randn(1, in_channels, 90, 160)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state", None))
    if state_dict is None:
        raise ValueError("Checkpoint lacks model_state_dict")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
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
    else:
        for pt_out, ort_out in zip(pt_outputs, ort_outputs, strict=False):
            np.testing.assert_allclose(pt_out.numpy(), ort_out, rtol=1e-3, atol=1e-5)
    print("PyTorch-ONNX comparison successful!")

    print("Benchmarking ONNX latency...")
    # Warmup
    for _ in range(50):
        _ = session.run(None, ort_inputs)

    benchmark_iters = 500
    start_time = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = session.run(None, ort_inputs)
    elapsed = (time.perf_counter() - start_time) / benchmark_iters
    print(f"ONNX Validation successful. Average latency on CPU: {elapsed * 1000:.2f} ms")


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

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} does not exist.")
        return

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

    # standardize model_type
    if model_type == "tiny_cnn" or model_type == "tiny-cnn":
        model_type = "tiny_cnn"

    output_onnx = Path(args.output)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(checkpoint_path, output_onnx, model_type)


if __name__ == "__main__":
    main()

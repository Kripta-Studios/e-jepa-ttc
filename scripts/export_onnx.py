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


def export_to_onnx(checkpoint_path: Path, output_onnx: Path, model_type: str):
    print(f"Exporting {model_type} from {checkpoint_path} to {output_onnx}")
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    in_channels = checkpoint.get("in_channels", 21)

    if model_type == "tiny_cnn" or model_type == "tiny-cnn":
        model = TinyCNNRegressor(in_channels=in_channels, num_risk_thresholds=4)
        dummy_input = torch.randn(1, in_channels, 90, 160)
    elif model_type == "event-tubelet-transformer":
        model = EventTubeletTransformerRegressor(in_channels=in_channels)
        dummy_input = torch.randn(1, in_channels, 90, 160)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["ttc_mean", "log_variance", "risk_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "ttc_mean": {0: "batch_size"},
            "log_variance": {0: "batch_size"},
            "risk_logits": {0: "batch_size"},
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
        for pt_out, ort_out in zip(pt_outputs, ort_outputs):
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


def main():
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

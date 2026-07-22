import argparse
import time
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torch import nn

from e_jepa_ttc.models import build_jepa, build_encoder
from e_jepa_ttc.models.tiny_cnn import TinyCNNTTCPredictor
from e_jepa_ttc.models.tubelet import EventTubeletTransformerTTCPredictor

def export_to_onnx(checkpoint_path: Path, output_onnx: Path, model_type: str):
    print(f"Exporting {model_type} from {checkpoint_path} to {output_onnx}")
    device = torch.device("cpu")
    
    if model_type == "tiny_cnn":
        model = TinyCNNTTCPredictor(
            input_channels=21,
            num_risk_thresholds=4
        )
        dummy_input = torch.randn(1, 21, 90, 160)
    elif model_type == "event-tubelet-transformer":
        model = EventTubeletTransformerTTCPredictor(
            input_channels=21,
            num_risk_thresholds=4
        )
        dummy_input = torch.randn(1, 21, 90, 160)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
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
        }
    )
    print("ONNX export completed.")

    print("Validating ONNX model with onnxruntime...")
    onnx_model = onnx.load(output_onnx.as_posix())
    onnx.checker.check_model(onnx_model)
    
    session = ort.InferenceSession(output_onnx.as_posix(), providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    
    start_time = time.perf_counter()
    for _ in range(10):
        _ = session.run(None, ort_inputs)
    elapsed = (time.perf_counter() - start_time) / 10.0
    print(f"ONNX Validation successful. Average latency on CPU: {elapsed * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=["tiny_cnn", "event-tubelet-transformer"])
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.model_type)

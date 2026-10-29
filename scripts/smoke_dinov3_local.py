"""Run a deterministic forward smoke for a local DINOv3 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.models.multimodal import DINOv3FeatureTeacher


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def _choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    args = _parse_args()
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("batch-size and steps must be positive.")
    model_path = args.model_path.expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError(f"--model-path must be a directory: {model_path}")
    weights_path = model_path / "model.safetensors"
    processor_path = model_path / "preprocessor_config.json"
    if not weights_path.is_file() or not processor_path.is_file():
        raise FileNotFoundError(
            "The local snapshot must contain model.safetensors and preprocessor_config.json."
        )

    device = _choose_device(args.device)
    teacher = DINOv3FeatureTeacher(str(model_path)).to(device)
    height, width = teacher.input_size
    input_tensor = torch.zeros(
        (args.batch_size, args.steps, 3, height, width),
        dtype=torch.uint8,
        device=device,
    )
    output = teacher(input_tensor)
    payload: dict[str, Any] = {
        "artifact_type": "dinov3_local_forward_smoke_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "snapshot_id": model_path.name,
        "model_weights_sha256": _sha256(weights_path),
        "processor_sha256": _sha256(processor_path),
        "device": str(device),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_is_finite": bool(torch.isfinite(output).all().item()),
        "output_requires_grad": bool(output.requires_grad),
        "uses_ttc_labels": False,
        "uses_boxes_for_sampling": False,
        "input_is_synthetic_zero_rgb": True,
        "status": "completed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a promptable, synthetic-image smoke for a local Hugging Face SAM snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image


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
    model_path = args.model_path.expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError(f"--model-path must be a directory: {model_path}")
    weights_path = model_path / "model.safetensors"
    processor_path = model_path / "preprocessor_config.json"
    if not weights_path.is_file() or not processor_path.is_file():
        raise FileNotFoundError(
            "The local SAM snapshot must contain model.safetensors and preprocessor_config.json."
        )

    try:
        from transformers import SamModel, SamProcessor  # type: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("Install the optional 'multimodal' dependency group.") from error

    device = _choose_device(args.device)
    processor = SamProcessor.from_pretrained(model_path, local_files_only=True)
    model = cast(Any, SamModel.from_pretrained(model_path, local_files_only=True)).to(device).eval()
    image = Image.new("RGB", (128, 128), (0, 0, 0))
    inputs = processor(
        images=image,
        input_points=[[[[64, 64]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**model_inputs)
    iou_scores = outputs.iou_scores
    pred_masks = outputs.pred_masks
    payload: dict[str, Any] = {
        "artifact_type": "sam_hf_promptable_forward_smoke_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "snapshot_id": model_path.name,
        "model_weights_sha256": _sha256(weights_path),
        "processor_sha256": _sha256(processor_path),
        "device": str(device),
        "input_image_shape": [128, 128, 3],
        "prompt_type": "single_synthetic_point",
        "iou_scores_shape": list(iou_scores.shape),
        "pred_masks_shape": list(pred_masks.shape),
        "outputs_are_finite": bool(
            torch.isfinite(iou_scores).all().item() and torch.isfinite(pred_masks).all().item()
        ),
        "uses_ttc_labels": False,
        "uses_gt_boxes": False,
        "automatic_bbox_free_proposals": False,
        "target_only": True,
        "status": "completed_promptable_smoke",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

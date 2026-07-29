"""ONNX export and ONNX Runtime parity for frozen OGE candidates."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC
from e_jepa_ttc.utils.io import write_structured


class _OGEExportWrapper(nn.Module):
    def __init__(self, model: ObjectGeometryJEPATTC) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        context_events: torch.Tensor,
        context_times_s: torch.Tensor,
        context_boxes: torch.Tensor,
        context_object_mask: torch.Tensor,
        context_ego_actions: torch.Tensor,
        context_ego_action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(
            context_events,
            context_times_s=context_times_s,
            context_boxes=context_boxes,
            context_object_mask=context_object_mask,
            context_ego_actions=context_ego_actions,
            context_ego_action_mask=context_ego_action_mask,
        )
        return (
            output.ttc_seconds,
            output.inverse_ttc_mean,
            output.inverse_ttc_log_variance,
            output.risk_logits,
        )


def export_oge_onnx(
    model: ObjectGeometryJEPATTC,
    example_inputs: dict[str, torch.Tensor],
    *,
    output_dir: str | Path,
    opset_version: int = 18,
) -> dict[str, Any]:
    """Export a fixed-shape, batch-one inference graph and verify numerical parity."""

    names = (
        "context_events",
        "context_times_s",
        "context_boxes",
        "context_object_mask",
        "context_ego_actions",
        "context_ego_action_mask",
    )
    missing = [name for name in names if name not in example_inputs]
    if missing:
        raise ValueError(f"Missing OGE export inputs: {missing}")
    floating = {
        "context_events",
        "context_times_s",
        "context_boxes",
        "context_ego_actions",
    }
    tensors = tuple(
        example_inputs[name]
        .detach()
        .cpu()
        .to(dtype=torch.float32 if name in floating else torch.bool)
        for name in names
    )
    if tensors[0].shape[0] != 1:
        raise ValueError("OGE deployment export requires example batch size one.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wrapper = _OGEExportWrapper(copy.deepcopy(model).cpu().eval()).eval()
    output_names = (
        "ttc_seconds",
        "inverse_ttc_mean",
        "inverse_ttc_log_variance",
        "risk_logits",
    )
    with torch.inference_mode():
        reference = wrapper(*tensors)
    onnx_path = output / "model.onnx"
    torch.onnx.export(
        wrapper,
        tensors,
        onnx_path,
        input_names=list(names),
        output_names=list(output_names),
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
    )
    import onnx
    import onnxruntime

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    session = onnxruntime.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {name: tensor.numpy() for name, tensor in zip(names, tensors, strict=True)}
    actual = session.run(list(output_names), ort_inputs)
    maximum_absolute_error: dict[str, float] = {}
    for name, expected, value in zip(output_names, reference, actual, strict=True):
        expected_array = expected.detach().cpu().numpy()
        error = float(np.max(np.abs(expected_array - value)))
        maximum_absolute_error[name] = error
        if not np.allclose(expected_array, value, rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"ONNX parity failed for {name}: max error {error}.")
    np.savez_compressed(output / "example_input.npz", **ort_inputs)
    (output / "example_output.json").write_text(
        json.dumps(
            {
                name: np.asarray(value).tolist()
                for name, value in zip(output_names, actual, strict=True)
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "format": "oge_jepa_ttc_onnx_v1",
        "opset_version": opset_version,
        "batch_size_contract": 1,
        "fixed_input_shapes": {
            name: list(tensor.shape) for name, tensor in zip(names, tensors, strict=True)
        },
        "model_config": asdict(model.config),
        "maximum_absolute_error": maximum_absolute_error,
        "verified_with_onnxruntime_cpu": True,
        "onnx_size_bytes": onnx_path.stat().st_size,
    }
    write_structured(output / "model_metadata.json", metadata)
    write_structured(
        output / "normalization.json",
        {
            "events": "occupied_voxel_noncentred_q95_magnitude",
            "boxes": "normalized_xyxy",
            "times": "seconds_relative_to_first_context_frame",
        },
    )
    return metadata


__all__ = ["export_oge_onnx"]

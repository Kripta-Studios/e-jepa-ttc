"""ONNX export and numerical verification for object-centric TTC inference."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA
from e_jepa_ttc.utils.io import write_structured


class _ObjectTTCExportWrapper(nn.Module):
    def __init__(self, model: ObjectCentricEventJEPA) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        context_events: torch.Tensor,
        context_boxes: torch.Tensor,
        context_object_mask: torch.Tensor,
        context_ego_actions: torch.Tensor,
        context_ego_action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model.predict_ttc(
            context_events,
            context_boxes,
            context_object_mask,
            context_ego_actions=context_ego_actions,
            context_ego_action_mask=context_ego_action_mask,
        )
        return (
            output.inverse_ttc_mean,
            output.inverse_ttc_log_variance,
            output.risk_logits,
            output.object_mask,
        )


def export_object_ttc_onnx(
    model: ObjectCentricEventJEPA,
    example_inputs: dict[str, torch.Tensor],
    *,
    output_dir: str | Path,
    opset_version: int = 18,
) -> dict[str, Any]:
    """Export batch-size-one inference and verify PyTorch/ONNX Runtime outputs."""

    required = (
        "context_events",
        "context_boxes",
        "context_object_mask",
        "context_ego_actions",
        "context_ego_action_mask",
    )
    missing = [name for name in required if name not in example_inputs]
    if missing:
        msg = f"Missing ONNX example inputs: {missing}."
        raise ValueError(msg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).cpu().eval()
    if not cpu_model.config.pre_cropped_events:
        msg = (
            "ONNX export requires pre_cropped_events=True because ONNX Runtime CPU "
            "does not provide the generic GridSample ROI path used by this project."
        )
        raise ValueError(msg)
    wrapper = _ObjectTTCExportWrapper(cpu_model).eval()
    floating_inputs = {
        "context_events",
        "context_boxes",
        "context_ego_actions",
    }
    tensors = tuple(
        example_inputs[name]
        .detach()
        .cpu()
        .to(dtype=torch.float32 if name in floating_inputs else torch.bool)
        for name in required
    )
    if tensors[0].shape[0] != 1:
        msg = "The deployment export contract requires an example batch size of one."
        raise ValueError(msg)
    output_names = (
        "inverse_ttc_mean",
        "inverse_ttc_log_variance",
        "risk_logits",
        "object_mask",
    )
    onnx_path = output / "model.onnx"
    with torch.inference_mode():
        reference = wrapper(*tensors)
    torch.onnx.export(
        wrapper,
        tensors,
        onnx_path,
        input_names=list(required),
        output_names=list(output_names),
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
        verbose=False,
    )

    import onnx
    import onnxruntime

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    session = onnxruntime.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(required, tensors, strict=True)
    }
    ort_outputs = session.run(list(output_names), ort_inputs)
    maximum_absolute_error: dict[str, float] = {}
    for name, expected, actual in zip(output_names, reference, ort_outputs, strict=True):
        expected_array = expected.detach().cpu().numpy()
        if expected_array.dtype == np.bool_:
            if not np.array_equal(expected_array, actual):
                msg = f"ONNX boolean output {name} differs from PyTorch."
                raise RuntimeError(msg)
            maximum_absolute_error[name] = 0.0
        else:
            error = float(np.max(np.abs(expected_array - actual)))
            maximum_absolute_error[name] = error
            if not np.allclose(expected_array, actual, rtol=1e-4, atol=1e-5):
                msg = f"ONNX output {name} exceeds numerical tolerance: {error}."
                raise RuntimeError(msg)

    np.savez_compressed(output / "example_input.npz", **ort_inputs)
    example_output = {
        name: np.asarray(value).tolist()
        for name, value in zip(output_names, ort_outputs, strict=True)
    }
    (output / "example_output.json").write_text(
        json.dumps(example_output, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "format": "object_event_jepa_ttc_onnx_v2",
        "exporter": "torch_export_dynamo",
        "opset_version": opset_version,
        "batch_axis_dynamic": False,
        "batch_size_contract": 1,
        "non_batch_axes_fixed": True,
        "model_config": asdict(cpu_model.config),
        "input_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(required, tensors, strict=True)
        },
        "output_shapes": {
            name: list(np.asarray(value).shape)
            for name, value in zip(output_names, ort_outputs, strict=True)
        },
        "maximum_absolute_error": maximum_absolute_error,
        "parameter_count_full_training_module": sum(
            parameter.numel() for parameter in cpu_model.parameters()
        ),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "verified_with_onnxruntime_cpu": True,
    }
    write_structured(output / "model_metadata.json", metadata)
    write_structured(
        output / "normalization.json",
        {
            "event_representation": "occupied_voxel_noncentred_q95_magnitude",
            "empty_voxels_remain_zero": True,
            "box_coordinates": "normalized_xyxy",
            "polarity_channels": "positive_bins_then_negative_bins",
        },
    )
    return metadata


__all__ = ["export_object_ttc_onnx"]

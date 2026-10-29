"""Portable geometry must match original arithmetic with learned gates and weights."""

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.joint_measurement.geometry_export import trace_portable
from operational.joint_measurement.geometry_model import GeometricPairMeasurement
from operational.range_transfer.common import read


def model_and_inputs():
    torch.manual_seed(19)
    model = GeometricPairMeasurement(CausalScaleTTC(CausalScaleTTCConfig())).eval()
    with torch.no_grad():
        model.gates[-1].weight.normal_(0, 0.02)
        model.gates[-1].bias.copy_(torch.tensor([0.3, -0.4]))
        model.spatial_weights.weight.normal_(0, 0.02)
        model.spatial_weights.bias.copy_(torch.tensor([0.1, 0.2, -0.1, -0.2]))
    inputs = [
        (torch.rand(1, 2, 12, 128, 128), torch.tensor([[0.05]])),
        (torch.rand(1, 2, 12, 128, 128), torch.tensor([[0.1]])),
        (torch.zeros(1, 2, 12, 128, 128), torch.tensor([[0.02]])),
    ]
    return model, inputs


def test_adapter_and_serialized_output_match_original_with_nonzero_gates(tmp_path):
    model, inputs = model_and_inputs()
    result = trace_portable(tmp_path, model, inputs, {"test_only": True})
    assert result["status"] == "COMPLETE"
    replay = read(tmp_path / "REPLAY.json")
    assert replay["eager_adapter_bit_exact"]
    assert replay["optimizer_updates"] == 0
    assert len(replay["checks"]) == 3
    assert trace_portable(tmp_path, model, inputs, {"test_only": True}) == result
    (tmp_path / "model.torchscript.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifact changed"):
        trace_portable(tmp_path, model, inputs, {"test_only": True})


def test_adapter_rejects_shapes_outside_fixed_export_contract(tmp_path):
    model, inputs = model_and_inputs()
    bad = [(inputs[0][0][:, :, :, :64], inputs[0][1]), inputs[1]]
    with pytest.raises(ValueError, match="native finite"):
        trace_portable(tmp_path, model, bad, {"test_only": True})

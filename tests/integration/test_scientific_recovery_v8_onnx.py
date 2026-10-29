"""Synthetic batch-one ONNX parity check for V8 dense delivery."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from e_jepa_ttc.evaluation.scientific_recovery_v8_delivery import export_v8_dense_onnx


class _DenseFixture(nn.Module):
    def forward(
        self, representations: torch.Tensor, delta_t_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = representations.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(-1) + delta_t_s.mean(
            dim=1, keepdim=True
        )
        return mean, torch.zeros_like(mean), torch.cat((mean, -mean), dim=1)


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="ONNX dependencies are optional outside the developer environment",
)
def test_v8_dense_onnx_is_atomic_and_cpu_verified(tmp_path) -> None:
    destination = tmp_path / "export"
    result = export_v8_dense_onnx(
        _DenseFixture(),
        {
            "representations": torch.ones(1, 2, 3, 4, 4),
            "delta_t_s": torch.full((1, 1), 0.1),
        },
        output_dir=destination,
        state_adapter_disclosure={"adapter": "none"},
        normalization={"representation": "synthetic"},
    )
    assert result["verified_with_onnxruntime_cpu"] is True
    assert (destination / "model.onnx").is_file()
    assert (destination / "normalization.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_v8_dense_onnx(
            _DenseFixture(),
            {
                "representations": torch.ones(1, 2, 3, 4, 4),
                "delta_t_s": torch.full((1, 1), 0.1),
            },
            output_dir=destination,
            state_adapter_disclosure={"adapter": "none"},
            normalization={"representation": "synthetic"},
        )

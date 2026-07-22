from __future__ import annotations

from pathlib import Path

import torch

from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig
from e_jepa_ttc.runtime.benchmark import benchmark_object_ttc_model
from e_jepa_ttc.runtime.export import export_object_ttc_onnx


def _model_and_inputs() -> tuple[ObjectCentricEventJEPA, dict[str, torch.Tensor]]:
    model = ObjectCentricEventJEPA(
        ObjectJEPAConfig(
            in_channels=4,
            embedding_dim=16,
            feature_dim=16,
            predictor_depth=1,
            predictor_heads=4,
            dropout=0.0,
            pre_cropped_events=True,
        )
    ).eval()
    inputs = {
        "context_events": torch.randn(1, 3, 4, 16, 16),
        "context_boxes": torch.tensor(
            [[[[0.2, 0.2, 0.8, 0.8]]] * 3],
            dtype=torch.float32,
        ),
        "context_object_mask": torch.ones(1, 3, 1, dtype=torch.bool),
        "context_sampling_boxes": torch.tensor(
            [[[[0.0, 0.0, 1.0, 1.0]]] * 3],
            dtype=torch.float32,
        ),
        "context_ego_actions": torch.zeros(1, 3, model.config.action_dim),
        "context_ego_action_mask": torch.zeros(1, 3, dtype=torch.bool),
    }
    return model, inputs


def test_object_ttc_onnx_export_is_numerically_verified(tmp_path: Path) -> None:
    model, inputs = _model_and_inputs()
    metadata = export_object_ttc_onnx(model, inputs, output_dir=tmp_path)

    assert metadata["verified_with_onnxruntime_cpu"] is True
    assert (tmp_path / "model.onnx").stat().st_size > 0
    assert (tmp_path / "example_input.npz").exists()


def test_object_ttc_latency_benchmark_reports_percentiles() -> None:
    model, inputs = _model_and_inputs()
    metrics = benchmark_object_ttc_model(
        model,
        inputs,
        device="cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )

    assert metrics["latency_ms_p50"] > 0
    assert metrics["windows_per_second"] > 0

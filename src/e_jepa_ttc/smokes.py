"""Bounded, explicitly synthetic smoke runners used by the CLI and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.evaluation.robustness import evaluate_robustness
from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig
from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor
from e_jepa_ttc.runtime.benchmark import benchmark_object_ttc_model
from e_jepa_ttc.runtime.export import export_object_ttc_onnx
from e_jepa_ttc.runtime.streaming import StreamingTTCEstimator


class _SyntheticEvents(Dataset[dict[str, object]]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        base = np.arange(64, dtype=np.int32)
        t_us = np.arange(64, dtype=np.int64) * 1_000
        events = EventBatch(
            x=(base + index) % 64,
            y=(base * 3 + index) % 64,
            t_us=t_us,
            polarity=np.where(base % 2, 1, -1).astype(np.int8),
            width=64,
            height=64,
            sequence_id=f"synthetic-{index}",
            t_start_us=0,
            t_end_us=int(t_us[-1] + 1_000),
        )
        return {"events": events, "ttc_seconds": 1.0 + 0.1 * index}


def run_robustness_smoke(*, output: Path, samples: int, seed: int) -> dict[str, Any]:
    """Run the deterministic synthetic robustness smoke and persist evidence."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyCNNRegressor(in_channels=10, width=16)
    result = evaluate_robustness(
        model=model,
        dataset=_SyntheticEvents(samples),
        device=torch.device("cpu"),
    )
    payload: dict[str, Any] = {
        "artifact_type": "synthetic_robustness_smoke_v1",
        "status": "completed",
        "seed": seed,
        "sample_count": samples,
        "model": "TinyCNNRegressor_random_initialization",
        "dataset": "synthetic_events_fixture",
        "metrics_are_not_real_dataset_results": True,
        **result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _runtime_model() -> ObjectCentricEventJEPA:
    return ObjectCentricEventJEPA(
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


def _runtime_inputs(model: ObjectCentricEventJEPA) -> dict[str, torch.Tensor]:
    return {
        "context_events": torch.randn(1, 3, 4, 16, 16),
        "context_boxes": torch.tensor([[[[0.2, 0.2, 0.8, 0.8]]] * 3]),
        "context_object_mask": torch.ones(1, 3, 1, dtype=torch.bool),
        "context_sampling_boxes": torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]] * 3]),
        "context_ego_actions": torch.zeros(1, 3, model.config.action_dim),
        "context_ego_action_mask": torch.zeros(1, 3, dtype=torch.bool),
    }


def _streaming_prediction() -> dict[str, object]:
    model = _runtime_model()
    estimator = StreamingTTCEstimator(
        model,
        width=8,
        height=6,
        event_bins=2,
        event_window_ms=100,
        history_steps=3,
        device="cpu",
    )
    timestamps = np.arange(0, 300_000, 1_000, dtype=np.int64)
    estimator.push_events(
        timestamps % 8,
        timestamps % 6,
        timestamps,
        np.where(np.arange(timestamps.size) % 2, 1, -1),
    )
    box = np.asarray([0.25, 0.25, 0.75, 0.75], dtype=np.float32)
    for endpoint in (100_000, 200_000, 300_000):
        estimator.push_observation(endpoint, box)
    prediction = estimator.predict(300_000)
    return {
        "timestamp_us": prediction.timestamp_us,
        "ttc_mean_s": prediction.ttc_mean_s,
        "ttc_std_s": prediction.ttc_std_s,
        "risk_probabilities": list(prediction.risk_probabilities),
        "risk_state": prediction.risk_state,
        "preprocessing_ms": prediction.preprocessing_ms,
        "inference_ms": prediction.inference_ms,
        "event_count": prediction.event_count,
        "event_rate_hz": prediction.event_rate_hz,
    }


def run_runtime_smoke(*, output_dir: Path, seed: int) -> dict[str, Any]:
    """Run the bounded synthetic ONNX/streaming smoke and persist evidence."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _runtime_model()
    inputs = _runtime_inputs(model)
    export_metadata = export_object_ttc_onnx(model, inputs, output_dir=output_dir)
    latency = benchmark_object_ttc_model(
        model,
        inputs,
        device="cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )
    payload: dict[str, Any] = {
        "artifact_type": "runtime_export_streaming_smoke_v1",
        "status": "completed",
        "seed": seed,
        "synthetic_fixture": True,
        "metrics_are_not_real_dataset_results": True,
        "export": export_metadata,
        "latency": latency,
        "streaming": _streaming_prediction(),
    }
    output = output_dir / "runtime_smoke_metrics.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = ["run_robustness_smoke", "run_runtime_smoke"]

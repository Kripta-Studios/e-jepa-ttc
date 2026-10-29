"""Robustness evaluation suite for E-JEPA-TTC."""

from __future__ import annotations

import logging
from collections.abc import Sequence, Sized
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.corruptions import (
    EventCorruptionSpec,
    corrupt_event_batch,
)
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid


def _event_and_target(sample: object) -> tuple[EventBatch, float | None]:
    """Extract an event sample and an optional scalar regression target."""

    if isinstance(sample, EventBatch):
        return sample, None
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        events = next((value for value in sample if isinstance(value, EventBatch)), None)
        if events is None:
            raise TypeError("Robustness samples must contain an EventBatch.")
        target = next(
            (value for value in sample if isinstance(value, (float, int, np.floating, np.integer))),
            None,
        )
        return events, None if target is None else float(target)
    if isinstance(sample, dict):
        events = next((value for value in sample.values() if isinstance(value, EventBatch)), None)
        if events is None:
            raise TypeError("Robustness mappings must contain an EventBatch.")
        target = sample.get("ttc_seconds", sample.get("target"))
        return events, float(target) if isinstance(target, (float, int)) else None
    raise TypeError("Robustness dataset items must be EventBatch, tuple or mapping.")


def _prediction_tensor(output: object) -> torch.Tensor:
    """Normalize common model output shapes to a one-dimensional tensor."""

    if isinstance(output, torch.Tensor):
        return output.reshape(-1)
    if isinstance(output, dict):
        for key in ("ttc_mean_seconds", "ttc_mean_s", "prediction", "pred"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value.reshape(-1)
    for key in ("ttc_mean_seconds", "ttc_mean_s", "prediction", "pred"):
        value = getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            return value.reshape(-1)
    raise TypeError("Model output does not expose a tensor TTC prediction.")


def _model_input_channels(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    channels = getattr(config, "in_channels", 10)
    if not isinstance(channels, int) or channels <= 0 or channels % 2:
        raise ValueError("Robustness model in_channels must be a positive even integer.")
    return channels


def evaluate_robustness(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    corruptions: list[EventCorruptionSpec] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic perturbations on raw events.

    Each corrupted sample is encoded independently, so the source cache stays
    unchanged and no future event is introduced by the runner.  Datasets may
    provide only events, or an event plus a scalar TTC target.
    """

    if corruptions is None:
        corruptions = [
            EventCorruptionSpec(kind="none", severity=0.0),
            EventCorruptionSpec(kind="event_dropout", severity=0.5),
            EventCorruptionSpec(kind="timestamp_jitter_us", severity=1000.0),
            EventCorruptionSpec(kind="background_event_rate", severity=0.1),
            EventCorruptionSpec(kind="dead_pixel_fraction", severity=0.1),
            EventCorruptionSpec(kind="temporal_window_fraction", severity=0.8),
        ]

    results: dict[str, Any] = {}
    bins = _model_input_channels(model) // 2
    was_training = model.training
    model.eval()
    try:
        dataset_size = len(cast(Sized, dataset))
        for spec in corruptions:
            logging.info(
                "Evaluating robustness against %s (severity: %s)",
                spec.kind,
                spec.severity,
            )
            predictions: list[float] = []
            targets: list[float] = []
            event_counts: list[int] = []
            finite_predictions = 0
            for index in range(dataset_size):
                events, target = _event_and_target(dataset[index])
                corrupted = corrupt_event_batch(events, spec, seed_offset=index)
                features = torch.from_numpy(
                    encode_voxel_grid(
                        corrupted,
                        bins=bins,
                        separate_polarity=True,
                        normalize=True,
                    )
                ).to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    prediction = _prediction_tensor(model(features[None]))
                value = float(prediction[0].detach().cpu())
                predictions.append(value)
                event_counts.append(corrupted.num_events)
                finite_predictions += int(np.isfinite(value))
                if target is not None and np.isfinite(target):
                    targets.append(target)
            summary: dict[str, Any] = {
                "status": "completed",
                "spec": {
                    "kind": spec.kind,
                    "severity": spec.severity,
                    "seed": spec.seed,
                },
                "sample_count": len(predictions),
                "finite_prediction_count": finite_predictions,
                "mean_event_count": float(np.mean(event_counts)) if event_counts else 0.0,
                "prediction_mean": float(np.mean(predictions)) if predictions else float("nan"),
                "prediction_std": float(np.std(predictions)) if predictions else float("nan"),
            }
            if targets and len(targets) == len(predictions):
                errors = np.asarray(predictions, dtype=np.float64) - np.asarray(
                    targets, dtype=np.float64
                )
                summary.update(
                    {
                        "target_count": len(targets),
                        "mae": float(np.mean(np.abs(errors))),
                        "rmse": float(np.sqrt(np.mean(errors**2))),
                    }
                )
            else:
                summary["target_count"] = len(targets)
                summary["metrics_status"] = "targets_unavailable"
            results[f"{spec.kind}_{spec.severity}"] = summary
    finally:
        model.train(was_training)

    return {"corruptions_tested": len(corruptions), "results": results}


__all__ = ["evaluate_robustness"]

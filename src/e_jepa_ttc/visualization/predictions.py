"""Plotting helpers for saved, rather than hand-entered, predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def save_ttc_curve(
    timestamps_s: Sequence[float],
    predictions_s: Sequence[float],
    targets_s: Sequence[float],
    output: str | Path,
) -> None:
    """Save a TTC curve; arrays must have equal lengths."""

    timestamps = np.asarray(timestamps_s)
    predictions = np.asarray(predictions_s)
    targets = np.asarray(targets_s)
    if not (timestamps.shape == predictions.shape == targets.shape):
        raise ValueError("timestamps, predictions and targets must have equal shapes.")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(timestamps, targets, label="target")
    axis.plot(timestamps, predictions, label="prediction")
    axis.set(xlabel="time (s)", ylabel="TTC (s)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


__all__ = ["save_ttc_curve"]

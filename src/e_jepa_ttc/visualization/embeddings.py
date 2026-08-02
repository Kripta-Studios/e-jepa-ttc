"""Low-dependency embedding diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Compute deterministic two-dimensional PCA coordinates by SVD."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("embeddings must have shape [N,D] with N >= 2.")
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    directions = components[:2]
    if directions.shape[0] == 1:
        directions = np.vstack([directions, np.zeros((1, values.shape[1]))])
    return (centered @ directions.T).astype(np.float32)


def save_embedding_plot(embeddings: np.ndarray, output: str | Path) -> None:
    """Save a PCA plot generated solely from the supplied embedding artifact."""

    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    coordinates = pca_2d(embeddings)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=8)
    axis.set(xlabel="PC1", ylabel="PC2")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


__all__ = ["pca_2d", "save_embedding_plot"]

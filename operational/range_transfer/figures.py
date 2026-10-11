"""Regenerate descriptive figures from persisted results without training."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def build(run: Path) -> None:
    """Keep synthetic learning and observed transfer diagnostics visibly separate."""
    output = run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    mass = pd.read_csv(run / "MASS_AUDIT.csv")
    mass = mass.loc[mass.group == "ALL"]
    figure, axis = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mass))
    for offset, column in enumerate(("native_mass", "M1_mass", "M2_mass")):
        axis.bar(x + (offset - 1) * 0.25, mass[column], width=0.25, label=column)
    axis.set(
        xticks=x,
        xticklabels=mass["range"],
        ylabel="Total objective mass",
        title="H metadata: weighting changes; uniform sampler unchanged",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "H_MASS_BY_RANGE.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4))
    for name in ("continuous10", "split5_restore5"):
        curve = run / "qa" / name / "CURVE.csv"
        if curve.exists():
            rows = pd.read_csv(curve)
            axis.plot(rows["update"], rows.loss, marker="o", label=name)
    axis.set(
        xlabel="Update",
        ylabel="Synthetic loss",
        title="Technical resume check only: 20 total optimizer updates",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "SYNTHETIC_RESUME_CURVES.png", dpi=150)
    plt.close(figure)
    traces = pd.read_csv(run / "v12_cached_attribution/HEAD_TRACES.csv")
    traces = traces.loc[traces.seed == 7]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.scatter(traces.truth_ttc, traces.residual_suppressed_ttc, label="Residual suppressed")
    axis.scatter(traces.truth_ttc, traces.prediction_ttc, label="Historical H8 seed7")
    low, high = float(traces.truth_ttc.min()), float(traces.truth_ttc.max())
    axis.plot([low, high], [low, high], color="black", linestyle="--", label="Identity")
    axis.set(
        xlabel="Observed TTC (s)",
        ylabel="Predicted TTC (s)",
        title="32 selected FCWD diagnostic cases; not a full-cohort score",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "V12_SELECTED_HEAD_ATTRIBUTION.png", dpi=150)
    plt.close(figure)
    scientific_curves = sorted((run / "fits").glob("*/CURVE.csv"))
    if scientific_curves:
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for path in scientific_curves:
            curve = pd.read_csv(path)
            axis = axes[1 if "-L1-" in path.parent.name else 0]
            axis.plot(
                curve["update"],
                curve.loss.rolling(100, min_periods=1).mean(),
                label=path.parent.name,
            )
        for loss, axis in enumerate(axes):
            axis.set(ylabel=f"L{loss} weighted objective (100-update mean)")
            axis.legend(fontsize=8)
        axes[0].set_title("Scientific H fits: fixed objectives, not validation scores")
        axes[1].set_xlabel("Scientific update")
        figure.tight_layout()
        figure.savefig(output / "SCIENTIFIC_TRAINING_CURVES.png", dpi=150)
        plt.close(figure)
    gradient_path = run / "GRADIENT_AUDIT.csv"
    if gradient_path.exists():
        gradients = pd.read_csv(gradient_path)
        matrix = gradients.pivot(index=["mass", "range"], columns="term", values="gradient_l2")
        figure, axis = plt.subplots(figsize=(10, 10))
        values = np.log10(np.maximum(matrix.to_numpy(), 1e-12))
        chart = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set(
            xticks=np.arange(len(matrix.columns)),
            xticklabels=matrix.columns,
            yticks=np.arange(len(matrix)),
            yticklabels=[f"{m} / {r}" for m, r in matrix.index],
            title="H initial gradients: log10 L2 norm (display floor 1e-12)",
        )
        figure.colorbar(chart, ax=axis)
        figure.tight_layout()
        figure.savefig(output / "H_GRADIENTS_BY_RANGE.png", dpi=150)
        plt.close(figure)

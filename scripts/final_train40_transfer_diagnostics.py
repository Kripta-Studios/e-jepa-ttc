"""Describe saved per-sequence errors and expose large errors without refitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OWN = "PHASE_ONLY_TRAIN40_S7"
GARL = "Garl_full"
RGB = "R_CTX_TRAIN40"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "docs/final_train40_transfer_20261027"
    source = directory / "PER_SEQUENCE.csv"
    frame = pd.read_csv(source)
    extended = frame.loc[frame.cohort == "extended_new"]
    table = extended.pivot(index="sequence", columns="model", values="RTE_pct")
    if table[[OWN, GARL, RGB]].isna().any().any():
        raise ValueError("Sequence diagnostic requires all three paired means")
    losses = table.loc[table[OWN] > table[GARL], [OWN, GARL, RGB]]
    result = {
        "scope": "Descriptive diagnostics of frozen, already-scored predictions",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_sequences": len(table),
        "own_wins_against_garl": int((table[OWN] < table[GARL]).sum()),
        "own_losses_against_garl": losses.to_dict(orient="index"),
        "median_sequence_mean_RTE_pct": {
            model: float(table[model].median()) for model in (OWN, GARL, RGB)
        },
        "largest_garl_sequence_errors": table.nlargest(4, GARL)[[OWN, GARL, RGB]].to_dict(
            orient="index"
        ),
        "interpretation": [
            "Pooled error differences are influenced by large errors; no outliers removed.",
            "Garl wins on all three lateral CCRs-side sequences in this sample.",
            "RGB context has lower pooled errors than final fusion; no checkpoint selected.",
            "Median sequence mean RTE is a descriptive diagnostic, not the primary metric.",
        ],
        "optimizer_updates": 0,
        "inference_reexecuted": False,
    }
    (directory / "DIAGNOSTICS.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if (
        not np.isfinite(table[[OWN, GARL]].to_numpy()).all()
        or (table[[OWN, GARL]] <= 0).any().any()
    ):
        raise ValueError("Logarithmic plot requires positive finite sequence errors")
    wins = table[OWN] < table[GARL]
    for mask, label, color in (
        (wins, "TRAIN40 lower error", "#167566"),
        (~wins, "Garl lower error", "#b34b3f"),
    ):
        part = table.loc[mask]
        ax.scatter(part[GARL], part[OWN], label=label, color=color, s=38)
    limits = [
        float(table[[OWN, GARL]].min().min()) * 0.7,
        float(table[[OWN, GARL]].max().max()) * 1.4,
    ]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlim=limits,
        ylim=limits,
        xlabel="Garl mean relative TTC error (%)",
        ylabel="Final TRAIN40 mean relative TTC error (%)",
        title=(
            "Expanded EvTTC sample: 32 physical sequences\n"
            "Equal errors on diagonal; no outliers removed"
        ),
    )
    for name in ("CPNAO-low", "CPLA-medium", "CCRs-1-low-100-overlap-100"):
        offset = {"CPNAO-low": (-8, 20), "CPLA-medium": (-8, -22)}.get(name, (-8, 8))
        ax.annotate(
            name,
            (table.loc[name, GARL], table.loc[name, OWN]),
            xytext=offset,
            textcoords="offset points",
            ha="right",
            fontsize=8,
        )
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=9)
    fig.savefig(directory / "PER_SEQUENCE_RTE_LOG.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"sequences": len(table), "wins": result["own_wins_against_garl"]}))


if __name__ == "__main__":
    main()

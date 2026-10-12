"""Export research figures directly from saved, cohort-labelled result tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import RUN


def main() -> None:
    folder = RUN / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    results = read(RUN / "RESULTS.json")
    results.update(read(RUN / "D_RESULTS.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("crucial", "negative"), strict=True):
        for index, arm in enumerate(("Z", "G", "D")):
            scores = [
                results[f"{arm}-S{seed}"]["bins"][metric]["group_macro_mid"] for seed in (7, 13, 23)
            ]
            axis.scatter(np.full(3, index), scores, s=24, color="#1565a7")
            axis.errorbar(
                index,
                np.mean(scores),
                yerr=np.std(scores, ddof=1),
                fmt="_",
                color="black",
                capsize=5,
            )
        axis.set_xticks(range(3), ["Native control Z", "Full tokens G", "Temporal D"])
        axis.set_ylabel(f"MiD {metric} (lower is better)")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Internal V · same 16,432 queries · dots: seeds 7/13/23; bar: mean ± SD")
    fig.tight_layout()
    fig.savefig(folder / "V_HEAD_COMPARISON.png")
    fig.savefig(folder / "V_HEAD_COMPARISON.pdf")
    plt.close(fig)
    clipping = pd.read_csv(RUN / "CLIPPING.csv")
    extra = pd.read_csv(RUN / "QUEUES_CLIPPING.csv")
    if "reused_control" in extra:
        extra = extra.loc[~extra.reused_control]
    clipping = pd.concat([clipping, extra], ignore_index=True)
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(clipping.fit, 100 * clipping.clipped_fraction, color="#1565a7")
    axis.set_ylabel("Updates with pre-clip norm > 1 (%)")
    axis.tick_params(axis="x", rotation=60)
    axis.set_title("Clipping by frozen fit · objectives differ across queues")
    fig.tight_layout()
    fig.savefig(folder / "CLIPPING_ALL_FITS.png")
    plt.close(fig)
    fits = [
        path
        for path in sorted((RUN / "fits").iterdir())
        if (path / "CHECKPOINT_RECEIPT.json").exists()
        and read(path / "CHECKPOINT_RECEIPT.json")["status"] == "COMPLETE"
    ]
    fig, axes = plt.subplots(
        (len(fits) + 3) // 4, 4, figsize=(14, 3 * ((len(fits) + 3) // 4)), squeeze=False
    )
    for axis, path in zip(axes.ravel(), fits, strict=False):
        curve = pd.read_csv(path / "CURVE.csv")
        axis.plot(curve["update"], curve.loss.rolling(50, min_periods=1).mean(), lw=1)
        axis.set_title(path.name)
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Own objective, mean of 50")
    for axis in axes.ravel()[len(fits) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Training curves · separate y scales; absolute losses are not comparable across objectives"
    )
    fig.tight_layout()
    fig.savefig(folder / "TRAINING_CURVES.png")
    plt.close(fig)
    write(
        folder / "MANIFEST.json",
        {
            path.name: sha(path)
            for path in sorted(folder.iterdir())
            if path.suffix in (".png", ".pdf")
        },
    )


if __name__ == "__main__":
    main()

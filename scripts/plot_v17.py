"""Export research figures directly from saved, cohort-labelled result tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
    fits = [
        path
        for path in sorted((RUN / "fits").iterdir())
        if (path / "CHECKPOINT_RECEIPT.json").exists()
    ]
    curves, clip_rows = {}, []
    for path in fits:
        receipt = read(path / "CHECKPOINT_RECEIPT.json")
        if sha(path / "checkpoint_last.pt") != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during plot generation")
        payload = torch.load(path / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        cursor = payload["cursor"]
        curve = pd.DataFrame(cursor.get("curve", cursor.get("curves")))
        if len(curve) != receipt["completed_updates"]:
            raise ValueError("Curve differs from durable update count")
        curves[path.name] = curve
        curve.to_csv(path / "CURVE_DURABLE.csv", index=False)
        enabled = curve.clipping_enabled if "clipping_enabled" in curve else True
        clip_rows.append(
            {
                "fit": path.name,
                "status": receipt["status"],
                "updates": len(curve),
                "clipped_fraction": float(((curve.gradient_l2 > 1) & enabled).mean()),
                "norm_above1_fraction": float((curve.gradient_l2 > 1).mean()),
                "label": path.name + ("*" if receipt["status"] != "COMPLETE" else ""),
            }
        )
    clipping = pd.DataFrame(clip_rows)
    clipping.to_csv(RUN / "ALL_FIT_CLIPPING.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(clipping.label, 100 * clipping.clipped_fraction, color="#1565a7")
    axis.set_ylabel("Updates with gradient clipping applied (%)")
    axis.tick_params(axis="x", rotation=60)
    axis.set_title("Actual clipping · N0 disables clipping; * incomplete fit; objectives differ")
    fig.tight_layout()
    fig.savefig(folder / "CLIPPING_ALL_FITS.png")
    plt.close(fig)
    fig, axes = plt.subplots(
        (len(fits) + 3) // 4, 4, figsize=(14, 3 * ((len(fits) + 3) // 4)), squeeze=False
    )
    for axis, path in zip(axes.ravel(), fits, strict=False):
        curve = curves[path.name]
        axis.plot(curve["update"], curve.loss.rolling(50, min_periods=1).mean(), lw=1)
        status = read(path / "CHECKPOINT_RECEIPT.json")["status"]
        axis.set_title(path.name + (" · incomplete" if status != "COMPLETE" else ""))
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Own objective, mean of 50")
    for axis in axes.ravel()[len(fits) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Training curves · separate y scales; absolute losses are not comparable across objectives"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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

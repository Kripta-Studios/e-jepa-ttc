"""Closed P/H fits and curve diagnostics for remaining conditional V18 families."""

from __future__ import annotations

import numpy as np
import pandas as pd

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN


def build() -> dict:
    """Diagnose existing results only; no capacity experiment or coefficient search."""
    locations = {
        "PAIR-CORR-S7": RUN,
        "PAIR-CORR-FROZEN-S7": RUN,
        "PAIR-CORR-GEOMETRY-S7": RUN / "adaptive/C1_geometry",
    }
    result = {}
    for name, root in locations.items():
        populations = {}
        for role in ("P", "H"):
            folder = root / "predictions" / role / name
            receipt = read(folder / "COMPLETE.json")
            if (
                receipt["status"] != "COMPLETE"
                or sha(folder / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError(
                    "Conditional-branch diagnosis requires complete unchanged predictions"
                )
            value = read(folder / "METRICS.json")
            populations[role] = {
                "prediction_sha256": receipt["sha256"],
                "metrics_sha256": sha(folder / "METRICS.json"),
                "aggregate": value["overall_MiD"],
                "bands": {band: entry["group_macro_mid"] for band, entry in value["bins"].items()},
                "mae": value["diagnostics"]["mae_s"],
                "sign": value["diagnostics"]["sign_error_rate"],
            }
        curve_path = root / "fits" / name / "CURVE.csv"
        curve = pd.read_csv(curve_path)
        norm_column = "norm" if name == "PAIR-CORR-GEOMETRY-S7" else "grad_norm"
        if len(curve) != 12000 or not np.array_equal(curve["update"], np.arange(1, 12001)):
            raise ValueError("Conditional-branch curve is incomplete")
        blocks = []
        for start in range(0, 12000, 1000):
            block = curve.iloc[start : start + 1000]
            blocks.append(
                {
                    "first_update": start + 1,
                    "last_update": start + 1000,
                    "mean_train_loss": float(block.loss.mean()),
                    "median_train_loss": float(block.loss.median()),
                    "mean_gradient_norm": float(block[norm_column].mean()),
                }
            )
        result[name] = {
            "populations": populations,
            "curve_sha256": sha(curve_path),
            "fixed_1000_update_blocks": blocks,
            "H_minus_P_aggregate": populations["H"]["aggregate"] - populations["P"]["aggregate"],
        }
    diagnostics = read(RUN / "diagnostics/real_temporal_P/RESULT.json")
    report = {
        "source_sha256": sha(ROOT / "operational/joint_measurement/branch_diagnostics.py"),
        "models": result,
        "correspondence_diagnostic_sha256": sha(RUN / "diagnostics/real_temporal_P/RESULT.json"),
        "registered_correspondence_signal": diagnostics["grouped"]["PAIR-CORR-S7"][
            "geometry_followup_signal"
        ],
        "interpretation_limits": [
            "P is training and H held-out for these producers; neither is official evaluation",
            "Nonzero P error or a flat curve alone does not identify insufficient capacity",
            "Gradient norms do not prove changed weights; separate endpoint proofs are retained",
            "Stochastic train-mode loss blocks differ from full eval-mode P metrics",
            "No extra family or hyperparameter is selected by this diagnostic",
        ],
        "optimizer_updates": 0,
        "V_used": False,
    }
    write(RUN / "diagnostics/conditional_families/P_H_FIT.json", report)
    return {
        name: {
            "P": value["populations"]["P"],
            "H": value["populations"]["H"],
            "first_1000_loss": value["fixed_1000_update_blocks"][0]["mean_train_loss"],
            "last_1000_loss": value["fixed_1000_update_blocks"][-1]["mean_train_loss"],
        }
        for name, value in result.items()
    }


if __name__ == "__main__":
    print(build())

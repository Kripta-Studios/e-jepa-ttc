#!/usr/bin/env python
"""Collect v3 base/ablation summaries into one comparison CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def row(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    evaluation = value["best_evaluation"]
    metrics = evaluation["metrics"]
    return {
        "path": path.as_posix(),
        "arm": path.parent.parent.name,
        "seed": value.get("seed"),
        "pretrained": get(value, "pretraining", "used"),
        "best_epoch": value.get("best_epoch"),
        "best_score": value.get("best_score"),
        "macro_mid": get(
            metrics,
            "sequence_macro",
            "sequence_macro_paper_MiD_overall",
        ),
        "global_mid": metrics.get("paper_MiD_overall"),
        "crucial_mid": get(metrics, "bins", "crucial", "mid"),
        "small_mid": get(metrics, "bins", "small", "mid"),
        "large_mid": get(metrics, "bins", "large", "mid"),
        "negative_mid": get(metrics, "bins", "negative", "mid"),
        "mae_s": metrics.get("mae_s"),
        "median_ae_s": metrics.get("median_ae_s"),
        "expansion_pearson": get(
            evaluation,
            "signed_expansion_health",
            "pearson",
        ),
        "expansion_mae": get(
            evaluation,
            "signed_expansion_health",
            "mae",
        ),
        "balanced_sign_accuracy": get(
            evaluation,
            "direction",
            "balanced_accuracy",
        ),
        "negative_recall": get(
            evaluation,
            "direction",
            "negative_recall",
        ),
        "sign_auc": get(evaluation, "direction", "auc"),
        "saturation_rate": evaluation.get("ttc_saturation_rate"),
        "latent_prediction_cosine": evaluation.get(
            "latent_prediction_cosine"
        ),
        "geometry_prior_macro_mid": get(
            evaluation,
            "geometry_prior_metrics",
            "sequence_macro",
            "sequence_macro_paper_MiD_overall",
        ),
        "residual_abs_mean": evaluation.get(
            "learned_residual_abs_mean"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.root.rglob("summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary.json files below {args.root}")
    frame = pd.DataFrame([row(path) for path in paths])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

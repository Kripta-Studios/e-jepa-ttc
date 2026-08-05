#!/usr/bin/env python
"""Create controlled v3 ablation YAMLs without editing the base experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


VARIANTS = {
    "supervised_only": {
        "latent_prediction_weight": 0.0,
        "latent_variance_weight": 0.0,
        "activity_reconstruction_weight": 0.0,
        "ordered_swap_weight": 0.0,
    },
    "no_latent_jepa": {
        "latent_prediction_weight": 0.0,
        "latent_variance_weight": 0.0,
    },
    "no_ratio_aux": {
        "visible_ratio_weight": 0.0,
        "official_ratio_weight": 0.0,
    },
    "no_activity_aux": {
        "activity_reconstruction_weight": 0.0,
    },
    "no_swap_aux": {
        "ordered_swap_weight": 0.0,
    },
    "no_geometry_regularizer": {
        "geometry_prior_weight": 0.0,
    },
    "frozen_backbone": {
        "scratch_backbone_learning_rate": 0.0,
        "pretrained_backbone_learning_rate": 0.0,
        "pretrained_backbone_warmup_epochs": 30,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-train",
        type=Path,
        default=Path("configs/train/garl_object_signed_expansion_screen_v3.yaml"),
    )
    parser.add_argument(
        "--base-model",
        default="../../model/e_jepa_object_signed_expansion_event_v3.yaml",
    )
    parser.add_argument(
        "--output-train-dir",
        type=Path,
        default=Path("configs/train/object_signed_expansion_v3_ablations"),
    )
    parser.add_argument(
        "--output-experiment-dir",
        type=Path,
        default=Path("configs/experiment/object_signed_expansion_v3_ablations"),
    )
    args = parser.parse_args()

    base = yaml.safe_load(args.base_train.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("Base train YAML must contain a mapping")

    args.output_train_dir.mkdir(parents=True, exist_ok=True)
    args.output_experiment_dir.mkdir(parents=True, exist_ok=True)

    for name, overrides in VARIANTS.items():
        train_value = dict(base)
        train_value.update(overrides)
        train_path = args.output_train_dir / f"{name}.yaml"
        train_path.write_text(
            yaml.safe_dump(train_value, sort_keys=False),
            encoding="utf-8",
        )

        experiment_value = {
            "model": args.base_model,
            "finetuning": (
                "../../train/object_signed_expansion_v3_ablations/"
                f"{name}.yaml"
            ),
            "protocol": f"object_signed_expansion_v3_ablation_{name}",
            "selection": "same_gated_selection_as_v3",
        }
        experiment_path = args.output_experiment_dir / f"{name}.yaml"
        experiment_path.write_text(
            yaml.safe_dump(experiment_value, sort_keys=False),
            encoding="utf-8",
        )
        print(experiment_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pretrain the EvTTC BASE encoder on CARLA DVS Looming without TTC labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.training.carla_jepa import (  # noqa: E402
    CarlaJEPATrainerConfig,
    inspect_carla_jepa_pairs,
    pretrain_carla_jepa,
)


def _profile(name: str) -> CarlaJEPATrainerConfig:
    if name == "smoke":
        return CarlaJEPATrainerConfig(
            epochs=2,
            batch_size=4,
            gradient_accumulation=1,
            num_workers=4,
            horizons_ms=(50, 100),
            max_windows_per_sequence=2,
            max_train_samples=32,
            max_validation_samples=16,
            early_stopping_min_epochs=1,
            early_stopping_patience=0,
        )
    return CarlaJEPATrainerConfig()


def main() -> int:
    """Parse a resource-aware profile and run CARLA JEPA pretraining."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/CARLA_DVS_Looming_Dataset/random_spawn"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/carla_dvs_looming_v1.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/carla_dvs_looming_blocked_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact profile from output/resume.pt after an interrupted epoch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact sequence/pair counts without reading event payloads or using GPU.",
    )
    args = parser.parse_args()

    config = _profile(args.profile)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_workers": args.workers,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
    }
    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    output = args.output or Path(f"artifacts/runs/carla_jepa_{args.profile}_seed{config.seed}")
    if args.dry_run:
        inspection = inspect_carla_jepa_pairs(
            root=args.root,
            manifest_path=args.manifest,
            split_path=args.split,
            config=config,
        )
        print(json.dumps(inspection, indent=2, sort_keys=True))
        return 0
    summary = pretrain_carla_jepa(
        root=args.root,
        manifest_path=args.manifest,
        split_path=args.split,
        output_dir=output,
        config=config,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pretrain EvTTC BASE on public eAP events without TTC labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.training.eap_jepa import (  # noqa: E402
    EAPJEPATrainerConfig,
    inspect_eap_jepa_windows,
    pretrain_eap_jepa,
)


def _profile(name: str, objective: str) -> EAPJEPATrainerConfig:
    geometry_weight = 0.25 if objective == "geo" else 0.0
    if name == "smoke":
        return EAPJEPATrainerConfig(
            epochs=1,
            batch_size=4,
            gradient_accumulation=1,
            num_workers=0,
            horizons_ms=(100,),
            max_windows_per_sequence=2,
            max_train_samples=12,
            max_validation_samples=4,
            early_stopping_min_epochs=1,
            early_stopping_patience=0,
            geometry_loss_weight=geometry_weight,
        )
    if name == "pilot":
        return EAPJEPATrainerConfig(
            epochs=3,
            batch_size=24,
            gradient_accumulation=2,
            num_workers=8,
            max_train_samples=1024,
            max_validation_samples=256,
            early_stopping_min_epochs=2,
            early_stopping_patience=1,
            geometry_loss_weight=geometry_weight,
        )
    return EAPJEPATrainerConfig(geometry_loss_weight=geometry_weight)


def main() -> int:
    """Run a resource-aware, sequence-disjoint eAP SSL or geometry pilot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("ssl", "geo"), required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/manifests/eap_train40_inventory_v1.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        help="Signed split override; defaults to pilot-12 or train-40 by profile.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-windows-per-sequence", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--geometry-loss-weight", type=float)
    parser.add_argument("--patch-objectness-weight", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count projected windows without opening event payloads, RGB, or the GPU.",
    )
    args = parser.parse_args()

    config = _profile(args.profile, args.objective)
    split = args.split or Path(
        "data/splits/eap_train40_v1.json"
        if args.profile == "full"
        else "data/splits/eap_pilot12_v1.json"
    )
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
        "max_windows_per_sequence": args.max_windows_per_sequence,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "geometry_loss_weight": args.geometry_loss_weight,
        "patch_objectness_weight": args.patch_objectness_weight,
    }
    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    if args.objective == "ssl" and config.geometry_loss_weight != 0.0:
        raise ValueError("The SSL control must keep geometry-loss-weight at zero.")
    if args.objective == "geo" and config.geometry_loss_weight <= 0.0:
        raise ValueError("The Geo objective requires a positive geometry-loss-weight.")
    output = args.output or Path(
        f"artifacts/runs/eap_{args.objective}_{args.profile}_seed{config.seed}"
    )
    if args.dry_run:
        result = inspect_eap_jepa_windows(
            root=args.root,
            inventory_path=args.inventory,
            split_path=split,
            config=config,
        )
    else:
        result = pretrain_eap_jepa(
            root=args.root,
            inventory_path=args.inventory,
            split_path=split,
            output_dir=output,
            config=config,
            device_name=args.device,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

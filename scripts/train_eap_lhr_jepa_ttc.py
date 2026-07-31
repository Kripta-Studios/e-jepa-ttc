"""Train the complete official-label eAP LHR object-JEPA TTC estimator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.models.eap_lhr_jepa_ttc import EAPLHRJEPATTCConfig  # noqa: E402
from e_jepa_ttc.training.eap_lhr_jepa_ttc import (  # noqa: E402
    EAPLHRTrainerConfig,
    train_eap_lhr_jepa_ttc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geo-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument("--ttc-residual-scale-s", type=float, default=0.25)
    parser.add_argument("--height-weight", type=float, default=0.5)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--ttc-weight", type=float, default=0.25)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--category-weight", type=float, default=0.05)
    parser.add_argument("--foreground-weight", type=float, default=0.0)
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    model = EAPLHRJEPATTCConfig(
        use_rgb=args.use_rgb,
        ttc_residual_scale_s=args.ttc_residual_scale_s,
    )
    trainer = EAPLHRTrainerConfig(
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        num_workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        precision=args.precision,
        seed=args.seed,
        height_loss_weight=args.height_weight,
        ratio_loss_weight=args.ratio_weight,
        ttc_loss_weight=args.ttc_weight,
        jepa_loss_weight=args.jepa_weight,
        geometry_loss_weight=args.geometry_weight,
        category_loss_weight=args.category_weight,
        foreground_loss_weight=args.foreground_weight,
        balanced_sampling=not args.no_balanced_sampling,
    )
    result = train_eap_lhr_jepa_ttc(
        manifest_path=args.manifest,
        output_dir=args.output,
        geo_checkpoint=args.geo_checkpoint,
        model_config=model,
        trainer_config=trainer,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

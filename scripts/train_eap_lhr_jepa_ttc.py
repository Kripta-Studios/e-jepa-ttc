"""Train the complete official-label eAP LHR object-JEPA TTC estimator."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.models.eap_lhr_jepa_ttc import EAPLHRJEPATTCConfig  # noqa: E402
from e_jepa_ttc.training.eap_lhr_jepa_ttc import (  # noqa: E402
    EAPLHRTrainerConfig,
    train_eap_lhr_jepa_ttc,
)


def _config_use_rgb(config_path: Path | None) -> bool:
    """Read modality and reject architecture declarations this trainer ignores."""

    if config_path is None:
        return False
    if not config_path.is_file():
        raise FileNotFoundError(f"Fine-tuning config does not exist: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Fine-tuning config must contain a YAML mapping.")
    model_ref = value.get("model")
    if isinstance(model_ref, str) and model_ref.startswith("e_jepa_tubelet_lhr"):
        raise NotImplementedError(
            "The supervised EAPLHRJEPATTC trainer does not construct "
            f"{model_ref!r}. The Tubelet high-resolution model is currently "
            "screen-only; refusing to silently train the legacy pooled model."
        )
    if isinstance(model_ref, str):
        candidates = (config_path.parent / model_ref, ROOT / model_ref)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            model_value: Any = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            if isinstance(model_value, dict):
                model_name = str(model_value.get("model", ""))
                if model_name.startswith("e_jepa_tubelet_lhr"):
                    raise NotImplementedError(
                        "The supervised EAPLHRJEPATTC trainer does not construct "
                        f"{model_name!r}. The Tubelet high-resolution model is currently "
                        "screen-only; refusing to silently train the legacy pooled model."
                    )
                return bool(model_value.get("use_rgb", False))
    return bool(value.get("use_rgb", False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--cache-manifest", dest="manifest", type=Path, required=True)
    parser.add_argument(
        "--geo-checkpoint",
        "--pretrained",
        dest="geo_checkpoint",
        type=Path,
    )
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
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
    parser.add_argument("--disable-observable-motion", action="store_true")
    parser.add_argument("--ttc-residual-scale-s", type=float, default=0.25)
    parser.add_argument("--height-weight", type=float, default=0.5)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--ttc-weight", type=float, default=0.25)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--category-weight", type=float, default=0.05)
    parser.add_argument("--foreground-weight", type=float, default=0.0)
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    use_rgb = args.use_rgb or _config_use_rgb(args.config)
    model = EAPLHRJEPATTCConfig(
        use_rgb=use_rgb,
        use_observable_motion=not args.disable_observable_motion,
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
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )
    try:
        result = train_eap_lhr_jepa_ttc(
            manifest_path=args.manifest,
            output_dir=args.output,
            geo_checkpoint=args.geo_checkpoint,
            model_config=model,
            trainer_config=trainer,
            device_name=args.device,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILURE.json").write_text(
            json.dumps(
                {
                    "artifact_type": "eap_lhr_training_failure_v1",
                    "status": "interrupted",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "manifest": str(args.manifest),
                    "output": str(args.output),
                    "device": args.device,
                    "error_type": "KeyboardInterrupt",
                    "error_message": "Training interrupted by the operator.",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 130
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILURE.json").write_text(
            json.dumps(
                {
                    "artifact_type": "eap_lhr_training_failure_v1",
                    "status": "failed",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "manifest": str(args.manifest),
                    "output": str(args.output),
                    "device": args.device,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

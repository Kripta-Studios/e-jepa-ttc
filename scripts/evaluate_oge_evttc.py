"""Evaluate a frozen OGE checkpoint on explicit EvTTC development/OOD splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.training.object_geo_trainer import evaluate_object_geo_ttc_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "calibration", "test"),
        default=("validation",),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--allow-diagnostic-test",
        action="store_true",
        help=(
            "Explicitly open the labelled EvTTC family holdout. This is not authorization "
            "for the separate sealed Benchmark-10."
        ),
    )
    args = parser.parse_args()
    result = evaluate_object_geo_ttc_checkpoint(
        checkpoint_path=args.checkpoint,
        cache_manifest_path=args.cache_manifest,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        allow_diagnostic_test=args.allow_diagnostic_test,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

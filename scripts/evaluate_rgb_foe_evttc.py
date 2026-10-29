"""Evaluate the source-traceable RGB/FoE geometry baseline on an EvTTC cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.evaluation.rgb_foe_evttc import (
    RGBFOEEvTTCConfig,
    evaluate_rgb_foe_evttc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-step", type=int, default=2)
    parser.add_argument("--minimum-flow-px", type=float, default=0.05)
    parser.add_argument("--maximum-flow-px", type=float, default=64.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--no-foreground-mask",
        action="store_true",
        help="Fit the entire shared object ROI instead of the labelled foreground mask.",
    )
    args = parser.parse_args()
    result = evaluate_rgb_foe_evttc(
        cache_manifest=args.cache_manifest,
        output_dir=args.output,
        config=RGBFOEEvTTCConfig(
            grid_step=args.grid_step,
            minimum_flow_px=args.minimum_flow_px,
            maximum_flow_px=args.maximum_flow_px,
            use_foreground_mask=not args.no_foreground_mask,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

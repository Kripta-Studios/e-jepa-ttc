"""Evaluate fixed reliability-gated fusion of neural and geometric TTC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.evaluation.geometry_fusion import (
    GeometryFusionConfig,
    evaluate_geometry_fusion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--geometry-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-scale-px", type=float, default=1.0)
    parser.add_argument("--maximum-geometry-weight", type=float, default=1.0)
    parser.add_argument("--disagreement-log-scale", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate_geometry_fusion(
        neural_predictions_path=args.neural_predictions,
        geometry_predictions_path=args.geometry_predictions,
        output_dir=args.output,
        config=GeometryFusionConfig(
            residual_scale_px=args.residual_scale_px,
            maximum_geometry_weight=args.maximum_geometry_weight,
            disagreement_log_scale=args.disagreement_log_scale,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

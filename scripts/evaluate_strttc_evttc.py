"""Evaluate the source-traceable causal STRTTC port on an EvTTC dev split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.evaluation.strttc_evttc import (
    EvTTCSTRTTCConfig,
    evaluate_evttc_strttc,
)
from e_jepa_ttc.geometry.strttc_frontend import STRTTCFrontendConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--lookback-ms", type=int, default=200)
    parser.add_argument("--maximum-contour-points", type=int, default=512)
    parser.add_argument("--nonlinear-refinement", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = evaluate_evttc_strttc(
        manifest_path=args.manifest,
        split_path=args.split,
        split_name=args.split_name,
        output_dir=args.output,
        config=EvTTCSTRTTCConfig(
            lookback_s=args.lookback_ms / 1000.0,
            maximum_samples_per_sequence=args.samples_per_sequence,
            nonlinear_refinement=args.nonlinear_refinement,
            frontend=STRTTCFrontendConfig(
                maximum_contour_points=args.maximum_contour_points,
                seed=args.seed,
            ),
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

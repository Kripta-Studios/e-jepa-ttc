"""Evaluate causal bbox geometry with a neural fallback on the same cache rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.baselines.causal_geometry import (
    run_cache_aligned_causal_geometry_hybrid,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derivative-window", type=int, default=21)
    parser.add_argument("--max-ttc-seconds", type=float, default=60.0)
    args = parser.parse_args()
    result = run_cache_aligned_causal_geometry_hybrid(
        manifest_path=args.manifest,
        cache_manifest_path=args.cache_manifest,
        neural_predictions_path=args.neural_predictions,
        output_path=args.output,
        derivative_window=args.derivative_window,
        max_ttc_seconds=args.max_ttc_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

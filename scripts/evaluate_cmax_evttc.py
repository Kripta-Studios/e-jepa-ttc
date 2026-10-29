"""Evaluate causal raw-event radial CMax on an EvTTC development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.evaluation.cmax_evttc import EvTTCCMaxConfig, evaluate_evttc_cmax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--lookback-ms", type=int, default=200)
    parser.add_argument("--maximum-events", type=int, default=50_000)
    parser.add_argument("--minimum-events", type=int, default=1_000)
    parser.add_argument("--coarse-steps", type=int, default=33)
    args = parser.parse_args()
    result = evaluate_evttc_cmax(
        manifest_path=args.manifest,
        split_path=args.split,
        split_name=args.split_name,
        output_dir=args.output,
        config=EvTTCCMaxConfig(
            lookback_s=args.lookback_ms / 1000.0,
            maximum_samples_per_sequence=args.samples_per_sequence,
            maximum_events_per_sample=args.maximum_events,
            minimum_roi_events=args.minimum_events,
            coarse_steps=args.coarse_steps,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

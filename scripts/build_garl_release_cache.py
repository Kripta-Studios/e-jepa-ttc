"""Build a cache with the unchanged official Garl-TTC input semantics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garl_release_cache import (  # noqa: E402
    GarlReleaseCacheConfig,
    build_garl_release_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--roi-bins", type=int, default=10)
    parser.add_argument("--event-pixel-diff", type=int, default=5)
    parser.add_argument("--target-delta-t-s", type=float, default=0.1)
    parser.add_argument("--delta-t-tolerance-s", type=float, default=0.025)
    parser.add_argument("--fy", type=float, default=1694.1323524131867)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--expected-rows", type=int, default=88_744)
    args = parser.parse_args()

    config = GarlReleaseCacheConfig(
        roi_size=args.roi_size,
        roi_bins=args.roi_bins,
        event_pixel_diff=args.event_pixel_diff,
        target_delta_t_s=args.target_delta_t_s,
        delta_t_tolerance_s=args.delta_t_tolerance_s,
        fy=args.fy,
        shard_size=args.shard_size,
        workers=args.workers,
        include_rgb=not args.no_rgb,
        compression_level=args.compression_level,
        expected_rows=args.expected_rows,
    )
    manifest = build_garl_release_cache(
        eap_root=args.eap_root,
        garlttc_root=args.garlttc_root,
        split_path=args.split,
        output_dir=args.output,
        config=config,
        max_samples_per_split=args.max_samples_per_split,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

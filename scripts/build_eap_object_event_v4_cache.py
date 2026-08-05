#!/usr/bin/env python
"""Build the three-step common-coordinate Object Event TTC v4 cache."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402
    GarlTTCLHRCacheConfig,
    materialize_garlttc_lhr_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("screen", "full"), default="screen")
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--margin-fraction", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    maximum = args.max_samples_per_split
    if maximum is None and args.profile == "screen":
        maximum = 2048
    config = GarlTTCLHRCacheConfig(
        roi_size=args.roi_size,
        shard_size=args.shard_size,
        store_full_frame_events=False,
        store_garl_event_roi=False,
        store_jepa_event_roi=False,
        store_event_v4_common_roi=True,
        event_v4_margin_fraction=args.margin_fraction,
        event_v4_require_precontext=True,
        event_v4_precontext_fallback="shifted_event_window",
        include_rgb=False,
        include_masks=False,
        workers=args.workers,
        preprocessing_device="cpu",
        compression="none",
    )
    result = materialize_garlttc_lhr_cache(
        eap_root=args.eap_root.resolve(),
        garlttc_root=args.garlttc_root.resolve(),
        split_path=args.split.resolve(),
        output_dir=args.output_dir.resolve(),
        config=config,
        max_samples_per_split=maximum,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "profile": args.profile,
                "config": asdict(config),
                "manifest": (args.output_dir.resolve() / "manifest.json").as_posix(),
                "split_counts": result.get("split_counts"),
                "precontext_motion_valid_fraction": result.get(
                    "precontext_motion_valid_fraction"
                ),
                "event_v4_precontext_valid_fraction": result.get(
                    "event_v4_precontext_valid_fraction"
                ),
                "event_v4_precontext_source_counts": result.get(
                    "event_v4_precontext_source_counts"
                ),
                "object_lhr_extension": result.get("object_lhr_extension"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

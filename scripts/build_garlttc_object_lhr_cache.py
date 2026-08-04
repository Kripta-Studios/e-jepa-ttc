#!/usr/bin/env python
"""Build the official object-centric GarlTTC cache used by Object-LHR."""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preprocessing-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compression", choices=("none", "gzip"), default="none")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument(
        "--storage-profile",
        choices=("object_lhr_minimal", "legacy_full"),
        default="object_lhr_minimal",
    )
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--require-masks", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    minimal = args.storage_profile == "object_lhr_minimal"
    config = GarlTTCLHRCacheConfig(
        roi_size=args.roi_size,
        shard_size=args.shard_size,
        store_full_frame_events=not minimal,
        store_garl_event_roi=not minimal,
        store_jepa_event_roi=True,
        workers=args.workers,
        preprocessing_device=args.preprocessing_device,
        compression=args.compression,
        compression_level=args.compression_level,
        include_rgb=args.include_rgb,
        include_masks=args.include_masks or args.require_masks,
        mask_required=args.require_masks,
    )
    manifest = materialize_garlttc_lhr_cache(
        eap_root=args.eap_root.resolve(),
        garlttc_root=args.garlttc_root.resolve(),
        split_path=args.split.resolve(),
        output_dir=args.output_dir.resolve(),
        config=config,
        max_samples_per_split=args.max_samples_per_split,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

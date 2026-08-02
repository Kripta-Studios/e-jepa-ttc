"""Build the leakage-safe official GarlTTC LHR object cache."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--target-delta-t-s", type=float, default=0.1)
    parser.add_argument("--delta-t-tolerance-s", type=float, default=0.025)
    parser.add_argument("--jepa-context-delta-t-s", type=float, default=0.1)
    parser.add_argument("--jepa-context-tolerance-s", type=float, default=0.05)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--expected-train-rows", type=int, default=88_744)
    parser.add_argument("--allow-dataset-version-change", action="store_true")
    parser.add_argument(
        "--calibration-mode",
        choices=("official_constant_fy", "per_sample_eap_intrinsics"),
        default="official_constant_fy",
    )
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--preprocessing-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compression", choices=("none", "gzip"), default="none")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="CPU materialization processes; use 1 with --preprocessing-device cuda.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only shards with matching build state and SHA-256 sidecars.",
    )
    args = parser.parse_args()
    config = GarlTTCLHRCacheConfig(
        include_rgb=args.include_rgb,
        target_delta_t_s=args.target_delta_t_s,
        delta_t_tolerance_s=args.delta_t_tolerance_s,
        jepa_context_delta_t_s=args.jepa_context_delta_t_s,
        jepa_context_tolerance_s=args.jepa_context_tolerance_s,
        shard_size=args.shard_size,
        expected_train_rows=args.expected_train_rows,
        allow_dataset_version_change=args.allow_dataset_version_change,
        calibration_mode=args.calibration_mode,
        selection_seed=args.selection_seed,
        preprocessing_device=args.preprocessing_device,
        workers=args.workers,
        compression=args.compression,
        compression_level=args.compression_level,
    )
    result = materialize_garlttc_lhr_cache(
        eap_root=args.eap_root,
        garlttc_root=args.garlttc_root,
        split_path=args.split,
        output_dir=args.output,
        config=config,
        max_samples_per_split=args.max_samples_per_split,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

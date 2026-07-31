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
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--expected-train-rows", type=int, default=88_744)
    parser.add_argument("--allow-dataset-version-change", action="store_true")
    args = parser.parse_args()
    config = GarlTTCLHRCacheConfig(
        include_rgb=args.include_rgb,
        shard_size=args.shard_size,
        expected_train_rows=args.expected_train_rows,
        allow_dataset_version_change=args.allow_dataset_version_change,
    )
    result = materialize_garlttc_lhr_cache(
        eap_root=args.eap_root,
        garlttc_root=args.garlttc_root,
        split_path=args.split,
        output_dir=args.output,
        config=config,
        max_samples_per_split=args.max_samples_per_split,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

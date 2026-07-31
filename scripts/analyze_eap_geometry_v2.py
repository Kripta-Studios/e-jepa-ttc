"""Summarize official GarlTTC LHR-v2 motion/category balance."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheDataset  # noqa: E402
from e_jepa_ttc.utils.io import write_structured  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = GarlTTCLHRCacheDataset(args.manifest, splits=tuple(args.splits))
    groups: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for index in range(len(dataset)):
        sample = dataset[index]
        groups[str(sample.get("sampling_group", "unlabelled"))] += 1
        categories[str(sample.get("category", "unknown"))] += 1
    result = {
        "artifact_type": "garlttc_lhr_v2_balance_analysis",
        "sample_count": len(dataset),
        "sampling_groups": dict(sorted(groups.items())),
        "categories": dict(sorted(categories.items())),
    }
    write_structured(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

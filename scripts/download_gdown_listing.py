"""Download files from a `gdown --json` folder listing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_src = Path(__file__).resolve().parents[1] / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

from e_jepa_ttc.data.downloads import download_gdown_listing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--suffix", action="append", default=[])
    args = parser.parse_args()

    records = download_gdown_listing(
        listing_path=args.listing,
        output_dir=args.output_dir,
        skip_existing=not args.overwrite,
        suffixes=tuple(args.suffix),
    )
    print(json.dumps({"count": len(records), "records": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

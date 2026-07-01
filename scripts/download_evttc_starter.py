"""Plan or execute EvTTC starter downloads from a public URL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from e_jepa_ttc.data.downloads import build_download_plan, run_gdown_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/evttc_starter_downloads.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--kind", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--continue", dest="resume", action="store_true")
    args = parser.parse_args()

    plan = build_download_plan(
        manifest_path=args.manifest,
        root=args.root,
        sequences=tuple(args.sequence),
        kinds=tuple(args.kind),
    )
    print(json.dumps({"count": len(plan), "plan": plan}, indent=2, sort_keys=True))
    if args.execute:
        run_gdown_plan(plan, python=sys.executable, quiet=args.quiet, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI wrapper for the package-level regenerable report builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tables/regenerable_report")
    )
    args = parser.parse_args()
    payload = build_report(args.repo_root.resolve(), args.output_dir)
    print(json.dumps({key: payload[key] for key in ("artifact_count", "status_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the frozen eAP public-train detection population without opening test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.detection.campaign_data import audit_population


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pinned-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report, _, _ = audit_population(
        data_root=args.data_root,
        candidate_split_path=args.split,
        pinned_manifest_path=args.pinned_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "roles": report["roles"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

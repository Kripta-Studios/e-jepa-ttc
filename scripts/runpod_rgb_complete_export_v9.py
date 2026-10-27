"""Publish the quiescent global RGB closure with V9 compatibility evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration.rgb_return_archive_v9 import export_if_ready


def _rotated(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        rotation_raw, separator, path = value.partition("=")
        if not separator or not rotation_raw.isdigit():
            raise ValueError("Run roots use ROTATION=/absolute/path")
        rotation = int(rotation_raw)
        if rotation in result:
            raise ValueError(f"Duplicate R{rotation} root")
        result[rotation] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-run", action="append", required=True, metavar="R=/PATH")
    parser.add_argument("--a5-run", action="append", required=True, metavar="R=/PATH")
    parser.add_argument("--c2f-run", action="append", required=True, metavar="R=/PATH")
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--operational-source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = export_if_ready(
        _rotated(args.downstream_run),
        _rotated(args.a5_run),
        _rotated(args.c2f_run),
        coverage_receipt=args.coverage_receipt,
        authorization_root=args.authorization_root,
        operational_sources=args.operational_source,
        output=args.output,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())

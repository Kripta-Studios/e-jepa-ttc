"""Publish the isolated R4 return through exact composed V8 authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration.rgb_return_archive_r4_v8 import export_if_ready


def _r4(value: str) -> dict[int, Path]:
    rotation, separator, path = value.partition("=")
    if not separator or rotation != "4" or not path:
        raise ValueError("The isolated exporter requires exactly 4=/absolute/path")
    return {4: Path(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-run", required=True, metavar="4=/PATH")
    parser.add_argument("--a5-run", required=True, metavar="4=/PATH")
    parser.add_argument("--c2f-run", required=True, metavar="4=/PATH")
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--operational-source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(4,), required=True)
    args = parser.parse_args()
    value = export_if_ready(
        _r4(args.downstream_run),
        _r4(args.a5_run),
        _r4(args.c2f_run),
        coverage_receipt=args.coverage_receipt,
        authorization_root=args.authorization_root,
        operational_sources=args.operational_source,
        output=args.output,
        rotations=(args.rotation,),
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())

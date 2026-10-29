#!/usr/bin/env python
"""Aggregate only complete, identity-verified E-Clock X0 OOF runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.evaluation.collision_clock_aggregate import aggregate_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute canonical metrics, paired bootstrap and gates for one X0 arm."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Read-only root containing the signed official A5 artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = aggregate_run(
        config_path=args.config.resolve(),
        protocol_path=args.protocol.resolve(),
        reference_path=args.reference.resolve(),
        run_root=args.run_root.resolve(),
        source_root=args.reference_root.resolve(),
        schema_root=repo / "schemas",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

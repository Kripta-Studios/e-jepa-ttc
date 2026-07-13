"""Aggregate evaluation metrics JSON files across seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from e_jepa_ttc.evaluation.aggregate import DEFAULT_METRIC_NAMES, aggregate_metric_files
from e_jepa_ttc.utils.io import ensure_parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Split key to aggregate, e.g. test.")
    parser.add_argument(
        "--split-protocol",
        type=Path,
        help="Split YAML/JSON whose claim metadata gates publication status.",
    )
    parser.add_argument(
        "--claim-level",
        choices=("development", "diagnostic", "official", "final"),
        default="diagnostic",
        help="Intended status of the generated aggregate table.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        help="Metric name to include. Repeat for multiple metrics.",
    )
    parser.add_argument("metrics_json", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = tuple(args.metrics) if args.metrics else DEFAULT_METRIC_NAMES
    payload = aggregate_metric_files(
        args.metrics_json,
        split=args.split,
        metric_names=metrics,
        split_protocol_path=args.split_protocol,
        claim_level=args.claim_level,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        ensure_parent(args.output)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

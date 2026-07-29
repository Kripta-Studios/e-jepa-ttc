"""Validate a frozen EvTTC submission package without changing predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.evaluation.submission_writer import validate_submission_root
from e_jepa_ttc.utils.io import write_structured


def _queries(path: Path | None) -> dict[str, list[tuple[int, float]]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(sequence["sequence_id"]): [
            (int(row["index"]), float(row["timestamp"])) for row in sequence["queries"]
        ]
        for sequence in payload["sequences"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--require-sequences", type=int)
    parser.add_argument("--write-report", type=Path, required=True)
    parser.add_argument("--require-finite", action="store_true")
    parser.add_argument("--require-positive-ttc", action="store_true")
    parser.add_argument("--require-nonnegative-runtime", action="store_true")
    parser.add_argument("--require-index-match", action="store_true")
    parser.add_argument("--require-timestamp-match", action="store_true")
    args = parser.parse_args()
    report = validate_submission_root(
        args.submission_root,
        expected_queries=_queries(args.benchmark_manifest),
        require_sequences=args.require_sequences,
    )
    write_structured(args.write_report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

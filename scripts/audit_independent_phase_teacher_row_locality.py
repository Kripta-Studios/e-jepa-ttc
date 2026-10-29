"""Audit historical DINO teacher row locality without loading model tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    """Run the exact-source CPU audit and optionally write a new receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-source", type=Path, required=True)
    parser.add_argument("--relation-source", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit.teacher_cache import audit_historical_row_locality

    result = audit_historical_row_locality(
        teacher_source_path=args.teacher_source,
        relation_source_path=args.relation_source,
        model_config_path=args.model_config,
        weight_path=args.weight_path,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

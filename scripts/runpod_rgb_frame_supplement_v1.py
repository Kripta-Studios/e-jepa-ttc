"""Plan or build the minimal path-compatible RGB frame supplement for Pod1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operational.runpod_migration.rgb_frame_supplement import (
    build_supplement,
    plan_supplement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--descriptor", type=Path, required=True)
    plan.add_argument("--fold-contract", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--rotations", type=int, nargs="+", default=[2, 3, 4])
    plan.add_argument(
        "--roles", choices=("refiner", "generation"), nargs="+", default=["refiner", "generation"]
    )
    build = subparsers.add_parser("build")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--destination-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        plan_supplement(
            args.descriptor,
            args.fold_contract,
            args.output,
            rotations=tuple(args.rotations),
            roles=tuple(args.roles),
        )
        if args.command == "plan"
        else build_supplement(args.plan, args.destination_root, args.output)
    )
    summary = {
        key: result[key]
        for key in (
            "schema",
            "status",
            "tar_count",
            "member_count",
            "receipt_sha256",
        )
    }
    summary["bytes"] = result.get("sparse_tar_bytes", result.get("minimum_sparse_tar_bytes"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

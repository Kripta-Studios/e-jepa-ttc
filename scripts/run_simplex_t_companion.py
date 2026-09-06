"""SIMPLEX-T T0 audit/resumption entry point; scientific integration is pending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_delivery import package_t0
from e_jepa_ttc.artifacts.simplex_t_preflight import audit, interface_status, write_new_json
from e_jepa_ttc.simplex_t.cache_status import context_cache_status
from e_jepa_ttc.simplex_t.compiled_context import compile_fold


def main() -> int:
    """Fail closed for scientific commands until owner interfaces are integrated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "prepare", "run", "status", "package"])
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compile-fold",
        type=int,
        choices=(0, 1, 2),
        help="prepare: compile one complete outer fold into a new --output directory",
    )
    parser.add_argument("--compile-pool", choices=("D0", "D1"), default=None)
    args = parser.parse_args()
    if args.compile_pool is not None and args.compile_fold is None:
        parser.error("--compile-pool requires --compile-fold")
    if args.compile_fold is not None:
        if args.command != "prepare" or args.output is None or args.resume:
            parser.error("--compile-fold requires prepare, a new --output, and no --resume")
        paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
        temporal = Path(paths["worktree"]) / "artifacts/simplex_t/T1"
        pool = args.compile_pool or "D0"
        prefix = "expansion_" if pool == "D1" else ""
        compile_fold(
            temporal / f"{prefix}context_features_fp32",
            temporal / f"{prefix}query_context_index",
            temporal / f"{prefix}query_context_dedup",
            args.output,
            args.compile_fold,
            pool=pool,
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE",
                    "outer": args.compile_fold,
                    "pool": pool,
                    "compiled": str(args.output),
                    "optimizer_updates": 0,
                }
            )
        )
        return 0
    if args.command == "package":
        if args.output is None:
            parser.error("package requires a new --output directory")
        print(json.dumps(package_t0(args.local_paths, args.output), indent=2))
        return 0
    if args.command == "audit":
        if args.output is None:
            parser.error("audit requires a new --output JSON evidence path")
        result = audit(args.local_paths)
        write_new_json(args.output, result)
        print(json.dumps(result["interfaces"], indent=2))
        return 0
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    result = interface_status(paths)
    result["amended_context_execution"] = context_cache_status(Path(paths["worktree"]))
    if args.command == "status":
        result["historical_interface_status"] = result.get("status")
        result["status"] = result["amended_context_execution"]["status"]
    result["implementation_status"] = "INDEPENDENT_COMPONENTS_NOT_SCIENTIFICALLY_FROZEN"
    print(json.dumps(result, indent=2))
    return 0 if args.command == "status" else 3


if __name__ == "__main__":
    raise SystemExit(main())

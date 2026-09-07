"""SIMPLEX-T T0 audit/resumption entry point; scientific integration is pending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_delivery import package_t0
from e_jepa_ttc.artifacts.simplex_t_preflight import audit, interface_status, write_new_json
from e_jepa_ttc.simplex_t.cache_status import context_cache_status
from e_jepa_ttc.simplex_t.compiled_context import compile_fold
from e_jepa_ttc.simplex_t.configuration_preflight import inspect_source_configuration
from e_jepa_ttc.simplex_t.configured_preparation import prepare_configured_sources
from e_jepa_ttc.simplex_t.reuse_catalog import D0ReuseCatalog


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
    parser.add_argument("--compile-pool", choices=("D0", "D1", "DENSE_OLD"), default=None)
    parser.add_argument("--reuse-d0-compiled", type=Path)
    parser.add_argument("--reuse-d0-compiled-sha256")
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--source-config-sha256")
    parser.add_argument(
        "--source-identities",
        action="store_true",
        help="prepare: load acknowledged cached sources and persist resumable identities; no fits",
    )
    parser.add_argument(
        "--other-reserved-bytes",
        type=int,
        help="pending output bytes on the shared write volume; required for identities/compilation",
    )
    args = parser.parse_args()
    if args.source_identities or (
        args.other_reserved_bytes is not None and args.compile_fold is None
    ):
        if (
            not args.source_identities
            or args.command != "prepare"
            or args.source_config is None
            or args.source_config_sha256 is None
            or args.output is None
            or args.other_reserved_bytes is None
            or args.other_reserved_bytes < 0
            or args.compile_fold is not None
            or args.compile_pool is not None
            or args.reuse_d0_compiled is not None
            or args.reuse_d0_compiled_sha256 is not None
        ):
            parser.error("source identities require prepare, source pins, output and reservations")
        try:
            result = prepare_configured_sources(
                args.local_paths,
                args.source_config,
                args.source_config_sha256,
                args.output,
                other_reserved_bytes=args.other_reserved_bytes,
                resume=args.resume,
            )
        except (InterruptedError, RuntimeError) as error:
            if not str(error).startswith(("PAUSED_RESOURCE:", "RESOURCE_PAUSE:")):
                raise
            result = {
                "status": "PAUSED_RESOURCE",
                "reason": str(error),
                "output": str(args.output),
                "optimizer_updates": 0,
                "scientific_freeze": False,
            }
        if result["status"] not in {
            "PAUSED_RESOURCE",
            "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE",
        }:
            raise ValueError("source preparation returned an unexpected terminal status")
        print(json.dumps(result, indent=2))
        return 3 if result["status"] == "PAUSED_RESOURCE" else 0
    if args.source_config is not None or args.source_config_sha256 is not None:
        if (
            args.command != "prepare"
            or args.output is None
            or args.resume
            or args.source_config is None
            or args.source_config_sha256 is None
            or args.compile_fold is not None
            or args.compile_pool is not None
            or args.reuse_d0_compiled is not None
            or args.reuse_d0_compiled_sha256 is not None
        ):
            parser.error("source inspection requires prepare, both source pins and a new --output")
        result = inspect_source_configuration(
            args.local_paths, args.source_config, args.source_config_sha256, args.output
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.reuse_d0_compiled is not None or args.reuse_d0_compiled_sha256 is not None:
        if (
            args.reuse_d0_compiled is None
            or args.reuse_d0_compiled_sha256 is None
            or args.compile_pool != "DENSE_OLD"
            or args.compile_fold is None
        ):
            parser.error("D0 reuse requires both source pins and a DENSE_OLD compilation")
    if args.compile_pool is not None and args.compile_fold is None:
        parser.error("--compile-pool requires --compile-fold")
    if args.compile_fold is not None:
        if args.command != "prepare" or args.output is None or args.resume:
            parser.error("--compile-fold requires prepare, a new --output, and no --resume")
        if args.other_reserved_bytes is None or args.other_reserved_bytes < 0:
            parser.error("compilation requires explicit nonnegative --other-reserved-bytes")
        paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
        temporal = Path(paths["worktree"]) / "artifacts/simplex_t/T1"
        pool = args.compile_pool or "D0"
        prefix = {"D0": "", "D1": "expansion_", "DENSE_OLD": "dense_"}[pool]
        reuse_options = {}
        if args.reuse_d0_compiled is not None and args.reuse_d0_compiled_sha256 is not None:
            reuse_options["reuse"] = D0ReuseCatalog(
                compiled=args.reuse_d0_compiled,
                compiled_sha256=args.reuse_d0_compiled_sha256,
                cache=temporal / "context_features_fp32",
                index_root=temporal / "query_context_index",
                dedup=temporal / "query_context_dedup" / f"outer{args.compile_fold}.npz",
                outer=args.compile_fold,
            )
        compile_fold(
            temporal / f"{prefix}context_features_fp32",
            temporal / f"{prefix}query_context_index",
            temporal / f"{prefix}query_context_dedup",
            args.output,
            args.compile_fold,
            pool=pool,
            other_reserved_bytes=args.other_reserved_bytes,
            **reuse_options,
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

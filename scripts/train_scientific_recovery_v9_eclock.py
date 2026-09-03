#!/usr/bin/env python
"""Fail-closed CLI for E-Clock X0 DAG inspection and future OOF execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e_jepa_ttc.evaluation.collision_clock_runner import (
    dry_run_dag,
    load_runner_contracts,
    run_outer_folds,
    run_outer_train_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or execute the frozen outer-train/outer-dev E-Clock pipeline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "outer-train-smoke", "oof"), required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Explicit read-only train8192 cache root; never inferred from the V8 checkout.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Explicit read-only root containing official A5 fold checkpoints.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--execute-authorized-outer-train-smoke", action="store_true")
    parser.add_argument("--execute-authorized-oof", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config, protocol, reference = load_runner_contracts(repo, config_path)
    if args.mode == "dry-run":
        print(
            json.dumps(
                dry_run_dag(
                    config=config,
                    config_path=config_path,
                    protocol=protocol,
                    reference=reference,
                    cache_root=args.cache_root,
                    source_root=args.reference_root,
                    output_root=args.output_root,
                ),
                sort_keys=True,
            )
        )
        return 0
    device = torch.device(args.device)
    if args.mode == "outer-train-smoke":
        if not args.execute_authorized_outer_train_smoke or args.fold is None:
            raise PermissionError(
                "real-data smoke requires --fold and explicit "
                "--execute-authorized-outer-train-smoke authorization"
            )
        print(
            json.dumps(
                run_outer_train_smoke(
                    repo=repo,
                    config=config,
                    protocol=protocol,
                    reference=reference,
                    cache_root=args.cache_root,
                    source_root=args.reference_root,
                    outer_fold=args.fold,
                    device=device,
                ),
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_authorized_oof:
        raise PermissionError("OOF requires explicit future --execute-authorized-oof authorization")
    summaries = run_outer_folds(
        repo=repo,
        config_path=config_path,
        config=config,
        protocol=protocol,
        reference=reference,
        cache_root=args.cache_root,
        source_root=args.reference_root,
        output_root=args.output_root,
        device=device,
    )
    print(json.dumps(summaries, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

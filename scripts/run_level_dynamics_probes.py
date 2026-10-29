"""Run frozen Level--Dynamics probes from an embedding/metadata artifact."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.level_dynamics_probes import (  # noqa: E402
    load_embedding_metadata_artifact,
    run_identity_shortcut_diagnostics,
    run_level_dynamics_probes,
    write_identity_diagnostics,
    write_probe_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit sequence-disjoint numeric/categorical probes on a frozen embedding "
            "artifact. No model, parquet, or EvTTC input is accepted."
        )
    )
    parser.add_argument(
        "--artifact", type=Path, required=True, help="Frozen .npz or JSON artifact."
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--identity-diagnostics-output",
        type=Path,
        help="Optional separate identity_shortcut_diagnostics_v1 output path.",
    )
    parser.add_argument("--checkpoint-hash", help="Checkpoint hash bound into diagnostics.")
    parser.add_argument("--manifest-hash", help="Matched-manifest hash bound into diagnostics.")
    parser.add_argument("--config-hash", help="Resolved config hash bound into probe artifacts.")
    parser.add_argument("--code-commit", help="Exact 40-hex code commit bound into artifacts.")
    parser.add_argument("--context-s", type=float, default=0.2)
    parser.add_argument("--max-horizon-s", type=float, default=0.3)
    parser.add_argument("--guard-gap-s", type=float)
    parser.add_argument("--identity-folds", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        embeddings, metadata = load_embedding_metadata_artifact(args.artifact)
        required_bindings = {
            "--checkpoint-hash": args.checkpoint_hash,
            "--manifest-hash": args.manifest_hash,
            "--config-hash": args.config_hash,
            "--code-commit": args.code_commit,
        }
        missing_bindings = [name for name, value in required_bindings.items() if not value]
        if missing_bindings:
            raise ValueError("Probe artifacts require bindings: " + ", ".join(missing_bindings))
        result = run_level_dynamics_probes(
            embeddings,
            metadata,
            checkpoint_hash=args.checkpoint_hash,
            manifest_hash=args.manifest_hash,
            config_hash=args.config_hash,
            code_commit=args.code_commit,
            seed=args.seed,
            ridge_alpha=args.ridge_alpha,
        )
        write_probe_outputs(result, args.output_json, args.output_csv)
        if args.identity_diagnostics_output is not None:
            if not args.checkpoint_hash or not args.manifest_hash:
                raise ValueError(
                    "Identity diagnostics require --checkpoint-hash and --manifest-hash."
                )
            identity = run_identity_shortcut_diagnostics(
                embeddings,
                metadata,
                checkpoint_hash=args.checkpoint_hash,
                manifest_hash=args.manifest_hash,
                config_hash=args.config_hash,
                code_commit=args.code_commit,
                seed=args.seed,
                context_s=args.context_s,
                max_horizon_s=args.max_horizon_s,
                guard_gap_s=args.guard_gap_s,
                n_folds=args.identity_folds,
            )
            write_identity_diagnostics(identity, args.identity_diagnostics_output)
        print(f"Frozen probes written: {args.output_json.resolve()}")
        return 0
    except Exception as exc:
        print(f"Frozen probe request rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

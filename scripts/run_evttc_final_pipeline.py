"""Reproducible orchestration for EvTTC architecture selection and final diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from e_jepa_ttc.training.object_geo_trainer import evaluate_object_geo_ttc_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_evttc_architecture_matrix.py"
AGGREGATOR = ROOT / "scripts" / "aggregate_evttc_architecture_selection.py"
FREEZER = ROOT / "scripts" / "freeze_final_architecture.py"
DEFAULT_RUN_ROOT = Path(
    "artifacts/runs/evttc32_architecture_v4_grouped_cv_confirm/core"
)
DEFAULT_CACHE_ROOT = Path("artifacts/features")
CORE_VARIANTS = (
    "A0_MATCHED_GLOBAL",
    "A1_MATCHED_DENSE_BLOCK",
    "A2_MATCHED_DENSE_ATTNRES",
    "K1_OBJECT_KDA",
)


def _auto_workers() -> int:
    return min(12, max(4, (os.cpu_count() or 8) // 2))


def _run(command: list[str], *, dry_run: bool = False) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _winner(aggregate_path: Path, requested: str | None) -> str:
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    complete = [row for row in payload["ranking"] if row["complete_for_final_selection"]]
    if requested is not None:
        row = next((item for item in complete if item["variant"] == requested), None)
        if row is None:
            raise ValueError(f"Variant {requested!r} is absent or incomplete.")
        return requested
    if not complete:
        raise ValueError("No CV-complete architecture is available to freeze.")
    return str(complete[0]["variant"])


def _execution_args(profile: str, batch_size: int, accumulation: int) -> list[str]:
    if profile == "matched":
        if batch_size or accumulation:
            raise ValueError("matched uses the frozen config; do not override batch controls.")
        return []
    active_batch = batch_size or 32
    active_accumulation = accumulation or 1
    if active_batch * active_accumulation != 32:
        raise ValueError("throughput must preserve the frozen effective batch of 32.")
    return [
        "--batch-size",
        str(active_batch),
        "--gradient-accumulation",
        str(active_accumulation),
    ]


def compare(args: argparse.Namespace) -> int:
    execution = _execution_args(args.execution_profile, args.batch_size, args.accumulation)
    for seed in args.seeds:
        for fold in args.folds:
            command = [
                sys.executable,
                str(RUNNER),
                "--mode",
                "confirm",
                "--stage-role",
                "core",
                "--split-protocol",
                "grouped_cv",
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--workers",
                str(args.workers),
                "--base-initialization",
                "random_control",
                "--variants",
                *args.variants,
                *execution,
            ]
            if args.resume:
                command.append("--resume")
            _run(command, dry_run=args.dry_run)
    aggregate_path = args.run_root / "aggregate.json"
    _run(
        [
            sys.executable,
            str(AGGREGATOR),
            "--root",
            str(args.run_root),
            "--output",
            str(aggregate_path),
            "--expected-folds",
            str(len(args.folds)),
            "--expected-seeds",
            str(len(args.seeds)),
        ],
        dry_run=args.dry_run,
    )
    return 0


def fit_holdout(args: argparse.Namespace) -> int:
    execution = _execution_args(args.execution_profile, args.batch_size, args.accumulation)
    for seed in args.seeds:
        command = [
            sys.executable,
            str(RUNNER),
            "--mode",
            "confirm",
            "--stage-role",
            "core",
            "--split-protocol",
            "historical_base",
            "--fold",
            "0",
            "--seed",
            str(seed),
            "--workers",
            str(args.workers),
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            str(args.output_dir),
            "--include-diagnostic-test-cache",
            "--variants",
            args.variant,
            *execution,
        ]
        if args.resume:
            command.append("--resume")
        _run(command, dry_run=args.dry_run)
    return 0


def evaluate_holdout(args: argparse.Namespace) -> int:
    result = evaluate_object_geo_ttc_checkpoint(
        checkpoint_path=args.checkpoint,
        cache_manifest_path=args.cache_manifest,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        allow_diagnostic_test=args.allow_diagnostic_test,
    )
    print(json.dumps(result, indent=2))
    return 0


def freeze(args: argparse.Namespace) -> int:
    variant = _winner(args.aggregate, args.variant)
    checkpoints = sorted(
        args.run_root.glob(f"fold-*/{variant}/seed-*/best.pt"),
        key=lambda path: path.as_posix(),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {variant} under {args.run_root}.")
    command = [
        sys.executable,
        str(FREEZER),
        "--aggregate",
        str(args.aggregate),
        "--variant",
        variant,
        "--checkpoints",
        *(str(path) for path in checkpoints),
        "--candidate-name",
        args.candidate_name or variant.lower(),
        "--candidate-role",
        args.candidate_role,
        "--output",
        str(args.output),
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    _run(command, dry_run=args.dry_run)
    return 0


def validate(_: argparse.Namespace) -> int:
    tests = (
        "tests/unit/test_oge_architecture.py",
        "tests/unit/test_evttc_object_cache.py",
        "tests/unit/test_grouped_cv.py",
        "tests/unit/test_benchmark10_guard.py",
        "tests/unit/test_training_controls.py",
    )
    _run([sys.executable, "-m", "pytest", *tests, "-q"])
    _run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"])
    return 0


def _common_execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=_auto_workers())
    parser.add_argument(
        "--execution-profile",
        choices=("matched", "throughput"),
        default="matched",
        help="matched reproduces evidence; throughput uses batch 32 x accumulation 1.",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--accumulation", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compare_parser = commands.add_parser("compare", help="Run matched grouped CV.")
    compare_parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    compare_parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    compare_parser.add_argument(
        "--variants", nargs="+", choices=CORE_VARIANTS, default=list(CORE_VARIANTS[:2])
    )
    compare_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    _common_execution(compare_parser)
    compare_parser.set_defaults(func=compare)

    fit_parser = commands.add_parser(
        "fit-holdout", help="Fit one frozen architecture on train/validation."
    )
    fit_parser.add_argument("--variant", choices=CORE_VARIANTS, required=True)
    fit_parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    fit_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT / "evttc32_final_family_holdout_core",
    )
    fit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/evttc32_final_family_holdout/core/fold-0"),
    )
    _common_execution(fit_parser)
    fit_parser.set_defaults(func=fit_holdout)

    eval_parser = commands.add_parser(
        "evaluate-holdout", help="Evaluate a frozen checkpoint on explicit cache splits."
    )
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--cache-manifest", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "calibration", "test"),
        default=["validation"],
    )
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--workers", type=int, default=_auto_workers())
    eval_parser.add_argument("--allow-diagnostic-test", action="store_true")
    eval_parser.set_defaults(func=evaluate_holdout)

    freeze_parser = commands.add_parser("freeze", help="Freeze a CV-complete ensemble.")
    freeze_parser.add_argument("--aggregate", type=Path, required=True)
    freeze_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    freeze_parser.add_argument("--variant", choices=CORE_VARIANTS)
    freeze_parser.add_argument("--candidate-name")
    freeze_parser.add_argument(
        "--candidate-role",
        choices=("SINGLE_REALTIME", "ENSEMBLE_ACCURACY"),
        default="ENSEMBLE_ACCURACY",
    )
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.add_argument("--allow-dirty", action="store_true")
    freeze_parser.add_argument("--dry-run", action="store_true")
    freeze_parser.set_defaults(func=freeze)

    validate_parser = commands.add_parser("validate", help="Run the architecture gates.")
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    function = args.func
    return int(function(args))


if __name__ == "__main__":
    raise SystemExit(main())

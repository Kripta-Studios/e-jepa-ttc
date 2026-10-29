"""Reproducible orchestration for EvTTC architecture selection and final diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
RUNNER = ROOT / "scripts" / "run_evttc_architecture_matrix.py"
AGGREGATOR = ROOT / "scripts" / "aggregate_evttc_architecture_selection.py"
FREEZER = ROOT / "scripts" / "freeze_final_architecture.py"
DEFAULT_RUN_ROOT = Path("artifacts/runs/evttc32_architecture_v4_grouped_cv_confirm/core")
DEFAULT_CACHE_ROOT = Path("artifacts/features")
CORE_VARIANTS = (
    "A0_MATCHED_GLOBAL",
    "A1_MATCHED_DENSE_BLOCK",
    "R1_MATCHED_BBOX_ROI",
    "A2_MATCHED_DENSE_ATTNRES",
    "K1_OBJECT_KDA",
)
EXTERNAL_INITIALIZATIONS = frozenset(
    {
        "external_ssl",
        "external_ttc",
        "external_eap_ssl",
        "external_eap_geo",
        "external_eap_ttc",
    }
)


def _auto_workers() -> int:
    return min(12, max(4, (os.cpu_count() or 8) // 2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, dry_run: bool = False) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(SOURCE_ROOT), str(ROOT), existing_pythonpath) if part
        )
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _winner(aggregate_path: Path, requested: str | None) -> dict[str, object]:
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not payload.get("all_variants_complete"):
        raise ValueError("The aggregate is incomplete; no architecture can be frozen.")
    if payload.get("matched_control_audit_passed") is not True:
        raise ValueError("The A0/A1 matched-control audit did not pass.")
    complete = [row for row in payload["ranking"] if row["complete_for_final_selection"]]
    if requested is not None:
        row = next((item for item in complete if item["variant"] == requested), None)
        if row is None:
            raise ValueError(f"Variant {requested!r} is absent or incomplete.")
        return row
    if not complete:
        raise ValueError("No CV-complete architecture is available to freeze.")
    return complete[0]


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
    if (
        args.base_initialization in EXTERNAL_INITIALIZATIONS
        and args.base_encoder_checkpoint is None
    ):
        raise ValueError(f"{args.base_initialization} requires --base-encoder-checkpoint.")
    for seed in args.seeds:
        for fold in args.folds:
            command = [
                sys.executable,
                str(RUNNER),
                "--mode",
                args.mode,
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
                "--output-dir",
                str(args.run_root / f"fold-{fold}"),
                "--base-initialization",
                args.base_initialization,
                "--variants",
                *args.variants,
                *execution,
            ]
            if args.base_encoder_checkpoint is not None:
                command.extend(["--base-encoder-checkpoint", str(args.base_encoder_checkpoint)])
            if args.base_initialization in EXTERNAL_INITIALIZATIONS:
                command.extend(
                    [
                        "--external-pretraining-split",
                        str(args.external_pretraining_split),
                    ]
                )
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
            "--folds",
            *(str(fold) for fold in args.folds),
            "--seeds",
            *(str(seed) for seed in args.seeds),
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
    if "test" in args.splits:
        if args.selection_manifest is None:
            raise ValueError(
                "Diagnostic test evaluation requires --selection-manifest created "
                "from validation-only profile selection."
            )
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if selection.get("artifact_type") != "evttc_final_profile_selection_v1":
            raise ValueError("Selection manifest has an unsupported artifact_type.")
        if selection.get("diagnostic_test_used_for_selection") is not False:
            raise ValueError("Selection manifest is not validation-only.")
        if selection.get("benchmark10_opened") is not False:
            raise ValueError("Selection manifest reports Benchmark-10 exposure.")
        checkpoint_sha256 = _sha256(args.checkpoint)
        if selection.get("selected_checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("Checkpoint hash does not match the frozen selection.")
    sys.path.insert(0, str(SOURCE_ROOT))
    from e_jepa_ttc.training.object_geo_trainer import (
        evaluate_object_geo_ttc_checkpoint,
    )

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


def select_final(args: argparse.Namespace) -> int:
    profiles: list[dict[str, object]] = []
    invariant_rows: list[dict[str, object]] = []
    for profile_name, root in (
        ("matched", args.matched_root),
        ("throughput", args.throughput_root),
    ):
        runs: list[dict[str, object]] = []
        for seed in args.seeds:
            summary_path = root / f"seed-{seed}" / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing final-fit summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validation = summary["validation"]
            trainer = summary["trainer"]
            checkpoint_path = Path(summary["best_checkpoint"])
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing final-fit checkpoint: {checkpoint_path}")
            run = {
                "seed": seed,
                "summary": summary_path.as_posix(),
                "summary_sha256": _sha256(summary_path),
                "checkpoint": checkpoint_path.as_posix(),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "selection_score": validation["sequence_macro_selection_score"],
                "mean_relative_error": validation["sequence_macro_mean_relative_error"],
                "mae_s": validation["sequence_macro_mae_s"],
                "milliseconds_per_window": validation["milliseconds_per_window"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "batch_size": trainer["batch_size"],
                "gradient_accumulation": trainer["gradient_accumulation"],
                "workers": trainer["num_workers"],
                "git_commit": summary["git_commit"],
                "source_tree_sha256": summary.get("source_tree_sha256"),
                "benchmark10_opened": summary["benchmark10_opened"],
            }
            runs.append(run)
            invariant_rows.append(
                {
                    "profile": profile_name,
                    "seed": seed,
                    "cache_manifest_sha256": summary["cache_manifest_sha256"],
                    "train_samples": summary["train_samples"],
                    "validation_samples": summary["validation_samples"],
                    "architecture": {
                        key: value
                        for key, value in summary["architecture"].items()
                        if key != "base_encoder_checkpoint"
                    },
                }
            )
        aggregate: dict[str, object] = {
            "profile": profile_name,
            "run_count": len(runs),
            "seeds": list(args.seeds),
            "runs": runs,
        }
        for metric in ("selection_score", "mean_relative_error", "mae_s"):
            values = [float(run[metric]) for run in runs]
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = variance**0.5
        aggregate["elapsed_seconds_sum"] = sum(float(run["elapsed_seconds"]) for run in runs)
        aggregate["milliseconds_per_window_mean"] = sum(
            float(run["milliseconds_per_window"]) for run in runs
        ) / len(runs)
        profiles.append(aggregate)
    reference = invariant_rows[0]
    invariant_fields = (
        "cache_manifest_sha256",
        "train_samples",
        "validation_samples",
        "architecture",
    )
    mismatches = [
        {"profile": row["profile"], "seed": row["seed"], "field": field}
        for row in invariant_rows[1:]
        for field in invariant_fields
        if row[field] != reference[field]
    ]
    if mismatches:
        raise ValueError(f"Final profile invariants differ: {mismatches}")
    profiles.sort(key=lambda row: float(row["selection_score_mean"]))
    selected_profile = profiles[0]
    selected_run = min(
        selected_profile["runs"],
        key=lambda run: float(run["selection_score"]),
    )
    profile_map = {str(profile["profile"]): profile for profile in profiles}
    payload = {
        "artifact_type": "evttc_final_profile_selection_v1",
        "architecture": args.variant,
        "selection_split": "validation",
        "profile_selection_rule": "lowest three-seed mean sequence-macro score",
        "checkpoint_selection_rule": "lowest validation score within selected profile",
        "profiles": profiles,
        "selected_profile": selected_profile["profile"],
        "selected_seed": selected_run["seed"],
        "selected_checkpoint": selected_run["checkpoint"],
        "selected_checkpoint_sha256": selected_run["checkpoint_sha256"],
        "matched_over_throughput_time_ratio": float(profile_map["matched"]["elapsed_seconds_sum"])
        / float(profile_map["throughput"]["elapsed_seconds_sum"]),
        "matched_relative_score_improvement_pct": 100.0
        * (
            float(profile_map["throughput"]["selection_score_mean"])
            - float(profile_map["matched"]["selection_score_mean"])
        )
        / float(profile_map["throughput"]["selection_score_mean"]),
        "diagnostic_test_used_for_selection": False,
        "selection_was_frozen_before_diagnostic_test": True,
        "benchmark10_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


def freeze(args: argparse.Namespace) -> int:
    row = _winner(args.aggregate, args.variant)
    variant = str(row["variant"])
    runs = row.get("runs")
    if not isinstance(runs, list):
        raise TypeError("Aggregate winner is missing its run list.")
    checkpoints = sorted(
        (
            args.run_root
            / f"fold-{int(run['fold'])}"
            / variant
            / f"seed-{int(run['seed'])}"
            / "best.pt"
            for run in runs
        ),
        key=lambda path: path.as_posix(),
    )
    if args.candidate_role == "SINGLE_REALTIME":
        if args.single_checkpoint is None:
            raise ValueError("SINGLE_REALTIME requires --single-checkpoint.")
        checkpoints = [args.single_checkpoint]
    elif args.single_checkpoint is not None:
        raise ValueError("--single-checkpoint is only valid for SINGLE_REALTIME.")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {variant} under {args.run_root}.")
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Aggregate-selected checkpoints are missing: {missing}")
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
        "--config",
        str(args.config),
        "--preprocessing",
        str(args.preprocessing),
        "--protocol",
        str(args.protocol),
        "--selection-audit",
        str(args.selection_audit),
        "--output",
        str(args.output),
    ]
    if args.ensemble_rule is not None:
        command.extend(["--ensemble-rule", str(args.ensemble_rule)])
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
        "tests/unit/test_oge_split_evaluation.py",
        "tests/unit/test_architecture_aggregate.py",
        "tests/unit/test_final_pipeline_cli.py",
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
    compare_parser.add_argument(
        "--mode",
        choices=("screen", "confirm"),
        default="confirm",
        help="screen is a quick diagnostic; confirm is the full early-stopped protocol.",
    )
    compare_parser.add_argument(
        "--base-initialization",
        choices=(
            "random_control",
            "external_ssl",
            "external_ttc",
            "external_eap_ssl",
            "external_eap_geo",
            "external_eap_ttc",
        ),
        default="random_control",
        help=(
            "random_control reproduces the closed A0/A1 comparison; external_ssl "
            "loads label-free CARLA pretraining; external_ttc loads the separate "
            "CARLA JEPA+synthetic-TTC ablation; the external_eap arms load public "
            "eAP train-only SSL, weak geometry, or TTC pretraining."
        ),
    )
    compare_parser.add_argument("--base-encoder-checkpoint", type=Path)
    compare_parser.add_argument(
        "--external-pretraining-split",
        type=Path,
        default=Path("data/splits/carla_dvs_looming_blocked_v1.json"),
    )
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
    eval_parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Required for test: validation-only selection with the checkpoint hash.",
    )
    eval_parser.set_defaults(func=evaluate_holdout)

    select_parser = commands.add_parser(
        "select-final", help="Select matched/throughput profile and seed on validation."
    )
    select_parser.add_argument("--variant", choices=CORE_VARIANTS, required=True)
    select_parser.add_argument("--matched-root", type=Path, required=True)
    select_parser.add_argument("--throughput-root", type=Path, required=True)
    select_parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.set_defaults(func=select_final)

    freeze_parser = commands.add_parser("freeze", help="Freeze a CV-complete ensemble.")
    freeze_parser.add_argument("--aggregate", type=Path, required=True)
    freeze_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    freeze_parser.add_argument("--variant", choices=CORE_VARIANTS)
    freeze_parser.add_argument("--candidate-name")
    freeze_parser.add_argument("--single-checkpoint", type=Path)
    freeze_parser.add_argument(
        "--candidate-role",
        choices=("SINGLE_REALTIME", "ENSEMBLE_ACCURACY"),
        default="ENSEMBLE_ACCURACY",
    )
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--preprocessing", type=Path, required=True)
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--selection-audit", type=Path, required=True)
    freeze_parser.add_argument("--ensemble-rule", type=Path)
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

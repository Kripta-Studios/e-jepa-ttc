"""Run paired eAP pretraining, matched EvTTC A0/A1 fine-tuning, and comparison."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PRETRAIN = REPOSITORY_ROOT / "scripts" / "pretrain_eap_jepa.py"
EVTTC_PIPELINE = REPOSITORY_ROOT / "scripts" / "run_evttc_final_pipeline.py"
COMPARE = REPOSITORY_ROOT / "scripts" / "compare_evttc_initializations.py"
VARIANTS = ("A0_MATCHED_GLOBAL", "A1_MATCHED_DENSE_BLOCK")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_logged(
    name: str,
    command: list[str],
    *,
    log_dir: Path,
    stages: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    log_path = log_dir / f"{name}.log"
    row: dict[str, Any] = {
        "name": name,
        "command": subprocess.list2cmdline(command),
        "log": log_path.as_posix(),
        "status": "planned" if dry_run else "running",
        "start_time_unix": time.time(),
    }
    stages.append(row)
    print(row["command"], flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(SOURCE_ROOT), str(REPOSITORY_ROOT), environment.get("PYTHONPATH"))
        if part
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            printable = line.encode(
                sys.stdout.encoding or "utf-8",
                errors="replace",
            ).decode(sys.stdout.encoding or "utf-8")
            print(printable, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    row["end_time_unix"] = time.time()
    row["elapsed_seconds"] = row["end_time_unix"] - row["start_time_unix"]
    row["return_code"] = return_code
    row["status"] = "completed" if return_code == 0 else "failed"
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _aggregate_complete(
    path: Path,
    variants: tuple[str, ...],
    *,
    folds: list[int],
    seeds: list[int],
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_pairs = {(fold, seed) for fold in folds for seed in seeds}
    rows = {
        str(row.get("variant")): row for row in payload.get("ranking", []) if isinstance(row, dict)
    }
    if payload.get("all_variants_complete") is not True or not set(variants) <= rows.keys():
        return False
    for variant in variants:
        row = rows[variant]
        actual_pairs = {
            (int(run["fold"]), int(run["seed"]))
            for run in row.get("runs", [])
            if isinstance(run, dict) and "fold" in run and "seed" in run
        }
        if (
            row.get("complete_for_final_selection") is not True
            or actual_pairs != expected_pairs
            or int(row.get("run_count", -1)) != len(expected_pairs)
            or int(row.get("required_run_count", -1)) != len(expected_pairs)
        ):
            return False
    return True


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        result["ram_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        result["ram_bytes"] = None
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_memory_bytes": (
                    int(torch.cuda.get_device_properties(0).total_memory)
                    if torch.cuda.is_available()
                    else 0
                ),
            }
        )
    except ImportError:
        result["torch"] = None
    return result


def _objectives(requested: list[str]) -> tuple[str, ...]:
    if "both" in requested:
        return ("ssl", "geo")
    return tuple(dict.fromkeys(requested))


def main() -> int:
    """Orchestrate quick analysis or the complete paired transfer protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("analysis", "full"), default="analysis")
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("both", "ssl", "geo"),
        default=["both"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", "pretrain", "evttc-control", "transfer", "compare"),
        default=["all"],
    )
    parser.add_argument("--eap-root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument(
        "--eap-inventory",
        type=Path,
        default=Path("data/manifests/eap_train40_inventory_v1.json"),
    )
    parser.add_argument(
        "--eap-split",
        type=Path,
        help="Signed split override; defaults to pilot-12 for analysis and train-40 for full.",
    )
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--pretrain-seed", type=int, default=42)
    parser.add_argument("--eap-workers", type=int, default=8)
    parser.add_argument("--evttc-workers", type=int, default=12)
    parser.add_argument("--eap-batch-size", type=int, default=24)
    parser.add_argument("--eap-gradient-accumulation", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("artifacts/metrics"),
    )
    parser.add_argument("--orchestration-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    objectives = _objectives(args.objectives)
    requested = set(args.stages)
    if "all" in requested:
        requested = {"pretrain", "evttc-control", "transfer", "compare"}
    folds = args.folds or ([0] if args.profile == "analysis" else list(range(5)))
    seeds = args.seeds or ([7] if args.profile == "analysis" else [7, 13, 21])
    if len(set(folds)) != len(folds) or not set(folds) <= set(range(5)):
        raise ValueError("EvTTC folds must be unique values in [0, 4].")
    if len(set(seeds)) != len(seeds):
        raise ValueError("EvTTC seeds must be unique.")
    if (
        min(
            args.eap_workers + 1,
            args.evttc_workers + 1,
            args.eap_batch_size,
            args.eap_gradient_accumulation,
        )
        <= 0
    ):
        raise ValueError("Worker, batch, and accumulation controls must be positive.")
    pretrain_profile = "pilot" if args.profile == "analysis" else "full"
    evttc_mode = "screen" if args.profile == "analysis" else "confirm"
    eap_split = args.eap_split or Path(
        "data/splits/eap_pilot12_v1.json"
        if args.profile == "analysis"
        else "data/splits/eap_train40_v1.json"
    )
    orchestration_dir = args.orchestration_dir or (args.run_root / f"eap_evttc_{args.profile}_v1")
    log_dir = orchestration_dir / "logs"
    status_path = orchestration_dir / "orchestration_status.json"
    control_root = args.run_root / f"evttc32_eap_random_control_{args.profile}_v1" / "core"
    control_aggregate = control_root / "aggregate.json"
    stages: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "artifact_type": "eap_evttc_complete_orchestration_v1",
        "status": "running",
        "profile": args.profile,
        "objectives": objectives,
        "requested_stages": sorted(requested),
        "folds": folds,
        "seeds": seeds,
        "pretrain_seed": args.pretrain_seed,
        "eap_split": eap_split.as_posix(),
        "hardware": _hardware(),
        "resource_profile": {
            "eap_workers": args.eap_workers,
            "evttc_workers": args.evttc_workers,
            "eap_batch_size": args.eap_batch_size,
            "eap_gradient_accumulation": args.eap_gradient_accumulation,
            "precision": "bf16",
            "pin_memory": True,
            "persistent_workers": True,
            "fused_adamw_when_supported": True,
            "tf32": True,
        },
        "early_stopping": {
            "eap_enabled": True,
            "eap_patience": 6 if args.profile == "full" else 1,
            "eap_minimum_epochs": 8 if args.profile == "full" else 2,
            "evttc_enabled": True,
            "evttc_patience": 6 if args.profile == "full" else 2,
            "evttc_minimum_epochs": 10 if args.profile == "full" else 3,
        },
        "benchmark10_opened": False,
        "stages": stages,
    }
    _write_json(status_path, status)
    try:
        for objective in objectives:
            pretrain_name = (
                f"eap_{objective}_{pretrain_profile}_seed{args.pretrain_seed}_v1"
                if args.profile == "analysis"
                else f"eap_{objective}_train40_full_seed{args.pretrain_seed}_v1"
            )
            pretrain_run = args.run_root / pretrain_name
            checkpoint = pretrain_run / "eap_jepa_encoder_best.pt"
            metrics = pretrain_run / "metrics.json"
            if "pretrain" in requested:
                if args.resume and checkpoint.is_file() and metrics.is_file():
                    stages.append(
                        {
                            "name": f"01_eap_{objective}_pretrain",
                            "status": "skipped_complete",
                        }
                    )
                else:
                    command = [
                        sys.executable,
                        str(PRETRAIN),
                        "--objective",
                        objective,
                        "--profile",
                        pretrain_profile,
                        "--root",
                        str(args.eap_root),
                        "--inventory",
                        str(args.eap_inventory),
                        "--split",
                        str(eap_split),
                        "--output",
                        str(pretrain_run),
                        "--device",
                        args.device,
                        "--workers",
                        str(args.eap_workers),
                        "--batch-size",
                        str(args.eap_batch_size),
                        "--gradient-accumulation",
                        str(args.eap_gradient_accumulation),
                        "--seed",
                        str(args.pretrain_seed),
                    ]
                    if args.resume and (pretrain_run / "resume.pt").is_file():
                        command.append("--resume")
                    _run_logged(
                        f"01_eap_{objective}_pretrain",
                        command,
                        log_dir=log_dir,
                        stages=stages,
                        dry_run=args.dry_run,
                    )
        common = [
            "--mode",
            evttc_mode,
            "--folds",
            *(str(value) for value in folds),
            "--seeds",
            *(str(value) for value in seeds),
            "--variants",
            *VARIANTS,
            "--workers",
            str(args.evttc_workers),
        ]
        if args.resume:
            common.append("--resume")
        if "evttc-control" in requested:
            if args.resume and _aggregate_complete(
                control_aggregate,
                VARIANTS,
                folds=folds,
                seeds=seeds,
            ):
                stages.append({"name": "02_evttc_random_control", "status": "skipped_complete"})
            else:
                _run_logged(
                    "02_evttc_random_control",
                    [
                        sys.executable,
                        str(EVTTC_PIPELINE),
                        "compare",
                        *common,
                        "--base-initialization",
                        "random_control",
                        "--run-root",
                        str(control_root),
                    ],
                    log_dir=log_dir,
                    stages=stages,
                    dry_run=args.dry_run,
                )
        for objective in objectives:
            pretrain_name = (
                f"eap_{objective}_{pretrain_profile}_seed{args.pretrain_seed}_v1"
                if args.profile == "analysis"
                else f"eap_{objective}_train40_full_seed{args.pretrain_seed}_v1"
            )
            pretrain_run = args.run_root / pretrain_name
            checkpoint = pretrain_run / "eap_jepa_encoder_best.pt"
            transfer_root = (
                args.run_root / f"evttc32_eap_{objective}_transfer_{args.profile}_v1" / "core"
            )
            transfer_aggregate = transfer_root / "aggregate.json"
            if "transfer" in requested:
                if not args.dry_run and not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"eAP {objective} best checkpoint is missing: {checkpoint}."
                    )
                if args.resume and _aggregate_complete(
                    transfer_aggregate,
                    VARIANTS,
                    folds=folds,
                    seeds=seeds,
                ):
                    stages.append(
                        {
                            "name": f"03_evttc_eap_{objective}_transfer",
                            "status": "skipped_complete",
                        }
                    )
                else:
                    _run_logged(
                        f"03_evttc_eap_{objective}_transfer",
                        [
                            sys.executable,
                            str(EVTTC_PIPELINE),
                            "compare",
                            *common,
                            "--base-initialization",
                            f"external_eap_{objective}",
                            "--base-encoder-checkpoint",
                            str(checkpoint),
                            "--external-pretraining-split",
                            str(eap_split),
                            "--run-root",
                            str(transfer_root),
                        ],
                        log_dir=log_dir,
                        stages=stages,
                        dry_run=args.dry_run,
                    )
            if "compare" in requested:
                for variant in VARIANTS:
                    output = (
                        args.metrics_root
                        / f"evttc_{variant.lower()}_eap_{objective}_{args.profile}_v1.json"
                    )
                    _run_logged(
                        f"04_compare_eap_{objective}_{variant.lower()}",
                        [
                            sys.executable,
                            str(COMPARE),
                            "--control",
                            str(control_aggregate),
                            "--transfer",
                            str(transfer_aggregate),
                            "--variant",
                            variant,
                            "--candidate-label",
                            f"eap_{objective}",
                            "--output",
                            str(output),
                        ],
                        log_dir=log_dir,
                        stages=stages,
                        dry_run=args.dry_run,
                    )
        status["status"] = "planned" if args.dry_run else "completed"
    except Exception:
        status["status"] = "failed"
        raise
    finally:
        status["end_time_unix"] = time.time()
        _write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

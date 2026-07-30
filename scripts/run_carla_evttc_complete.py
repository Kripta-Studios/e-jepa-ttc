"""Run resumable CARLA SSL, EvTTC OOF control, and CARLA-to-EvTTC transfer."""

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
PRETRAIN = REPOSITORY_ROOT / "scripts" / "pretrain_carla_jepa.py"
EVALUATE_CARLA = REPOSITORY_ROOT / "scripts" / "evaluate_carla_jepa.py"
EVTTC_PIPELINE = REPOSITORY_ROOT / "scripts" / "run_evttc_final_pipeline.py"
COMPARE = REPOSITORY_ROOT / "scripts" / "compare_evttc_initializations.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                sys.stdout.encoding or "utf-8", errors="replace"
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


def _complete_aggregate(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("all_variants_complete")) and any(
        row.get("variant") == "A0_MATCHED_GLOBAL"
        and row.get("complete_for_final_selection") is True
        for row in payload.get("ranking", [])
    )


def _hardware() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        payload["ram_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        payload["ram_bytes"] = None
    try:
        import torch

        payload.update(
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
        payload["torch"] = None
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", "carla", "evttc-control", "transfer", "compare"),
        default=["all"],
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--carla-root",
        type=Path,
        default=Path("datasets/CARLA_DVS_Looming_Dataset/random_spawn"),
    )
    parser.add_argument(
        "--carla-manifest",
        type=Path,
        default=Path("data/manifests/carla_dvs_looming_v1.json"),
    )
    parser.add_argument(
        "--carla-split",
        type=Path,
        default=Path("data/splits/carla_dvs_looming_blocked_v1.json"),
    )
    parser.add_argument("--carla-run-dir", type=Path)
    parser.add_argument("--control-root", type=Path, default=Path(
        "artifacts/runs/evttc32_architecture_v4_grouped_cv_confirm/core"
    ))
    parser.add_argument("--transfer-root", type=Path, default=Path(
        "artifacts/runs/evttc32_carla_ssl_transfer_v1/core"
    ))
    parser.add_argument("--orchestration-dir", type=Path, default=Path(
        "artifacts/runs/carla_evttc_complete_v1"
    ))
    parser.add_argument("--comparison-output", type=Path, default=Path(
        "artifacts/metrics/evttc_a0_carla_ssl_transfer_v1.json"
    ))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--carla-workers", type=int, default=8)
    parser.add_argument("--evttc-workers", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested = set(args.stages)
    if "all" in requested:
        requested = {"carla", "evttc-control", "transfer", "compare"}
    carla_run = args.carla_run_dir or Path(
        f"artifacts/runs/carla_jepa_{args.profile}_seed42_v1"
    )
    best_checkpoint = carla_run / "carla_jepa_encoder_best.pt"
    carla_metrics = carla_run / "metrics.json"
    control_aggregate = args.control_root / "aggregate.json"
    transfer_aggregate = args.transfer_root / "aggregate.json"
    log_dir = args.orchestration_dir / "logs"
    status_path = args.orchestration_dir / "orchestration_status.json"
    stages: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "artifact_type": "carla_evttc_complete_orchestration_v1",
        "status": "running",
        "profile": args.profile,
        "requested_stages": sorted(requested),
        "folds": args.folds,
        "seeds": args.seeds,
        "hardware": _hardware(),
        "resource_profile": {
            "carla_workers": args.carla_workers,
            "evttc_workers": args.evttc_workers,
            "carla_batch_size": 24 if args.profile == "full" else 4,
            "carla_gradient_accumulation": 2 if args.profile == "full" else 1,
            "carla_precision": "bf16",
            "evttc_precision": "bf16",
        },
        "benchmark10_opened": False,
        "stages": stages,
    }
    _write_json(status_path, status)
    try:
        if "carla" in requested:
            if not (args.resume and carla_metrics.is_file() and best_checkpoint.is_file()):
                command = [
                    sys.executable,
                    str(PRETRAIN),
                    "--profile",
                    args.profile,
                    "--root",
                    str(args.carla_root),
                    "--manifest",
                    str(args.carla_manifest),
                    "--split",
                    str(args.carla_split),
                    "--output",
                    str(carla_run),
                    "--workers",
                    str(args.carla_workers),
                    "--device",
                    args.device,
                ]
                if args.resume and (carla_run / "resume.pt").is_file():
                    command.append("--resume")
                if args.dry_run:
                    command.append("--dry-run")
                _run_logged(
                    "01_carla_pretrain",
                    command,
                    log_dir=log_dir,
                    stages=stages,
                    dry_run=False,
                )
            else:
                stages.append({"name": "01_carla_pretrain", "status": "skipped_complete"})
            if not args.dry_run:
                for index, role in enumerate(("validation", "test"), start=2):
                    output = carla_run / f"{role}_evaluation.json"
                    if args.resume and output.is_file():
                        stages.append(
                            {"name": f"0{index}_carla_{role}", "status": "skipped_complete"}
                        )
                        continue
                    command = [
                        sys.executable,
                        str(EVALUATE_CARLA),
                        "--checkpoint",
                        str(best_checkpoint),
                        "--root",
                        str(args.carla_root),
                        "--manifest",
                        str(args.carla_manifest),
                        "--split",
                        str(args.carla_split),
                        "--role",
                        role,
                        "--output",
                        str(output),
                        "--workers",
                        str(args.carla_workers),
                        "--device",
                        args.device,
                    ]
                    if args.profile == "smoke":
                        command.extend(["--max-samples", "16"])
                    _run_logged(
                        f"0{index}_carla_{role}",
                        command,
                        log_dir=log_dir,
                        stages=stages,
                        dry_run=False,
                    )
        common = [
            "--folds",
            *(str(value) for value in args.folds),
            "--seeds",
            *(str(value) for value in args.seeds),
            "--variants",
            "A0_MATCHED_GLOBAL",
            "--workers",
            str(args.evttc_workers),
        ]
        if args.resume:
            common.append("--resume")
        if args.dry_run:
            common.append("--dry-run")
        if "evttc-control" in requested:
            if not (args.resume and _complete_aggregate(control_aggregate)):
                _run_logged(
                    "04_evttc_random_control",
                    [
                        sys.executable,
                        str(EVTTC_PIPELINE),
                        "compare",
                        *common,
                        "--base-initialization",
                        "random_control",
                        "--run-root",
                        str(args.control_root),
                    ],
                    log_dir=log_dir,
                    stages=stages,
                    dry_run=False,
                )
            else:
                stages.append(
                    {"name": "04_evttc_random_control", "status": "skipped_complete"}
                )
        if "transfer" in requested:
            if not args.dry_run and not best_checkpoint.is_file():
                raise FileNotFoundError(f"CARLA best checkpoint is missing: {best_checkpoint}.")
            if not (args.resume and _complete_aggregate(transfer_aggregate)):
                _run_logged(
                    "05_evttc_carla_transfer",
                    [
                        sys.executable,
                        str(EVTTC_PIPELINE),
                        "compare",
                        *common,
                        "--base-initialization",
                        "external_ssl",
                        "--base-encoder-checkpoint",
                        str(best_checkpoint),
                        "--external-pretraining-split",
                        str(args.carla_split),
                        "--run-root",
                        str(args.transfer_root),
                    ],
                    log_dir=log_dir,
                    stages=stages,
                    dry_run=False,
                )
            else:
                stages.append(
                    {"name": "05_evttc_carla_transfer", "status": "skipped_complete"}
                )
        if "compare" in requested:
            if args.dry_run:
                stages.append({"name": "06_compare_transfer", "status": "planned"})
            else:
                _run_logged(
                    "06_compare_transfer",
                    [
                        sys.executable,
                        str(COMPARE),
                        "--control",
                        str(control_aggregate),
                        "--transfer",
                        str(transfer_aggregate),
                        "--output",
                        str(args.comparison_output),
                    ],
                    log_dir=log_dir,
                    stages=stages,
                    dry_run=False,
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

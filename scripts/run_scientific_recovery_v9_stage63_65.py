"""Resume-safe Stage 63 to 64 to 65 state machine; gates cannot be forced."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.evaluation.stage63_65 import next_protocol_action  # noqa: E402
from scripts import run_scientific_recovery_v9_stage64 as stage64_runner  # noqa: E402
from scripts.audit_scientific_recovery_v9_stage63 import run as run_stage63  # noqa: E402
from scripts.run_scientific_recovery_v9_stage65 import run as run_stage65  # noqa: E402


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()


def _append_ledger(path: Path, state: str, **details: object) -> None:
    record = {"time_ns": time.time_ns(), "state": state, **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


class CampaignLock:
    """Exclusive PID/host lock that preserves stale evidence instead of deleting it."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> CampaignLock:
        value = {"pid": os.getpid(), "host": socket.gethostname(), "created_ns": time.time_ns()}
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            prior = json.loads(self.path.read_text(encoding="utf-8"))
            alive = prior.get("host") == socket.gethostname() and psutil.pid_exists(
                int(prior.get("pid", -1))
            )
            if alive:
                raise RuntimeError(f"campaign already has a live writer: {prior}") from exc
            preserved = self.path.with_name(f"STALE_CAMPAIGN_LOCK_{time.time_ns()}.json")
            self.path.replace(preserved)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
        return self

    def __exit__(self, *_args: object) -> None:
        if self.path.is_file():
            self.path.unlink()


def _write_training_lock(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("TRAINING_LOCK requires a clean versioned worktree")
    head = _git(repo, "rev-parse", "HEAD")
    protocol_path = repo / "configs" / "protocol" / "scientific_recovery_v9_stage63_65.json"
    handoff_protocol = args.handoff_root / "PROTOCOL.json"
    value = {
        "artifact_type": "scientific_recovery_v9_training_lock_v1",
        "training_commit": head,
        "base_commit": "e69bc5195ed4cccf88835be79797c5ef6cf6be2c",
        "x3_commit": "3eda5b799497310081934c680700a13863756e1f",
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "handoff_protocol_sha256": _sha(handoff_protocol),
        "microbatch": args.microbatch,
        "accumulation": 64 // args.microbatch,
        "updates": 3000,
        "seeds": [7, 13, 23],
        "a5_producer_seed": 7,
        "device": args.device,
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.platform(),
        "authorization": "user_goal_stage63_65_2026-09-04",
        "outer_dev_checkpoint_selection": False,
        "fixed_final_update": 3000,
    }
    value = sign_stage63_65_artifact(value, evidence_type="prospective_training_lock")
    path = args.output_root / "TRAINING_LOCK.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_args(args: argparse.Namespace, **changes: object) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(changes)
    return argparse.Namespace(**values)


def _resource_preflight(args: argparse.Namespace) -> dict[str, Any]:
    disk = psutil.disk_usage(str(args.output_root.parent))
    memory = psutil.virtual_memory()
    result: dict[str, Any] = {
        "disk_free_bytes": disk.free,
        "disk_minimum_bytes": 40 * 1024**3,
        "ram_available_fraction": memory.available / memory.total,
        "ram_minimum_available_fraction": 0.2,
    }
    if disk.free < result["disk_minimum_bytes"] or result["ram_available_fraction"] < 0.2:
        raise RuntimeError(f"resource preflight failed: {result}")
    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(torch.device(args.device))
        result.update({"vram_free_bytes": free, "vram_total_bytes": total})
        if free < 2 * 1024**3:
            raise RuntimeError(f"GPU free-memory margin is below 2 GiB: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA Stage 64 was requested but PyTorch cannot access a GPU")
    if not args.output_root.exists():
        args.output_root.mkdir(parents=True)
    elif not args.resume:
        raise FileExistsError("output root exists; an identity-checked --resume is required")
    ledger = args.output_root / "RUN_LEDGER.jsonl"
    with CampaignLock(args.output_root / "CAMPAIGN_LOCK.json"):
        preflight = _resource_preflight(args)
        _append_ledger(ledger, "resource_preflight_passed", **preflight)
        _append_ledger(ledger, "package_verification_started")
        verification = subprocess.run(
            [
                sys.executable,
                str(args.handoff_root / "tools" / "verify_package.py"),
                "--root",
                str(args.handoff_root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        _append_ledger(ledger, "package_verified", output=verification.stdout.strip())
        stage63_path = args.output_root / "stage63" / "X3_FEASIBILITY_V2.json"
        if stage63_path.is_file():
            stage63 = json.loads(stage63_path.read_text(encoding="utf-8"))
            if not verify_artifact_hash(stage63):
                raise RuntimeError("Stage 63 resume artifact signature mismatch")
        else:
            _append_ledger(ledger, "stage63_running")
            stage63 = run_stage63(_stage_args(args, max_hours=12.0))
            _append_ledger(
                ledger,
                "stage63_completed",
                decision=stage63["decision"],
                training_ready=stage63["training_ready"],
            )
        if (
            stage63.get("training_ready")
            and not (args.output_root / "qa" / "STAGE64_REAL_TRAIN_ONLY_SMOKE.json").is_file()
        ):
            _append_ledger(ledger, "stage64_real_train_only_smoke_running")
            smoke = stage64_runner.run_real_train_only_smoke(args)
            _append_ledger(ledger, "stage64_real_train_only_smoke_passed", **smoke)
        if not (args.output_root / "TRAINING_LOCK.json").is_file():
            lock = _write_training_lock(args, repo)
            _append_ledger(ledger, "training_lock_frozen", training_commit=lock["training_commit"])
        else:
            lock = json.loads((args.output_root / "TRAINING_LOCK.json").read_text(encoding="utf-8"))
            if not verify_artifact_hash(lock):
                raise RuntimeError("TRAINING_LOCK signature mismatch")
            if lock["training_commit"] != _git(repo, "rev-parse", "HEAD") or _git(
                repo, "status", "--porcelain"
            ):
                raise RuntimeError("resume worktree/commit no longer matches TRAINING_LOCK")
        integrity = bool(stage63.get("integrity_passed"))
        action = next_protocol_action(
            stage63_integrity=integrity,
            stage63_training_ready=bool(stage63.get("training_ready")),
            raw_available_or_supported=bool(stage63.get("support_readiness", {}).get("passed")),
            stage64_seed7=None,
        )
        stage64_results: dict[int, dict[str, Any]] = {}
        stage65_result: dict[str, Any] | None = None
        if action == "RUN_STAGE64_SEED7":
            _append_ledger(ledger, "stage64_seed7_running")
            result = stage64_runner.run(
                _stage_args(args, seed=7, max_hours=24.0, stage61_worktree=args.stage61_worktree)
            )
            stage64_results[7] = result
            _append_ledger(ledger, "stage64_seed7_completed", status=result["status"])
            action = next_protocol_action(
                stage63_integrity=True,
                stage63_training_ready=True,
                raw_available_or_supported=True,
                stage64_seed7=result["status"],
            )
        if action == "RUN_STAGE64_REPLICATIONS":
            for seed in (13, 23):
                _append_ledger(ledger, f"stage64_seed{seed}_running")
                result = stage64_runner.run(
                    _stage_args(
                        args, seed=seed, max_hours=48.0, stage61_worktree=args.stage61_worktree
                    )
                )
                stage64_results[seed] = result
                _append_ledger(ledger, f"stage64_seed{seed}_completed", status=result["status"])
            action = next_protocol_action(
                stage63_integrity=True,
                stage63_training_ready=True,
                raw_available_or_supported=True,
                stage64_seed7=stage64_results[7]["status"],
                replication={seed: stage64_results[seed]["status"] for seed in (13, 23)},
            )
        if action == "RUN_STAGE65":
            _append_ledger(ledger, "stage65_running")
            try:
                stage65_result = run_stage65(_stage_args(args, max_hours=1.0))
            except (FileNotFoundError, ValueError) as exc:
                stage65_result = {
                    "status": "RISK_ROUTER_PREREQUISITES_MISSING",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (args.output_root / "stage65").mkdir(exist_ok=True)
                (args.output_root / "stage65" / "STAGE65_RESULT.json").write_text(
                    json.dumps(stage65_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            _append_ledger(ledger, "stage65_completed", status=stage65_result["status"])
            action = f"STOP_{stage65_result['status']}"
        final = {
            "status": action.removeprefix("STOP_"),
            "stage63": stage63,
            "stage64": {str(seed): value for seed, value in stage64_results.items()},
            "stage65": stage65_result,
            "training_commit": lock["training_commit"],
            "automatic_next_action": "none",
        }
        (args.output_root / "CAMPAIGN_RESULT.json").write_text(
            json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        _append_ledger(ledger, "campaign_endpoint", status=final["status"])
        return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--raw-train-root", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, choices=(4, 8), default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        result = run_stage63(_stage_args(args, max_hours=12.0))
    else:
        result = run(args)
    print(json.dumps({"status": result.get("status", result.get("decision"))}))


if __name__ == "__main__":
    main()

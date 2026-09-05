"""Resume-safe Stage 63 to 64 to 65 state machine; gates cannot be forced."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.campaign_session import CampaignLock, append_transition  # noqa: E402
from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.artifacts.training_authorization import read_signed  # noqa: E402
from e_jepa_ttc.evaluation.stage63_65 import next_protocol_action  # noqa: E402
from e_jepa_ttc.training.raw_time_residual import (  # noqa: E402
    RawTrainingConfig,
    deterministic_schedule,
)
from scripts import run_scientific_recovery_v9_stage64 as stage64_runner  # noqa: E402
from scripts.audit_scientific_recovery_v9_stage63 import run as run_stage63  # noqa: E402
from scripts.run_scientific_recovery_v9_stage65 import run as run_stage65  # noqa: E402


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()


def _append_ledger(path: Path, state: str, **details: object) -> None:
    append_transition(path, state, **details)


def _write_training_lock(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("TRAINING_LOCK requires a clean versioned worktree")
    head = _git(repo, "rev-parse", "HEAD")
    protocol_path = repo / "configs" / "protocol" / "scientific_recovery_v9_stage63_65.json"
    handoff_protocol = args.handoff_root / "PROTOCOL.json"
    qa_path = args.output_root / "qa/QA_MANIFEST.json"
    qa = read_signed(qa_path)
    input_bindings = collect_lock_inputs(args, repo)
    if (
        qa.get("status") != "passed"
        or qa.get("training_commit") != head
        or qa.get("input_bindings") != input_bindings
    ):
        raise ValueError("final QA does not bind this code and complete input inventory")
    smoke_path = args.output_root / "qa/STAGE64_REAL_TRAIN_ONLY_SMOKE.json"
    if read_signed(args.output_root / "stage63/X3_FEASIBILITY_V2.json")["training_ready"]:
        smoke = read_signed(smoke_path)
        if smoke["status"] != "passed" or smoke["identity"]["training_commit"] != head:
            raise ValueError("training lock requires the current train-only smoke")
    value = {
        "artifact_type": "scientific_recovery_v9_training_lock_v1",
        "authorization_version": "complete_identity_v2",
        "input_bindings": input_bindings,
        "qa_sha256": _sha(qa_path),
        "smoke_sha256": _sha(smoke_path) if smoke_path.is_file() else None,
        "invocation": {
            name: str(getattr(args, name).resolve())
            for name in (
                "output_root",
                "reference_root",
                "stage61_worktree",
                "handoff_root",
                "raw_train_root",
                "train_parquet",
                "coherent_a5_root",
                "frozen_teacher_audit",
            )
        }
        | {"device": args.device, "microbatch": args.microbatch},
        "schedule": {
            "algorithm": "deterministic_schedule_uniform_permutations",
            "seed_expression": "encoder_seed*10+outer_fold",
            "effective_batch": 64,
            "updates": 3000,
            "global_macro_mass": "strict_sequence_bucket_train_only",
        },
        "training_config": asdict(
            RawTrainingConfig(
                microbatch=args.microbatch,
                accumulation=64 // args.microbatch,
            )
        ),
        "schedule_sha256_by_seed_outer": {
            f"seed{seed}/outer{fold}": hashlib.sha256(
                deterministic_schedule(
                    int(record["train_rows"]), 3000, 64, seed * 10 + int(fold)
                ).tobytes()
            ).hexdigest()
            for seed in (7, 13, 23)
            for fold, record in (
                read_signed(args.output_root / "crossfitted_a5_state/manifest.json")["outer_folds"]
                if (args.output_root / "crossfitted_a5_state/manifest.json").is_file()
                else {}
            ).items()
        },
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
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_num_threads": torch.get_num_threads(),
        "tf32_policy": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
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
    (args.output_root / "TRAINING_COMMIT.txt").write_text(head + "\n", encoding="ascii")
    return value


def collect_lock_inputs(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    """Build the input inventory shared identically by final QA and training lock."""
    source = args.stage61_worktree / "artifacts/scientific_recovery_v9_stage61_stage62"
    paths = {
        "protocol": repo / "configs/protocol/scientific_recovery_v9_stage63_65.json",
        "handoff_protocol": args.handoff_root / "PROTOCOL.json",
        "stage63": args.output_root / "stage63/X3_FEASIBILITY_V2.json",
        "supervision": source / "stage61/aggregate_seed7/R2_oof.csv",
        "router_reference": source / "stage61/aggregate_seed7/RouterR_oof.csv",
        "frozen_teacher_audit": args.frozen_teacher_audit,
        "historical_x3_manifest": source / "X3_RAW_BINDING_MANIFEST.json",
        "stage61_canonical_metadata": source / "feature_cache/outer0_final.metadata.csv",
    }
    x3 = read_signed(paths["historical_x3_manifest"])
    paths["train_parquet"] = Path(x3["garl_train_manifest"]["path"])
    if _sha(paths["train_parquet"]) != x3["garl_train_manifest"]["sha256"]:
        raise ValueError("physical train parquet differs from the preserved X3 source identity")
    if _sha(paths["stage61_canonical_metadata"]) != x3["stage_metadata"]["sha256"]:
        raise ValueError("canonical token metadata differs from the preserved X3 source identity")
    stage63 = read_signed(paths["stage63"])
    prerequisites = args.output_root / "stage65/PREREQUISITES_MANIFEST.json"
    if prerequisites.is_file():
        proof = read_signed(prerequisites)
        if proof.get("status") != "passed" or proof.get("producer_count") != 36:
            raise ValueError("Stage65 prerequisite proof is incomplete")
        paths["stage65_prerequisites"] = prerequisites
        for name, binding in proof["input_bindings"].items():
            path = Path(binding["path"])
            if path.stat().st_size != binding["bytes"] or _sha(path) != binding["sha256"]:
                raise ValueError(f"Stage65 source changed after prerequisite audit: {name}")
            paths[f"stage65_source:{name}"] = path
    if stage63.get("training_ready"):
        for outer in range(3):
            for role in ("final", "inner0", "inner1", "inner2"):
                for name in ("manifest.json", "state.npy", "metadata.csv"):
                    paths[f"coherent_a5:outer{outer}_{role}/{name}"] = (
                        args.coherent_a5_root / f"outer{outer}_{role}" / name
                    )
        paths["raw_binding"] = args.output_root / "stage63/X3_RAW_BINDING_V2.csv"
        paths["raw_binding_manifest"] = args.output_root / "stage63/X3_RAW_BINDING_MANIFEST_V2.json"
        for cache in ("raw_temporal_cache", "crossfitted_a5_state"):
            root = args.output_root / cache
            paths[f"{cache}:manifest"] = root / "manifest.json"
            manifest = read_signed(root / "manifest.json")
            if cache == "raw_temporal_cache":
                groups = {"": manifest["components"]}
            else:
                groups = {
                    f"outer{fold}": value["components"]
                    for fold, value in manifest["outer_folds"].items()
                }
            for group, components in groups.items():
                for name in components:
                    component = (root / group / name).resolve(strict=True)
                    if component.parent != (root / group).resolve(strict=True):
                        raise ValueError("lock input component escapes cache root")
                    paths[f"{cache}:{group}/{name}"] = component
    else:
        paths["stage65_prerequisites"] = args.output_root / "stage65/PREREQUISITES_MANIFEST.json"
    return {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
        for name, path in paths.items()
    }


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
        raise TimeoutError(f"resource preflight failed: {result}")
    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(torch.device(args.device))
        result.update({"vram_free_bytes": free, "vram_total_bytes": total})
        if free < 2 * 1024**3:
            raise TimeoutError(f"GPU free-memory margin is below 2 GiB: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    if (repo / "CAMPAIGN_INTEGRITY_HOLD.json").exists():
        raise RuntimeError("Campaign integrity hold: remediation and a new training lock required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA Stage 64 was requested but PyTorch cannot access a GPU")
    if not args.output_root.exists():
        args.output_root.mkdir(parents=True)
    elif not args.resume:
        raise FileExistsError("output root exists; an identity-checked --resume is required")
    ledger = args.output_root / "RUN_LEDGER.jsonl"
    with CampaignLock(args.output_root / "CAMPAIGN_LOCK.json"):
        previous_result = args.output_root / "CAMPAIGN_RESULT.json"
        if previous_result.is_file():
            preserved = args.output_root / f"PRIOR_CAMPAIGN_RESULT_{time.time_ns()}.json"
            previous_result.replace(preserved)
            _append_ledger(
                ledger,
                "previous_attempt_preserved",
                preserved_path=str(preserved),
                sha256=_sha(preserved),
            )
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
        if stage63.get("training_ready"):
            _append_ledger(ledger, "stage64_real_train_only_smoke_running")
            smoke = stage64_runner.run_real_train_only_smoke(args)
            _append_ledger(ledger, "stage64_real_train_only_smoke_passed", **smoke)
            if smoke["seed7_training_eta_seconds"] > 24 * 3600:
                raise TimeoutError("train-only smoke ETA exceeds the fixed Stage64 seed7 cap")
        if stage63.get("integrity_passed") is not True:
            final = {
                "status": "INTEGRITY_BLOCKED",
                "stage63": stage63,
                "stage64": {},
                "stage65": None,
                "training_commit": _git(repo, "rev-parse", "HEAD"),
                "scientific_negative": False,
                "automatic_next_action": "none",
            }
            (args.output_root / "CAMPAIGN_RESULT.json").write_text(
                json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            _append_ledger(ledger, "campaign_endpoint", status=final["status"])
            return final
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
                if result["status"] != "RAW_ALL_GATES_PASSED":
                    break
            action = next_protocol_action(
                stage63_integrity=True,
                stage63_training_ready=True,
                raw_available_or_supported=True,
                stage64_seed7=stage64_results[7]["status"],
                replication={
                    seed: stage64_results[seed]["status"]
                    for seed in (13, 23)
                    if seed in stage64_results
                },
            )
        if action == "RUN_STAGE65":
            _append_ledger(ledger, "stage65_running")
            stage65_result = run_stage65(_stage_args(args, max_hours=1.0))
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
    parser.add_argument("--coherent-a5-root", type=Path, required=True)
    parser.add_argument("--frozen-teacher-audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, choices=(4, 8), default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with CampaignLock(args.output_root / "CAMPAIGN_LOCK.json"):
            result = run_stage63(_stage_args(args, max_hours=12.0))
    else:
        result = run(args)
    print(json.dumps({"status": result.get("status", result.get("decision"))}))


if __name__ == "__main__":
    main()

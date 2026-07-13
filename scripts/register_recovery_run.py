"""Append one fully specified recovery run to the artifact registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.rstrip()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _selection_criterion(stage: str) -> str:
    if stage == "ssl_pretrain":
        return "validation_loss"
    if stage == "downstream_ttc":
        return "validation_mae"
    raise ValueError(f"Unsupported recovery stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("ssl_pretrain", "downstream_ttc"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pretrain-seed", type=int, required=True)
    parser.add_argument("--downstream-seed", type=int)
    parser.add_argument("--requested-backbone", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--completed-at", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.jsonl"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    metrics_path = args.run_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    best = Path(str(metrics["best_checkpoint"]))
    last = Path(str(metrics["last_checkpoint"]))
    predictions = args.run_dir / "predictions.npz"
    commit = _git(root, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise RuntimeError("Repository commit changed during the recovery matrix")
    status_lines = _git(root, "status", "--porcelain").splitlines()
    unexpected_dirty = [
        line
        for line in status_lines
        if line[3:].replace("\\", "/") != "artifacts/registry.jsonl"
    ]
    if unexpected_dirty and not args.smoke:
        raise RuntimeError(
            "Refusing to register valid_post_fix after non-registry worktree changes: "
            + ", ".join(unexpected_dirty)
        )
    dirty = bool(unexpected_dirty)
    validity = "smoke_only" if args.smoke else "valid_post_fix"
    model_seed = args.downstream_seed if args.downstream_seed is not None else args.pretrain_seed
    record = {
        "schema_version": 1,
        "record_type": "run",
        "run_id": args.run_id,
        "project": "e-jepa-ttc",
        "stage": args.stage,
        "status": validity,
        "run_status": "smoke" if args.smoke else "official_candidate",
        "validity_status": validity,
        "claim_level": "development",
        "created_at": args.completed_at,
        "started_at": args.started_at,
        "completed_at": args.completed_at,
        "git_commit": commit,
        "dirty_worktree": dirty,
        "command": args.command,
        "config_path": "configs/experiment/recovery_full_starter_multiseed.yaml",
        "config_hash": _sha256(root / "configs/experiment/recovery_full_starter_multiseed.yaml"),
        "dataset_path": "datasets/evttc",
        "dataset_manifest_path": "data/manifests/evttc_full_starter_local.yaml",
        "dataset_manifest_hash": _sha256(root / "data/manifests/evttc_full_starter_local.yaml"),
        "feature_schema_version": "voxel_160x90_b5_raw_meta_nav_recovery_v1",
        "input_view": "causal_event_and_navigation",
        "split_protocol": "evttc-full-starter-diagnostic-2026-07-13",
        "split_path": "data/splits/evttc_full_starter_sealed.yaml",
        "split_hash": _sha256(root / "data/splits/evttc_full_starter_sealed.yaml"),
        "test_open_count": 0,
        "pretrain_seed": args.pretrain_seed,
        "downstream_seed": args.downstream_seed,
        "evaluation_seed": model_seed,
        "data_seed": 0,
        "model_seed": model_seed,
        "requested_backbone": args.requested_backbone,
        "actual_backbone": metrics.get("model_name"),
        "checkpoint_path": best.as_posix(),
        "checkpoint_sha256": _sha256(root / best),
        "checkpoint_role": "best",
        "checkpoint_selected_by": _selection_criterion(args.stage),
        "best_checkpoint": best.as_posix(),
        "last_checkpoint": last.as_posix(),
        "selection_criterion": _selection_criterion(args.stage),
        "best_checkpoint_path": best.as_posix(),
        "last_checkpoint_path": last.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "metrics_sha256": _sha256(root / metrics_path),
        "predictions_path": predictions.as_posix() if predictions.exists() else None,
        "predictions_sha256": _sha256(predictions if predictions.exists() else None),
        "hardware": {
            "device": metrics.get("device"),
            "gpu": metrics.get("gpu_name"),
            "torch": metrics.get("torch_version"),
        },
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "artifact_exists": best.exists() and last.exists() and metrics_path.exists(),
        "notes": (
            "Validation-only recovery artifact; CPLA-high was not opened. "
            "dirty_worktree excludes the runner-authorized registry append. "
            "Promotion beyond development requires the frozen diagnostic/final gate."
        ),
    }
    existing_ids: set[str] = set()
    if args.registry.exists():
        for raw_line in args.registry.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                existing_ids.add(str(json.loads(raw_line).get("run_id")))
    if args.run_id in existing_ids:
        print(f"Duplicate registry run_id (already registered): {args.run_id}")
        return 0
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    with args.registry.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

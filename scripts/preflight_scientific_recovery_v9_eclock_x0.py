#!/usr/bin/env python
"""Fail-closed hardware, identity, model and cache preflight for E-Clock X0."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockOuterTrainView,
    CollisionClockTrain8192Cache,
)
from e_jepa_ttc.evaluation.collision_clock_config import (
    assert_arm_execution_authorized,
    validate_matched_base_dyn,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.evaluation.collision_clock_runner import (
    _direct_model,
    _official_checkpoint,
    load_runner_contracts,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--authorized-commit", required=True)
    parser.add_argument(
        "--cache-mode", choices=("auto", "fold_ram", "shard_lru", "direct"), default="auto"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if _git(repo, "rev-parse", "HEAD") != args.authorized_commit:
        raise ValueError("preflight HEAD differs from authorized commit")
    if _git(repo, "branch", "--show-current") != "scientific-recovery-v9-eclock-x0":
        raise ValueError("preflight branch mismatch")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("preflight requires a clean versioned worktree")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            "af66f2c8ca2017059d7765b5f171e1cda866ab07",
            args.authorized_commit,
        ],
        check=True,
    )
    if shutil.disk_usage(args.output_root.resolve().anchor).free < 30 * 1024**3:
        raise OSError("less than 30 GiB free on the campaign output volume")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_properties(0).total_memory < 11 * 1024**3:
        raise RuntimeError("GPU exposes less than 11 GiB VRAM")
    config_root = repo / "configs/experiment/scientific_recovery_v9_eclock"
    base_path = config_root / "x0_base_u.yaml"
    dyn_path = config_root / "x0_dyn_u.yaml"
    base, protocol, reference = load_runner_contracts(repo, base_path)
    dyn, dyn_protocol, dyn_reference = load_runner_contracts(repo, dyn_path)
    if dyn_protocol != protocol or dyn_reference != reference:
        raise ValueError("BASE/DYN protocol/reference contracts differ")
    validate_matched_base_dyn(base, dyn)
    torch.manual_seed(7)
    base_model = _direct_model(base)
    torch.manual_seed(7)
    dyn_model = _direct_model(dyn)
    if sum(p.numel() for p in base_model.parameters()) != 308005:
        raise ValueError("BASE parameter count drifted from 308005")
    if sum(p.numel() for p in dyn_model.parameters()) != 308005:
        raise ValueError("DYN parameter count drifted from 308005")
    if module_topology_sha256(base_model) != module_topology_sha256(dyn_model):
        raise ValueError("BASE/DYN model topology differs")
    if tensor_state_sha256(base_model) != tensor_state_sha256(dyn_model):
        raise ValueError("BASE/DYN matched initialization differs")
    dyn_w, _p, _r = load_runner_contracts(repo, config_root / "x0_dyn_w.yaml")
    try:
        assert_arm_execution_authorized(dyn_w)
    except PermissionError:
        pass
    else:
        raise ValueError("X0-DYN-W unexpectedly became executable")
    for fold in (0, 1, 2):
        _official_checkpoint(fold=fold, reference=reference, source_root=args.reference_root)
    adapter = CollisionClockTrain8192Cache(args.cache_root, protocol, cache_mode=args.cache_mode)
    train, _dev = adapter.outer_views(0)
    selected_mode = adapter.select_mode_for_train_view(train)
    subset_locators = train.locators[:2]
    subset = CollisionClockOuterTrainView(0, subset_locators, adapter._subset_sha(subset_locators))
    adapter.cache_mode = "direct"
    direct = adapter._materialize(subset.locators)
    adapter.cache_mode = "shard_lru"
    lru = adapter._materialize(subset.locators)
    adapter.cache_mode = "fold_ram"
    adapter.stage_view(subset)
    fold_ram = adapter._materialize(subset.locators)
    equality = {
        "direct_equals_shard_lru": all(torch.equal(a, b) for a, b in zip(direct, lru, strict=True)),
        "direct_equals_fold_ram": all(
            torch.equal(a, b) for a, b in zip(direct, fold_ram, strict=True)
        ),
    }
    if not all(equality.values()):
        raise ValueError("cache modes are not bitwise equal")
    adapter.release_staged_view()
    adapter.cache_mode = selected_mode
    decision = sign_artifact(
        {
            "artifact_type": "eclock_x0_cache_engineering_decision_v1",
            "selection_uses_outer_train_only": True,
            "loss_or_mid_used_for_selection": False,
            "outer_dev_opened": False,
            "requested_mode": args.cache_mode,
            "selected_mode": selected_mode,
            "mode_equality": equality,
            "cache_manifest_sha256": protocol["cache_binding"]["file_sha256"],
            "cache_stats": adapter.engineering_stats(),
        }
    )
    summary: dict[str, Any] = sign_artifact(
        {
            "artifact_type": "eclock_x0_preflight_v1",
            "status": "passed",
            "git_commit": args.authorized_commit,
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "disk_free_bytes": shutil.disk_usage(args.output_root.resolve().anchor).free,
            "cache_shards_verified": 32,
            "official_a5_checkpoints_verified": 3,
            "base_dyn_parameter_count": 308005,
            "base_dyn_topology_matched": True,
            "base_dyn_initialization_matched": True,
            "dyn_w_rejected": True,
            "sealed_paths_opened": False,
            "cache_decision_sha256": decision["artifact_sha256"],
            "protocol_sha256": protocol["artifact_sha256"],
            "reference_sha256": reference["artifact_sha256"],
            "config_hashes": {
                "base": compute_file_hash(str(base_path)),
                "dyn": compute_file_hash(str(dyn_path)),
            },
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (args.output_root / "cache_engineering_decision.json", decision),
        (args.output_root / "preflight_summary.json", summary),
    ):
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"preflight": summary, "cache_decision": decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

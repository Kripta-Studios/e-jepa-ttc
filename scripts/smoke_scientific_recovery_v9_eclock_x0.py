#!/usr/bin/env python
"""Run the authorized real outer-train-only X0 smoke and resume parity gates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockOuterTrainSequence,
    CollisionClockOuterTrainView,
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.evaluation.collision_clock_runner import (
    _direct_model,
    _official_checkpoint,
    _prediction_coordinates,
    load_runner_contracts,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.models.collision_clock_ttc import (
    CollisionClockConfig,
    X0A5Replay,
    X0PairDirectPhase,
)
from e_jepa_ttc.training.collision_clock_eap import (
    CollisionClockScientificIdentity,
    CollisionClockTrainingConfig,
    train_collision_clock_updates,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _model(
    config: dict[str, Any], reference: dict[str, Any], source_root: Path, device: torch.device
) -> nn.Module:
    torch.manual_seed(7)
    arm = config["arm_id"]
    if arm in {"X0-A5-REPLAY", "X0-PAIR-U"}:
        checkpoint, _sha = _official_checkpoint(
            fold=0, reference=reference, source_root=source_root
        )
        source = load_causal_scale_replay_checkpoint(checkpoint, device=device)
        if arm == "X0-A5-REPLAY":
            return X0A5Replay(source).to(device)
        values = dict(config["model"])
        values["feature_source"] = config["feature_source"]
        values["motion_feature_mode"] = config["motion_feature_mode"]
        return X0PairDirectPhase(source, CollisionClockConfig(**values)).to(device)
    return _direct_model(config).to(device)


def _identity(
    *,
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    reference: dict[str, Any],
    model: nn.Module,
    train_view: CollisionClockOuterTrainView,
    dev_sha: str,
    update_budget: int,
) -> CollisionClockScientificIdentity:
    return CollisionClockScientificIdentity(
        git_commit_observed=_git(repo, "rev-parse", "HEAD"),
        git_dirty_observed=bool(_git(repo, "status", "--porcelain=v1", "--untracked-files=no")),
        arm_id=config["arm_id"],
        scientific_role=config["scientific_role"],
        reference_family=("official_a5_oof" if config["arm_id"] == "X0-PAIR-U" else None),
        seed=7,
        outer_fold=0,
        motion_feature_mode=config["motion_feature_mode"],
        model_class=model.__class__.__name__,
        model_topology_sha256=module_topology_sha256(model),
        initialization_sha256=tensor_state_sha256(model),
        config_path=str(config_path),
        config_sha256=compute_file_hash(str(config_path)),
        protocol_path=config["protocol_path"],
        protocol_sha256=protocol["artifact_sha256"],
        reference_path=config["reference_path"],
        reference_sha256=reference["artifact_sha256"],
        split_manifest_path=protocol["split_binding"]["path"],
        split_manifest_sha256=protocol["split_binding"]["file_sha256"],
        cache_manifest_path=protocol["cache_binding"]["path"],
        cache_manifest_sha256=protocol["cache_binding"]["file_sha256"],
        ordered_token_identity_sha256=protocol["canonical_hashes"]["token_identity_sha256"],
        target_sha256=protocol["canonical_hashes"]["target_sha256"],
        fold_assignment_sha256=protocol["canonical_hashes"]["fold_assignment_sha256"],
        sample_weight_sha256=protocol["canonical_hashes"]["sample_weight_sha256"],
        train_token_subset_sha256=train_view.ordered_token_identity_sha256,
        dev_token_subset_sha256=dev_sha,
        optimizer_config={"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4},
        scheduler_config={"name": "constant"},
        precision_mode="float32",
        update_budget=update_budget,
        checkpoint_policy="last_update_fixed_budget",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-mode", choices=("direct", "shard_lru", "fold_ram"), required=True)
    parser.add_argument("--fold", type=int, choices=(0,), default=0)
    parser.add_argument("--max-rows", type=int, choices=range(1, 33), default=32)
    parser.add_argument("--execute-authorized-outer-train-smoke", action="store_true")
    args = parser.parse_args()
    if not args.execute_authorized_outer_train_smoke:
        raise PermissionError("real smoke requires explicit outer-train authorization")
    repo = Path(__file__).resolve().parents[1]
    if args.output_root.exists():
        raise FileExistsError("smoke output root already exists")
    args.output_root.mkdir(parents=True)
    config_root = repo / "configs/experiment/scientific_recovery_v9_eclock"
    config_paths = {
        arm: config_root / name
        for arm, name in {
            "X0-A5-REPLAY": "x0_a5_replay.yaml",
            "X0-BASE-U": "x0_base_u.yaml",
            "X0-DYN-U": "x0_dyn_u.yaml",
            "X0-PAIR-U": "x0_pair_u.yaml",
        }.items()
    }
    first_config, protocol, reference = load_runner_contracts(repo, config_paths["X0-BASE-U"])
    del first_config
    adapter = CollisionClockTrain8192Cache(
        args.cache_root,
        protocol,
        cache_mode=args.cache_mode,
        canonical_supervision=load_canonical_supervision(reference, args.reference_root),
    )
    train_view_full, dev_view = adapter.outer_views(0)
    locators = train_view_full.locators[: args.max_rows]
    train_view = CollisionClockOuterTrainView(0, locators, adapter._subset_sha(locators))
    adapter.stage_view(train_view)
    batch = CollisionClockOuterTrainSequence(adapter, train_view, batch_size=args.max_rows)[0]
    outputs: dict[str, Any] = {}
    device = torch.device(args.device)
    for arm, config_path in config_paths.items():
        config, arm_protocol, arm_reference = load_runner_contracts(repo, config_path)
        if arm_protocol["artifact_sha256"] != protocol["artifact_sha256"]:
            raise ValueError("smoke arm protocol mismatch")
        arm_root = args.output_root / arm
        arm_root.mkdir()
        if arm == "X0-A5-REPLAY":
            model = _model(config, arm_reference, args.reference_root, device)
            model.eval()
            with torch.no_grad():
                result = model(batch.inputs.to(device), batch.delta_t_s.to(device))
            coordinates = _prediction_coordinates(
                result,
                delta_t_s=float(protocol["metric"]["metric_delta_t_s"]),
            )
            finite = all(
                bool(torch.isfinite(torch.from_numpy(value)).all()) for value in coordinates
            )
            if not finite:
                raise FloatingPointError("A5 replay smoke produced non-finite coordinates")
            outputs[arm] = {"inference_rows": len(batch.sample_tokens), "finite": True}
            continue
        updates = 2 if arm == "X0-PAIR-U" else 10
        split = 1 if arm == "X0-PAIR-U" else 5
        config_train = CollisionClockTrainingConfig(arm_id=arm, update_budget=updates)
        continuous = _model(config, arm_reference, args.reference_root, device)
        continuous_identity = _identity(
            repo=repo,
            config_path=config_path,
            config=config,
            protocol=protocol,
            reference=reference,
            model=continuous,
            train_view=train_view,
            dev_sha=dev_view.ordered_token_identity_sha256,
            update_budget=updates,
        )
        continuous_result = train_collision_clock_updates(
            continuous,
            [batch],
            config=config_train,
            scientific_identity=continuous_identity,
            checkpoint_path=arm_root / "continuous" / "resume_latest.pt",
            progress_path=arm_root / "continuous" / "progress.jsonl",
            milestone_updates=(updates,),
        )
        interrupted = _model(config, arm_reference, args.reference_root, device)
        interrupted_identity = _identity(
            repo=repo,
            config_path=config_path,
            config=config,
            protocol=protocol,
            reference=reference,
            model=interrupted,
            train_view=train_view,
            dev_sha=dev_view.ordered_token_identity_sha256,
            update_budget=updates,
        )
        resume_path = arm_root / "resumed" / "resume_latest.pt"
        progress_path = arm_root / "resumed" / "progress.jsonl"
        train_collision_clock_updates(
            interrupted,
            [batch],
            config=config_train,
            scientific_identity=interrupted_identity,
            checkpoint_path=resume_path,
            progress_path=progress_path,
            stop_after_updates=split,
            milestone_updates=(updates,),
        )
        resumed = _model(config, arm_reference, args.reference_root, device)
        resumed_identity = _identity(
            repo=repo,
            config_path=config_path,
            config=config,
            protocol=protocol,
            reference=reference,
            model=resumed,
            train_view=train_view,
            dev_sha=dev_view.ordered_token_identity_sha256,
            update_budget=updates,
        )
        resumed_result = train_collision_clock_updates(
            resumed,
            [batch],
            config=config_train,
            scientific_identity=resumed_identity,
            checkpoint_path=resume_path,
            progress_path=progress_path,
            resume=True,
            milestone_updates=(updates,),
        )
        if (
            tensor_state_sha256(continuous) != tensor_state_sha256(resumed)
            or continuous_result.losses != resumed_result.losses
            or continuous_result.batch_schedule_sha256 != resumed_result.batch_schedule_sha256
        ):
            raise ValueError(f"{arm} continuous/resume smoke mismatch")
        resumed.eval()
        with torch.no_grad():
            final = resumed(batch.inputs.to(device), batch.delta_t_s.to(device))
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (
                final.benchmark_phase_mean,
                final.inverse_ttc_mean,
                final.predicted_ttc_raw,
            )
        )
        if not finite:
            raise FloatingPointError(f"{arm} final smoke coordinates are non-finite")
        outputs[arm] = {
            "updates": updates,
            "resume_split": [split, updates - split],
            "resume_exact": True,
            "finite": True,
            "continuous": {
                "completed_updates": continuous_result.completed_updates,
                "checkpoint_path": str(continuous_result.checkpoint_path),
                "batch_schedule_sha256": continuous_result.batch_schedule_sha256,
            },
            "resumed": {
                "completed_updates": resumed_result.completed_updates,
                "checkpoint_path": str(resumed_result.checkpoint_path),
                "batch_schedule_sha256": resumed_result.batch_schedule_sha256,
            },
        }
    adapter.release_staged_view()
    summary = sign_artifact(
        {
            "artifact_type": "eclock_x0_real_outer_train_smoke_v1",
            "evidence_class": "smoke",
            "scientific_result": False,
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "seed": 7,
            "outer_fold": 0,
            "max_rows": args.max_rows,
            "cache_mode": args.cache_mode,
            "cache_engineering": adapter.engineering_stats(),
            "outer_dev_opened": False,
            "outer_dev_evaluated": False,
            "arms": outputs,
        }
    )
    (args.output_root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

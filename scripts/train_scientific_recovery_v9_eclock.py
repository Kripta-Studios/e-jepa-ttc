#!/usr/bin/env python
"""Fail-closed CLI for E-Clock X0 dry-runs, smokes and future OOF jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_config import (
    assert_arm_execution_authorized,
    load_x0_config,
)
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.models.collision_clock_features import (
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
)
from e_jepa_ttc.models.collision_clock_ttc import CollisionClockConfig, X0HeightBypassDirectPhase

PARENT_COMMIT = "718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _model(config: dict[str, Any]) -> X0HeightBypassDirectPhase:
    model_values = dict(config["model"])
    model_values["feature_source"] = config["feature_source"]
    model_values["motion_feature_mode"] = config["motion_feature_mode"]
    clock_config = CollisionClockConfig(**model_values)
    encoder_config = HeightBypassEncoderConfig(
        in_channels=clock_config.in_channels,
        hidden_dim=clock_config.encoder_hidden_dim,
        token_dim=clock_config.encoder_token_dim,
        residual_depth=clock_config.residual_depth,
        dropout=clock_config.dropout,
    )
    return X0HeightBypassDirectPhase(HeightBypassEndpointEncoder(encoder_config), clock_config)


def _manifest(
    repo: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    evidence_class: str,
    model: X0HeightBypassDirectPhase | None,
) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=no")
    payload: dict[str, Any] = {
        "artifact_type": "eclock_x0_run_manifest_v1",
        "arm_id": config["arm_id"],
        "evidence_class": evidence_class,
        "scientific_result": False,
        "git_commit_observed": commit,
        "git_dirty_observed": bool(status),
        "parent_git_commit": PARENT_COMMIT,
        "config_file_sha256": compute_file_hash(str(config_path)),
        "loss_reduction": config["loss_reduction"],
        "upstream_roi_is_box_conditioned": True,
        "explicit_foreground_height_interface_bypassed": config[
            "explicit_foreground_height_interface_bypassed"
        ],
        "execution_authorized": config["execution_authorized"],
        "sealed_evaluation": "closed",
    }
    if model is not None:
        payload.update(
            {
                "initial_model_state_sha256": tensor_state_sha256(model),
                "learnable_topology_sha256": module_topology_sha256(model),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "feature_vector_dimension": model.input_dim,
                "feature_schema": model.feature_schema.manifest(),
            }
        )
    return sign_artifact(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "synthetic-smoke", "oof"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--execute-authorized-oof", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    schema = repo / "schemas/scientific_recovery_v9_eclock_config_v1.schema.json"
    config_path = args.config.resolve()
    config = load_x0_config(config_path, schema_path=schema)
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text(
            encoding="utf-8"
        )
    )
    if not verify_artifact_hash(protocol):
        raise ValueError("protocol signature mismatch")
    if args.seed != 7 or config["seed"] != 7:
        raise ValueError("this phase authorizes seed 7 only")
    torch.manual_seed(args.seed)
    if args.mode == "dry-run":
        model = (
            _model(config)
            if config["feature_source"] == "raw_endpoint"
            and config["execution_authorized"] is True
            else None
        )
        print(
            json.dumps(_manifest(repo, config_path, config, evidence_class="dry_run", model=model))
        )
        return 0
    assert_arm_execution_authorized(config)
    if config["arm_id"] not in {"X0-BASE-U", "X0-DYN-U"}:
        raise ValueError("local smoke supports BASE/DYN; PAIR/A5 require verified fold checkpoints")
    model = _model(config)
    if args.mode == "synthetic-smoke":
        output = model(torch.randn(2, 3, 12, 16, 16), torch.full((2, 2), 0.05))
        manifest = _manifest(repo, config_path, config, evidence_class="smoke", model=model)
        manifest["smoke_rows"] = 2
        manifest["finite_predictions"] = bool(torch.isfinite(output.ttc_mean_seconds).all())
        manifest = sign_artifact(manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if not args.execute_authorized_oof:
        raise PermissionError("OOF requires explicit future --execute-authorized-oof authorization")
    if args.fold is None or args.output is None:
        raise ValueError("OOF requires --fold and --output")
    raise RuntimeError(
        "OOF launch contract reached; this implementation phase intentionally did not resolve "
        "or open the production cache. Supply the separately authorized cache adapter."
    )


if __name__ == "__main__":
    raise SystemExit(main())

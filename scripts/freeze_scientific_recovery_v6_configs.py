#!/usr/bin/env python
"""Freeze fold-local A5-causal and V6.1 radius-2 run configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402

A8_CONFIGS = tuple(
    ROOT
    / "configs"
    / "experiment"
    / "scientific_recovery_v5_fold_chain"
    / f"a8_0_dual_transport_grouped_fold{fold}_seed7.yaml"
    for fold in range(3)
)
A5_MODEL = "configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_causal.yaml"
V6_MODEL = "configs/model/e_jepa_causal_scale_event_v12_dual_transport_r2_t002_causal.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _a5_config(base: dict[str, Any], fold: int) -> dict[str, Any]:
    value = copy.deepcopy(base)
    experiment = value["experiment"]
    experiment.update(
        {
            "name": f"scientific_recovery_v6_a5_causal_grouped_fold{fold}_seed7",
            "protocol_version": "scientific_recovery_v6_a5_causal_grouped_dev_v1",
            "purpose": "fold_local_test_of_unfrozen_a5_under_causal_smoothing",
            "single_scientific_difference": "a8_dual_frozen_to_a5_shared_unfrozen_encoder",
            "grouped_dev_role": "diagnostic_geometry_unconstrained_comparator",
        }
    )
    value["model_config"] = A5_MODEL
    training = value["training"]
    training.update(
        {
            "foreground_warmup_epochs": 3,
            "initialization_checkpoint": None,
            "initialization_checkpoint_sha256": None,
            "initialization_mode": "none",
            "freeze_encoder": False,
        }
    )
    decision = value["decision_contract"]
    decision["expected_parameter_count"] = 424274
    decision["representation_change"]["type"] = (
        "a4_endpoint_dino_plus_event_native_local_cross_time_transport"
    )
    decision["representation_change"]["transport_radius"] = 1
    decision["representation_change"]["transport_candidates_per_position"] = 9
    decision.pop("dual_stream_contract", None)
    decision.pop("a8_0_gate", None)
    decision["primary_development_comparator"] = "A8.0_same_fold_seed_budget"
    decision["geometry_preservation_required"] = False
    decision["diagnostic_only_until_geometry_is_reassessed"] = True
    decision["grouped_dev_role"] = "diagnostic_geometry_unconstrained_comparator"
    return value


def _v6_config(
    base: dict[str, Any], fold: int, d0_path: Path, d0: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(base)
    experiment = value["experiment"]
    experiment.update(
        {
            "name": f"scientific_recovery_v6_1_dual_transport_r2_fold{fold}_seed7",
            "protocol_version": "scientific_recovery_v6_1_radius2_grouped_dev_v1",
            "purpose": "test_D0_motion_scale_hypothesis_with_larger_local_support",
            "single_scientific_difference": "A8_transport_radius_1_to_2",
            "grouped_dev_role": "candidate",
        }
    )
    value["model_config"] = V6_MODEL
    decision = value["decision_contract"]
    change = decision["representation_change"]
    change["type"] = "a4_frozen_geometry_plus_trainable_transport_encoder_radius_expansion"
    change["transport_radius"] = 2
    change["transport_candidates_per_position"] = 25
    decision["dual_stream_contract"]["transport_radius"] = 2
    decision["scale_changes_transport_radius"] = True
    decision["primary_development_comparator"] = "A8.0_radius1_same_fold_seed_budget"
    decision.pop("preflight_contract", None)
    decision.pop("a8_0_gate", None)
    try:
        d0_reference = d0_path.relative_to(ROOT).as_posix()
    except ValueError:
        d0_reference = d0_path.resolve().as_posix()
    decision["v6_d0_contract"] = {
        "artifact": d0_reference,
        "file_sha256": _sha256(d0_path),
        "artifact_sha256": d0["artifact_sha256"],
        "selected_family": "motion_scale",
        "selected_branch": "V6.1_MULTI_SCALE_TRANSPORT",
        "source_radius": 1,
        "candidate_radius": 2,
        "single_scientific_difference": "transport_radius_1_to_2",
        "public_validation_opened": False,
        "private_test_opened": False,
    }
    decision["v6_1_gate"] = {
        "comparator": "A8.0_radius1_same_fold_seed_budget",
        "must_improve_A8_outer_dev_MiD": True,
        "first_stage_sequence_macro_MiD_max": 175.0,
        "geometry_exact_parent_required": True,
        "model_prefix_causal_required": True,
        "public_validation_remains_closed": True,
        "private_test_remains_closed": True,
    }
    return value


def build_configs(
    bases: list[dict[str, Any]], d0_path: Path, d0: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build the six frozen configs without reading targets or predictions."""

    if len(bases) != 3:
        raise ValueError("three A8 fold configs are required")
    if not verify_artifact_hash(d0):
        raise ValueError("V6-D0 artifact signature is invalid")
    decision = d0.get("decision", {})
    if decision.get("selected_branch") != "V6.1_MULTI_SCALE_TRANSPORT":
        raise ValueError("V6-D0 did not select the radius-expansion branch")
    return {
        **{f"a5_causal_fold{fold}": _a5_config(base, fold) for fold, base in enumerate(bases)},
        **{
            f"v6_1_r2_fold{fold}": _v6_config(base, fold, d0_path, d0)
            for fold, base in enumerate(bases)
        },
    }


def freeze(*, d0_path: Path, output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("V6 config freeze requires a tracked-clean worktree")
    d0 = json.loads(d0_path.read_text(encoding="utf-8"))
    bases = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in A8_CONFIGS]
    configs = build_configs(bases, d0_path, d0)
    written: dict[str, Any] = {}
    for name, config in configs.items():
        path = output_dir / f"{name}.yaml"
        _atomic_text(path, yaml.safe_dump(config, sort_keys=False))
        written[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v6_fold_run_configs_v1",
        "status": "frozen_before_v6_training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "d0": {
            "path": d0_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(d0_path),
            "artifact_sha256": d0["artifact_sha256"],
        },
        "source_a8_configs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for path in A8_CONFIGS
        ],
        "model_configs": {
            "a5_causal": {"path": A5_MODEL, "sha256": _sha256(ROOT / A5_MODEL)},
            "v6_1_radius2": {"path": V6_MODEL, "sha256": _sha256(ROOT / V6_MODEL)},
        },
        "configs": written,
        "contracts": {
            "folds_unchanged_from_v5": True,
            "a5_causal_is_diagnostic_geometry_unconstrained": True,
            "v6_1_single_change_transport_radius_1_to_2": True,
            "public_validation_opened": False,
            "private_test_opened": False,
        },
    }
    sign_artifact(result)
    _atomic_text(manifest_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = freeze(
            d0_path=args.d0.resolve(strict=True),
            output_dir=args.output_dir.resolve(),
            manifest_path=args.manifest.resolve(),
        )
    except Exception as error:
        parser.exit(2, f"V6 config freeze failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": result["status"], "artifact_sha256": result["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

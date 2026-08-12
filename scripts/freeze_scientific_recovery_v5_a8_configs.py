"""Freeze A6 comparator and A8.0 configs on the signed V5 grouped-dev folds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402

PROTOCOL_PATH = ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
A6_SOURCE = (
    ROOT
    / "artifacts/scientific_recovery_master_v3/configs/causal_a6/"
    "a6_s1_causal_left_seed7.yaml"
)
A8_SOURCE = (
    ROOT
    / "artifacts/scientific_recovery_master_v3/configs/a6_a7_s1/"
    "a7_s1_seed7.yaml"
)
OUTPUT_DIR = ROOT / "configs/experiment/scientific_recovery_v5"
MANIFEST_PATH = ROOT / "configs/protocol/scientific_recovery_v5_a8_0_configs.json"
PARENT_CHECKPOINT = "artifacts/runs/scientific_recovery_a4_causal_left_seed7/model_best.pt"
PARENT_CHECKPOINT_SHA256 = (
    "ac3a0ff7ec4fba3fcde31c95f18ba33e7905a0fb0c9ea70bb3541ffb12515431"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _protocol_reference(protocol: dict[str, Any], fold: int) -> dict[str, Any]:
    return {
        "path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(PROTOCOL_PATH),
        "artifact_sha256": protocol["artifact_sha256"],
        "fold": fold,
    }


def _fold_config(
    source: dict[str, Any],
    protocol: dict[str, Any],
    fold: dict[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    index = int(fold["fold"])
    is_a8 = arm == "a8_0"
    experiment = config["experiment"]
    experiment.update(
        {
            "name": f"scientific_recovery_v5_{arm}_grouped_fold{index}_seed7",
            "protocol_version": f"scientific_recovery_v5_{arm}_grouped_dev_v1",
            "evidence_scope": "public_train_only_grouped_development",
            "grouped_dev_role": "candidate" if is_a8 else "causal_a6_comparator",
            "parent_stochasticity": "fixed_A4_causal_seed7",
            "transport_training_seed": 7,
        }
    )
    if is_a8:
        experiment["single_scientific_difference"] = (
            "A6_adapter_to_separate_trainable_transport_encoder_copy"
        )
        config["model_config"] = (
            "configs/model/"
            "e_jepa_causal_scale_event_v11_dual_transport_r1_t002_causal.yaml"
        )
    else:
        config["model_config"] = (
            "configs/model/"
            "e_jepa_causal_scale_event_v10_transport_adapter_r1_t002_causal.yaml"
        )

    data = config["data"]
    for key in (
        "validation_cache_manifest",
        "validation_cache_manifest_sha256",
        "validation_cache_artifact_sha256",
        "validation_sequence_ids",
        "expected_validation_rows",
        "expected_train_rows",
    ):
        data.pop(key, None)
    data.update(
        {
            "train_sequence_ids": fold["train_sequence_ids"],
            "dev_sequence_ids": fold["dev_sequence_ids"],
            "opened_splits": ["train"],
            "expected_source_train_rows": protocol["sample_count"],
            "development_protocol": _protocol_reference(protocol, index),
        }
    )
    training = config["training"]
    training.update(
        {
            "seed": 7,
            "num_workers": 0,
            "initialization_checkpoint": PARENT_CHECKPOINT,
            "initialization_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "initialization_mode": "shape_compatible",
            "freeze_encoder": True,
        }
    )
    decision = config["decision_contract"]
    decision["checkpoint_selection"] = "dev_sequence_macro_MiD_then_failure_rate"
    decision["grouped_dev_protocol"] = {
        "artifact_sha256": protocol["artifact_sha256"],
        "fold": index,
        "train_rows": fold["train_rows"],
        "dev_rows": fold["dev_rows"],
        "public_validation_used_for_selection": False,
        "private_test_opened": False,
    }
    decision["a8_0_gate"] = {
        "comparator": "A6_causal_same_fold_seed_budget",
        "first_stage_sequence_macro_MiD_max": 175.0,
        "strong_sequence_macro_MiD_max": 160.0,
        "geometry_exact_parent_required": True,
        "model_prefix_causal_required": True,
        "aspirational_144_9_is_not_a_clean_gate": True,
    }
    decision["public_validation_used_for_selection"] = False
    decision["private_test_remains_closed"] = True
    for key in (
        "baseline_sequence_macro_MiD",
        "baseline_failure_rate_pct",
        "parent_a4_sequence_macro_MiD",
        "parent_a4_failure_rate_pct",
        "transport_gate",
    ):
        decision.pop(key, None)
    contract_name = "dual_stream_contract" if is_a8 else "adapter_contract"
    contract = decision[contract_name]
    contract.update(
        {
            "initialization_checkpoint": PARENT_CHECKPOINT,
            "initialization_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "train_rows": fold["train_rows"],
            "dev_rows": fold["dev_rows"],
        }
    )
    contract.pop("validation_rows", None)
    return config


def build_configs(
    protocol: dict[str, Any],
    a6_source: dict[str, Any],
    a8_source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build all six configs without inspecting targets or model performance."""

    if not verify_artifact_hash(protocol):
        raise ValueError("grouped-development protocol signature is invalid")
    if protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("grouped-development protocol is not frozen")
    checks = protocol.get("checks", {})
    if checks.get("public_validation_used_for_selection") is not False:
        raise ValueError("public validation may not be used for A8 config freezing")
    if checks.get("private_test_opened") is not False:
        raise ValueError("private/test must remain closed")
    outputs: dict[str, dict[str, Any]] = {}
    for fold in protocol["folds"]:
        index = int(fold["fold"])
        outputs[f"a6_grouped_fold{index}_seed7.yaml"] = _fold_config(
            a6_source, protocol, fold, arm="a6"
        )
        outputs[f"a8_0_dual_transport_grouped_fold{index}_seed7.yaml"] = (
            _fold_config(a8_source, protocol, fold, arm="a8_0")
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--a6-source", type=Path, default=A6_SOURCE)
    parser.add_argument("--a8-source", type=Path, default=A8_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outputs = build_configs(
        protocol,
        _read_yaml(args.a6_source.resolve(strict=True)),
        _read_yaml(args.a8_source.resolve(strict=True)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, config in outputs.items():
        path = args.output_dir / name
        path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
        )
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "arm": "a8_0" if name.startswith("a8_0") else "a6_comparator",
                "fold": int(name.split("fold", 1)[1].split("_", 1)[0]),
                "seed": 7,
            }
        )
    manifest = {
        "artifact_type": "scientific_recovery_v5_a8_0_frozen_configs_v1",
        "status": "frozen_before_grouped_dev_training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "file_sha256": _sha256(protocol_path),
            "artifact_sha256": protocol["artifact_sha256"],
        },
        "sources": {
            "a6": {"path": str(args.a6_source), "sha256": _sha256(args.a6_source)},
            "a8": {"path": str(args.a8_source), "sha256": _sha256(args.a8_source)},
        },
        "configs": sorted(entries, key=lambda row: (row["arm"], row["fold"])),
        "seed_policy": {
            "parent_checkpoint": PARENT_CHECKPOINT,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_seed": 7,
            "transport_training_seed": 7,
            "folds_are_replications": True,
        },
        "contracts": {
            "train_only_grouped_dev": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "event_only": True,
            "causal_left": True,
            "num_workers": 0,
            "cuda_runs_concurrent": False,
        },
    }
    sign_artifact(manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"manifest": str(args.manifest), "configs": len(entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

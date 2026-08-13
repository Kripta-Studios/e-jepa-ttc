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
    ROOT / "artifacts/scientific_recovery_master_v3/configs/causal_a6/a6_s1_causal_left_seed7.yaml"
)
A8_SOURCE = ROOT / "artifacts/scientific_recovery_master_v3/configs/a6_a7_s1/a7_s1_seed7.yaml"
OUTPUT_DIR = ROOT / "configs/experiment/scientific_recovery_v5_fold_chain"
MANIFEST_PATH = ROOT / "configs/protocol/scientific_recovery_v5_fold_downstream_configs.json"
PARENT_RUN_ROOT = ROOT / "artifacts/runs"


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
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
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
    parent_checkpoint: str,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    index = int(fold["fold"])
    is_a8 = arm == "a8_0"
    experiment = config["experiment"]
    experiment.update(
        {
            "name": f"scientific_recovery_v5_{arm}_fold_chain_fold{index}_seed7",
            "protocol_version": f"scientific_recovery_v5_{arm}_grouped_dev_v1",
            "evidence_scope": "public_train_only_grouped_development",
            "grouped_dev_role": "candidate" if is_a8 else "causal_a6_comparator",
            "parent_stochasticity": "fold_specific_A4_seed7",
            "transport_training_seed": 7,
        }
    )
    if is_a8:
        experiment["single_scientific_difference"] = (
            "A6_adapter_to_separate_trainable_transport_encoder_copy"
        )
        config["model_config"] = (
            "configs/model/e_jepa_causal_scale_event_v11_dual_transport_r1_t002_causal.yaml"
        )
    else:
        config["model_config"] = (
            "configs/model/e_jepa_causal_scale_event_v10_transport_adapter_r1_t002_causal.yaml"
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
            "initialization_checkpoint": parent_checkpoint,
            "initialization_checkpoint_sha256": parent_checkpoint_sha256,
            "initialization_mode": "shape_compatible",
            "freeze_encoder": True,
        }
    )
    decision = config["decision_contract"]
    decision["checkpoint_selection"] = "dev_sequence_macro_MiD_then_failure_rate"
    decision["require_finite_metrics_for_all_dev_sequences"] = True
    decision["grouped_dev_protocol"] = {
        "artifact_sha256": protocol["artifact_sha256"],
        "fold": index,
        "train_rows": fold["train_rows"],
        "dev_rows": fold["dev_rows"],
        "public_validation_used_for_selection": False,
        "private_test_opened": False,
    }
    if is_a8:
        decision["primary_development_comparator"] = "A6_causal_same_fold_seed_budget"
        decision["a8_0_gate"] = {
            "comparator": "A6_causal_same_fold_seed_budget",
            "first_stage_sequence_macro_MiD_max": 175.0,
            "strong_sequence_macro_MiD_max": 160.0,
            "geometry_exact_parent_required": True,
            "model_prefix_causal_required": True,
            "aspirational_144_9_is_not_a_clean_gate": True,
        }
    else:
        decision["grouped_dev_role"] = "causal_A6_comparator_only"
    decision["public_validation_used_for_selection"] = False
    decision["private_test_remains_closed"] = True
    for key in (
        "baseline_sequence_macro_MiD",
        "baseline_failure_rate_pct",
        "parent_a4_sequence_macro_MiD",
        "parent_a4_failure_rate_pct",
        "transport_gate",
        "primary_baseline",
        "require_finite_metrics_for_all_validation_sequences",
        "frozen_validation_rows",
        "lambda_cv_must_not_change_2048_causal_screen",
        "transport_radius_selected_before_a5_validation",
        "transport_temperature_selected_before_a5_validation",
        "scale_transport_claim_requires_A4_control_at_same_lambda",
    ):
        decision.pop(key, None)
    contract_name = "dual_stream_contract" if is_a8 else "adapter_contract"
    contract = decision[contract_name]
    contract.update(
        {
            "initialization_checkpoint": parent_checkpoint,
            "initialization_checkpoint_sha256": parent_checkpoint_sha256,
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
    parent_refs: dict[int, dict[str, str]],
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
        parent = parent_refs.get(index)
        if not isinstance(parent, dict):
            raise ValueError(f"fold {index} lacks a fold-specific A4 parent")
        outputs[f"a6_grouped_fold{index}_seed7.yaml"] = _fold_config(
            a6_source,
            protocol,
            fold,
            arm="a6",
            parent_checkpoint=parent["checkpoint"],
            parent_checkpoint_sha256=parent["checkpoint_sha256"],
        )
        outputs[f"a8_0_dual_transport_grouped_fold{index}_seed7.yaml"] = _fold_config(
            a8_source,
            protocol,
            fold,
            arm="a8_0",
            parent_checkpoint=parent["checkpoint"],
            parent_checkpoint_sha256=parent["checkpoint_sha256"],
        )
    return outputs


def load_parent_refs(protocol: dict[str, Any], parent_run_root: Path) -> dict[int, dict[str, str]]:
    """Load all three signed parent contracts or fail without partial output."""

    refs: dict[int, dict[str, str]] = {}
    for fold in protocol["folds"]:
        index = int(fold["fold"])
        run = parent_run_root / f"scientific_recovery_v5_a4_parent_grouped_fold{index}_seed7"
        contract_path = run / "parent_contract.json"
        checkpoint_path = run / "model_best.pt"
        summary_path = run / "summary.json"
        if (
            not contract_path.is_file()
            or not checkpoint_path.is_file()
            or not summary_path.is_file()
        ):
            raise FileNotFoundError(f"fold {index} parent run is incomplete: {run}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict) or not verify_artifact_hash(contract):
            raise ValueError(f"fold {index} parent contract signature is invalid")
        expected = {
            "artifact_type": "scientific_recovery_v5_fold_parent_v1",
            "status": "completed_fold_specific_parent",
            "fold": index,
            "train_sequence_ids": fold["train_sequence_ids"],
            "dev_sequence_ids": fold["dev_sequence_ids"],
            "train_token_sha256": fold["train_sample_tokens_sha256"],
            "dev_token_sha256": fold["dev_sample_tokens_sha256"],
            "grouped_protocol_artifact_sha256": protocol["artifact_sha256"],
            "checkpoint_sha256": _sha256(checkpoint_path),
            "public_validation_opened": False,
            "private_test_opened": False,
        }
        mismatches = {
            key: {"expected": value, "observed": contract.get(key)}
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise ValueError(f"fold {index} parent contract differs: {mismatches}")
        refs[index] = {
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "parent_contract": contract_path.relative_to(ROOT).as_posix(),
            "parent_contract_sha256": _sha256(contract_path),
            "parent_contract_artifact_sha256": contract["artifact_sha256"],
        }
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--a6-source", type=Path, default=A6_SOURCE)
    parser.add_argument("--a8-source", type=Path, default=A8_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--parent-run-root", type=Path, default=PARENT_RUN_ROOT)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent_refs = load_parent_refs(protocol, args.parent_run_root.resolve())
    outputs = build_configs(
        protocol,
        _read_yaml(args.a6_source.resolve(strict=True)),
        _read_yaml(args.a8_source.resolve(strict=True)),
        parent_refs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, config in outputs.items():
        path = args.output_dir / name
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
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
        "artifact_type": "scientific_recovery_v5_fold_downstream_frozen_configs_v1",
        "status": "frozen_after_parents_before_a6_a8_training",
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
        "parents": parent_refs,
        "seed_policy": {
            "parent_seed": 7,
            "transport_training_seed": 7,
            "folds_measure_sequence_population_change_not_multiseed": True,
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

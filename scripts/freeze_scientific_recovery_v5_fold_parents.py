"""Freeze three fold-local A4 parents before any clean fold-chain training."""

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
A4_SOURCE = (
    ROOT
    / "artifacts/scientific_recovery_master_v3/configs/causal_a6/"
    "a4_s1_lambda8_causal_left_seed7.yaml"
)
OUTPUT_DIR = ROOT / "configs/experiment/scientific_recovery_v5_fold_chain"
MANIFEST_PATH = ROOT / "configs/protocol/scientific_recovery_v5_fold_parent_configs.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def build_parent_configs(
    protocol: dict[str, Any], source: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Build fold-local parents without reading targets or prior fold performance."""

    if not verify_artifact_hash(protocol):
        raise ValueError("grouped-development protocol signature is invalid")
    if protocol.get("artifact_type") != "scientific_recovery_v5_train_only_grouped_dev_v1":
        raise ValueError("grouped-development protocol type is incompatible")
    if protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("grouped-development protocol is not frozen")
    checks = protocol.get("checks", {})
    if checks.get("public_validation_used_for_selection") is not False:
        raise ValueError("public validation may not be used for fold-parent selection")
    if checks.get("private_test_opened") is not False:
        raise ValueError("private/test must remain closed")

    configs: dict[int, dict[str, Any]] = {}
    protocol_reference = {
        "path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(PROTOCOL_PATH),
        "artifact_sha256": protocol["artifact_sha256"],
    }
    for frozen_fold in protocol["folds"]:
        fold = int(frozen_fold["fold"])
        config = copy.deepcopy(source)
        config["experiment"].update(
            {
                "name": f"scientific_recovery_v5_a4_parent_grouped_fold{fold}_seed7",
                "protocol_version": "scientific_recovery_v5_fold_specific_a4_parent_v1",
                "evidence_scope": "public_train_only_grouped_development",
                "purpose": "fold_local_geometry_parent_without_outer_dev_gradients",
                "grouped_dev_role": "fold_specific_a4_parent",
            }
        )
        config["experiment"].pop("parent_arm", None)
        config["experiment"].pop("arm_role", None)
        config["provenance"]["v5_fold_parent_interpretation"] = {
            "weights_inherited_from_historical_a4": False,
            "architecture_and_training_recipe_inherited": True,
            "historical_lambda_recipe_was_fixed_before_fold_parent_training": True,
            "historical_lambda_recipe_used_the_nine_sequence_train_universe": True,
            "outer_dev_is_development_not_test": True,
        }
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
                "opened_splits": ["train"],
                "train_sequence_ids": frozen_fold["train_sequence_ids"],
                "dev_sequence_ids": frozen_fold["dev_sequence_ids"],
                "expected_source_train_rows": protocol["sample_count"],
                "development_protocol": {**protocol_reference, "fold": fold},
            }
        )
        training = config["training"]
        training.update(
            {
                "seed": 7,
                "num_workers": 0,
                "initialization_mode": "none",
                "initialization_checkpoint": None,
                "initialization_checkpoint_sha256": None,
                "freeze_encoder": False,
            }
        )
        decision = config["decision_contract"]
        for key in (
            "primary_baseline",
            "parent_a4_sequence_macro_MiD",
            "parent_a4_failure_rate_pct",
            "parent_a4_log_ratio_pearson",
            "external_reference",
            "garl_sequence_macro_MiD",
            "garl_failure_rate_pct",
            "train_scaling",
            "lambda_selection",
            "require_finite_metrics_for_all_validation_sequences",
        ):
            decision.pop(key, None)
        decision.update(
            {
                "checkpoint_selection": "dev_sequence_macro_MiD_then_failure_rate",
                "require_finite_metrics_for_all_dev_sequences": True,
                "outer_dev_used_for_checkpoint_selection": True,
                "outer_dev_is_not_test": True,
                "parent_weights_trained_on_outer_dev": False,
                "grouped_fold": fold,
                "grouped_protocol_artifact_sha256": protocol["artifact_sha256"],
                "public_validation_used_for_selection": False,
                "private_test_remains_closed": True,
            }
        )
        configs[fold] = config
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--source", type=Path, default=A4_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve(strict=True)
    source_path = args.source.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configs = build_parent_configs(protocol, _read_mapping(source_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for fold, config in sorted(configs.items()):
        path = args.output_dir / f"a4_parent_grouped_fold{fold}_seed7.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
        )
        entries.append(
            {
                "fold": fold,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "training_seed": 7,
                "train_rows": protocol["folds"][fold]["train_rows"],
                "dev_rows": protocol["folds"][fold]["dev_rows"],
                "train_sample_tokens_sha256": protocol["folds"][fold][
                    "train_sample_tokens_sha256"
                ],
                "dev_sample_tokens_sha256": protocol["folds"][fold][
                    "dev_sample_tokens_sha256"
                ],
            }
        )
    manifest: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v5_fold_parent_frozen_configs_v1",
        "status": "frozen_before_clean_fold_parent_training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(protocol_path),
            "artifact_sha256": protocol["artifact_sha256"],
            "split_seed": protocol["split_seed"],
        },
        "cache": protocol["sources"]["cache_manifest"],
        "source_recipe": {
            "path": source_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(source_path),
        },
        "configs": entries,
        "contracts": {
            "folds_unchanged_from_original_freeze": True,
            "one_parent_per_fold": True,
            "parent_initialization_from_scratch": True,
            "teacher_targets_limited_to_fold_train": True,
            "outer_dev_used_for_checkpoint_selection": True,
            "outer_dev_is_not_test": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
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

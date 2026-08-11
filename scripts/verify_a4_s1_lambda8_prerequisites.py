#!/usr/bin/env python
"""Verify immutable prerequisites for the A4-S1 8192-row follow-up before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402

PRIMARY_CONFIG = (
    ROOT
    / "configs/experiment/"
    "e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml"
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _resolve(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("artifact paths must be strings")
    path = (ROOT / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def run(config_path: Path) -> dict[str, Any]:
    raw = _read_yaml(config_path)
    experiment = raw["experiment"]
    provenance = raw["provenance"]
    data = raw["data"]
    training = raw["training"]
    decision = raw["decision_contract"]

    if experiment["arm_role"] != "primary_selected_lambda":
        raise ValueError("preflight is for the primary lambda=8 S1 arm only")

    # Bind the config to the signed train-only CV decision.
    selection_path = _resolve(provenance["lambda_selection_artifact"])
    selection_file_sha = _sha256(selection_path)
    if selection_file_sha != provenance["lambda_selection_file_sha256"]:
        raise ValueError("lambda-selection file SHA256 mismatch")
    selection = _read_json(selection_path)
    if not verify_artifact_hash(selection):
        raise ValueError("lambda-selection artifact signature is invalid")
    if selection["artifact_sha256"] != provenance["lambda_selection_artifact_sha256"]:
        raise ValueError("lambda-selection artifact identity mismatch")
    if selection["status"] != "passed":
        raise ValueError("lambda-selection protocol did not pass")
    if float(selection["selected_lambda_candidate"]) != 8.0:
        raise ValueError("signed lambda-selection did not select lambda=8")
    if selection["lambda_grid_boundary_hit"] is not False:
        raise ValueError("selected lambda hit the candidate-grid boundary")
    if selection["promotion_ready"] is not True:
        raise ValueError("selected lambda is not promotion-ready")
    scope = selection["scope"]
    if scope["public_validation_samples_opened"] != 0:
        raise ValueError("lambda selection opened public validation")
    if scope["official_test_opened"] is not False:
        raise ValueError("lambda selection opened official test")
    if (
        selection["code_identity"]["git_commit"]
        != provenance["lambda_selection_code_commit"]
    ):
        raise ValueError("lambda-selection code commit mismatch")

    # Bind train and frozen validation cache identities.
    train_manifest_path = _resolve(data["cache_manifest"])
    if _sha256(train_manifest_path) != data["cache_manifest_sha256"]:
        raise ValueError("train-cache manifest SHA256 mismatch")
    train_manifest = _read_json(train_manifest_path)
    if train_manifest["artifact_sha256"] != data["cache_artifact_sha256"]:
        raise ValueError("train-cache artifact identity mismatch")
    if int(train_manifest["split_counts"]["train"]) != int(data["expected_train_rows"]):
        raise ValueError("train-cache row count mismatch")

    validation_manifest_path = _resolve(data["validation_cache_manifest"])
    if _sha256(validation_manifest_path) != data["validation_cache_manifest_sha256"]:
        raise ValueError("validation-cache manifest SHA256 mismatch")
    validation_manifest = _read_json(validation_manifest_path)
    if (
        validation_manifest["artifact_sha256"]
        != data["validation_cache_artifact_sha256"]
    ):
        raise ValueError("validation-cache artifact identity mismatch")
    if int(validation_manifest["split_counts"]["validation"]) != int(
        data["expected_validation_rows"]
    ):
        raise ValueError("validation-cache row count mismatch")

    train_sequences = {str(value) for value in data["train_sequence_ids"]}
    validation_sequences = {str(value) for value in data["validation_sequence_ids"]}
    if len(train_sequences) != 9 or len(validation_sequences) != 3:
        raise ValueError("expected 9 train and 3 validation sequences")
    if train_sequences & validation_sequences:
        raise ValueError("train and validation sequence IDs overlap")

    # Bind the expanded train-only RGB teacher.
    teacher_cfg = data["dinov3_relational_teacher"]
    teacher_path = _resolve(teacher_cfg["manifest"])
    if _sha256(teacher_path) != teacher_cfg["manifest_sha256"]:
        raise ValueError("DINO teacher manifest SHA256 mismatch")
    teacher = _read_json(teacher_path)
    if teacher["artifact_sha256"] != teacher_cfg["artifact_sha256"]:
        raise ValueError("DINO teacher artifact identity mismatch")
    if teacher["status"] != "passed":
        raise ValueError("DINO teacher status is not passed")
    if int(teacher["scope"]["row_count"]) != 8192:
        raise ValueError("DINO teacher must contain exactly 8192 train rows")
    if teacher["scope"]["validation_or_test_opened"] is not False:
        raise ValueError("DINO teacher opened validation/test")
    if teacher["scope"]["ttc_labels_read"] is not False:
        raise ValueError("DINO teacher read TTC labels")
    if teacher["claim_boundary"]["teacher_source_modality"] != "rgb":
        raise ValueError("DINO teacher source modality must be RGB")
    if teacher["claim_boundary"]["event_tensor_used_as_teacher_input"] is not False:
        raise ValueError("event tensor was used as DINO teacher input")

    # Freeze the actual S1 training contract.
    if float(training["representation_distillation_weight"]) != 8.0:
        raise ValueError("primary S1 representation_distillation_weight must equal 8")
    if (
        training["representation_teacher_cache_artifact_sha256"]
        != teacher["artifact_sha256"]
    ):
        raise ValueError("training/teacher artifact identities differ")
    if int(training["epochs"]) != 18:
        raise ValueError("primary S1 must preserve the 18-epoch A4/CV horizon")
    if int(training["seed"]) != 7:
        raise ValueError("primary S1 seed must remain 7")
    if int(decision["expected_parameter_count"]) != 355118:
        raise ValueError("primary S1 must preserve the 355,118-parameter student")
    if decision["architecture_scaling_in_this_arm"] is not False:
        raise ValueError("architecture scaling must remain outside S1")

    # The S1 branch is allowed to be a descendant of the frozen CV commit.
    required_ancestor = str(provenance["lambda_selection_code_commit"])
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_ancestor, "HEAD"],
        cwd=ROOT,
        check=True,
    )

    payload = {
        "status": "passed",
        "git_head": _git("rev-parse", "HEAD"),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "selected_lambda": 8.0,
        "lambda_selection_artifact_sha256": selection["artifact_sha256"],
        "train_rows": int(data["expected_train_rows"]),
        "validation_rows": int(data["expected_validation_rows"]),
        "train_validation_sequence_overlap": sorted(
            train_sequences & validation_sequences
        ),
        "teacher_rows": int(teacher["scope"]["row_count"]),
        "teacher_source_modality": teacher["claim_boundary"]["teacher_source_modality"],
        "student_parameter_count_expected": int(decision["expected_parameter_count"]),
        "epochs": int(training["epochs"]),
        "seed": int(training["seed"]),
        "official_test_opened": False,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PRIMARY_CONFIG)
    args = parser.parse_args()
    payload = run(args.config.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Static contract tests for the A4-S1 8192-row lambda-selected follow-up."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = (
    ROOT
    / "configs/experiment/"
    "e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml"
)
CONTROL = (
    ROOT
    / "configs/experiment/"
    "e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda4_control_v1.yaml"
)


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_primary_s1_contract_is_frozen_to_train_only_selected_lambda() -> None:
    raw = _load(PRIMARY)
    experiment = raw["experiment"]
    provenance = raw["provenance"]
    data = raw["data"]
    training = raw["training"]
    decision = raw["decision_contract"]

    assert experiment["arm_role"] == "primary_selected_lambda"
    assert provenance["lambda_selection_candidate"] == 8.0
    assert provenance["lambda_selection_boundary_hit"] is False
    assert provenance["lambda_selection_promotion_ready"] is True
    assert provenance["lambda_selection_public_validation_samples_opened"] == 0
    assert (
        provenance["lambda_selection_artifact_sha256"]
        == "68a8d049509167be5db7217acb9cbd69dbd4845cf93833fcab324cc6da919318"
    )

    assert data["expected_train_rows"] == 8192
    assert data["expected_validation_rows"] == 2048
    assert data["cache_manifest"] != data["validation_cache_manifest"]
    assert data["opened_splits"] == ["train", "validation"]
    assert data["official_test_opened"] is False
    assert data["codabench_opened"] is False
    assert data["evttc_test_opened"] is False

    assert training["seed"] == 7
    assert training["epochs"] == 18
    assert training["representation_supervision"] == "dinov3_local_relational"
    assert training["representation_distillation_weight"] == 8.0

    assert decision["expected_parameter_count"] == 355118
    assert decision["architecture_scaling_in_this_arm"] is False
    assert decision["student_capacity_change_in_this_arm"] is False
    assert decision["train_scaling"]["parent_train_rows"] == 2048
    assert decision["train_scaling"]["followup_train_rows"] == 8192
    assert decision["train_scaling"]["frozen_validation_rows"] == 2048


def test_lambda4_control_differs_from_primary_only_in_declared_role_and_weight(
) -> None:
    primary = _load(PRIMARY)
    control = _load(CONTROL)

    assert primary["experiment"]["arm_role"] == "primary_selected_lambda"
    assert control["experiment"]["arm_role"] == "attribution_control_lambda4"

    assert primary["training"]["representation_distillation_weight"] == 8.0
    assert control["training"]["representation_distillation_weight"] == 4.0

    for section in ("model_config", "data", "loss"):
        assert primary[section] == control[section]

    p_training = dict(primary["training"])
    c_training = dict(control["training"])
    p_training.pop("representation_distillation_weight")
    c_training.pop("representation_distillation_weight")
    assert p_training == c_training

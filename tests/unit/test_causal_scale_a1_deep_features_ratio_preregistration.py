from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import scripts.train_causal_scale_eap_screen as runner
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.training.causal_scale_eap import _foreground_only_loss_config

ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_deep_features_v1.yaml"
)
CANDIDATE = (
    ROOT
    / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_deep_features_ratio_v1.yaml"
)


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_ratio_arm_changes_only_identity_contract_and_pair_ratio_weight() -> None:
    parent = _yaml(PARENT)
    candidate = _yaml(CANDIDATE)

    assert candidate["model_config"] == parent["model_config"]
    assert candidate["data"] == parent["data"]
    assert candidate["training"] == parent["training"]
    changed_loss = {
        key
        for key in parent["loss"] | candidate["loss"]
        if parent["loss"].get(key) != candidate["loss"].get(key)
    }
    assert changed_loss == {"foreground_pair_ratio_weight"}
    assert parent["loss"]["foreground_pair_ratio_weight"] == 0.0
    assert candidate["loss"]["foreground_pair_ratio_weight"] == 5.0


def test_ratio_arm_hash_model_count_and_warmup_are_frozen() -> None:
    raw = _yaml(CANDIDATE)
    model_path = ROOT / str(raw["model_config"])
    model_config = runner._model_config(model_path)
    model = CausalScaleTTC(model_config)
    loss_config = CausalScaleTTCLossConfig(**raw["loss"])
    warmup = _foreground_only_loss_config(loss_config)

    assert sum(parameter.numel() for parameter in model.parameters()) == 355_118
    assert runner._sha256(model_path) == raw["decision_contract"][
        "model_config_sha256"
    ]
    assert loss_config.foreground_pair_ratio_weight == 5.0
    assert warmup.foreground_pair_ratio_weight == 0.0


def test_ratio_weight_matches_preregistered_train_only_scale() -> None:
    contract = _yaml(CANDIDATE)["decision_contract"]["train_only_weight_normalization"]

    assert contract["validation_used_to_choose_weight"] is False
    assert contract["hyperparameter_sweep_permitted"] is False
    assert contract["weighted_pair_ratio_contribution"] == pytest.approx(
        5.0 * contract["raw_pair_ratio_loss"]
    )
    assert contract["weighted_width_contribution"] == pytest.approx(
        1.25 * contract["raw_width_loss"]
    )
    ratio = (
        contract["weighted_pair_ratio_contribution"]
        / contract["weighted_width_contribution"]
    )
    assert 1.0 <= ratio <= 1.1

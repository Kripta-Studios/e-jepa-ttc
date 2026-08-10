from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

import scripts.train_causal_scale_eap_screen as runner
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC

ROOT = Path(__file__).resolve().parents[2]
PARENT_MODEL = ROOT / "configs/model/e_jepa_causal_scale_event_v8_t015.yaml"
CANDIDATE_MODEL = ROOT / "configs/model/e_jepa_causal_scale_event_v8_t015_fullres.yaml"
PARENT_EXPERIMENT = (
    ROOT
    / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_geometry_v1.yaml"
)
CANDIDATE_EXPERIMENT = (
    ROOT
    / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_fullres_v1.yaml"
)


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_fullres_model_changes_only_decoder() -> None:
    parent = _yaml(PARENT_MODEL)
    candidate = _yaml(CANDIDATE_MODEL)

    changed = {key for key in parent | candidate if parent.get(key) != candidate.get(key)}

    assert changed == {"foreground_decoder"}
    assert parent["foreground_decoder"] == "equivariant_separable"
    assert candidate["foreground_decoder"] == "equivariant_fullres"


def test_fullres_experiment_preserves_a1_data_training_and_loss() -> None:
    parent = _yaml(PARENT_EXPERIMENT)
    candidate = _yaml(CANDIDATE_EXPERIMENT)

    assert candidate["data"] == parent["data"]
    assert candidate["training"] == parent["training"]
    assert candidate["loss"] == parent["loss"]
    assert candidate["loss"]["foreground_pair_ratio_weight"] == 0.0
    assert candidate["loss"]["foreground_bce_weight"] == 0.0
    assert candidate["loss"]["foreground_dice_weight"] == 0.0


def test_fullres_preregistered_hash_and_parameter_count_are_exact() -> None:
    raw = _yaml(CANDIDATE_EXPERIMENT)
    config = runner._model_config(CANDIDATE_MODEL)
    model = CausalScaleTTC(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert config.foreground_decoder == "equivariant_fullres"
    assert parameter_count == 340_870
    assert parameter_count < 344_591
    assert (
        runner._sha256(CANDIDATE_MODEL)
        == raw["decision_contract"]["model_config_sha256"]
    )
    parent_config = runner._model_config(PARENT_MODEL)
    parent_values = asdict(parent_config)
    candidate_values = asdict(config)
    changed = {
        key for key in parent_values if parent_values[key] != candidate_values[key]
    }
    assert changed == {"foreground_decoder"}

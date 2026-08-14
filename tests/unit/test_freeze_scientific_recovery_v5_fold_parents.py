"""Contracts for fold-local A4 parent configuration freezing."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.freeze_scientific_recovery_v5_fold_parents import (
    PROTOCOL_PATH,
    build_parent_configs,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "scientific_recovery_v5"
# This retained V5 recipe is minimized from the signed V7 effective-config evidence.
A4_SOURCE = FIXTURE_ROOT / "a4_s1_lambda8_causal_left_seed7.yaml"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_fold_parent_configs_are_self_contained_and_preserve_a4_recipe() -> None:
    protocol = _json(PROTOCOL_PATH)
    source = _yaml(A4_SOURCE)

    configs = build_parent_configs(protocol, source)

    assert len(configs) == 3
    for fold, config in configs.items():
        frozen_fold = protocol["folds"][fold]
        data = config["data"]
        training = config["training"]
        decision = config["decision_contract"]
        assert data["opened_splits"] == ["train"]
        assert data["train_sequence_ids"] == frozen_fold["train_sequence_ids"]
        assert data["dev_sequence_ids"] == frozen_fold["dev_sequence_ids"]
        assert set(data["train_sequence_ids"]).isdisjoint(data["dev_sequence_ids"])
        assert "validation_cache_manifest" not in data
        assert training["initialization_mode"] == "none"
        assert training["initialization_checkpoint"] is None
        assert training["freeze_encoder"] is False
        assert training["num_workers"] == 0
        assert training["seed"] == 7
        assert training["epochs"] == source["training"]["epochs"]
        assert training["learning_rate"] == source["training"]["learning_rate"]
        assert decision["outer_dev_used_for_checkpoint_selection"] is True
        assert decision["outer_dev_is_not_test"] is True
        assert decision["public_validation_used_for_selection"] is False
        assert decision["private_test_remains_closed"] is True
        assert config["model_config"] == source["model_config"]
        assert "parent_arm" not in config["experiment"]
        assert (
            config["provenance"]["v5_fold_parent_interpretation"][
                "weights_inherited_from_historical_a4"
            ]
            is False
        )

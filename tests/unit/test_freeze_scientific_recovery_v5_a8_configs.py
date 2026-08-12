"""Tests for the preregistered V5 A6/A8.0 grouped-dev configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from e_jepa_ttc.artifacts.hashing import sign_artifact
from scripts.freeze_scientific_recovery_v5_a8_configs import (
    A6_SOURCE,
    A8_SOURCE,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PROTOCOL_PATH,
    build_configs,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_a8_configs_are_train_only_causal_and_use_fixed_parent() -> None:
    configs = build_configs(_json(PROTOCOL_PATH), _yaml(A6_SOURCE), _yaml(A8_SOURCE))

    assert len(configs) == 6
    for name, config in configs.items():
        data = config["data"]
        training = config["training"]
        decision = config["decision_contract"]
        assert data["opened_splits"] == ["train"]
        assert "validation_cache_manifest" not in data
        assert set(data["train_sequence_ids"]).isdisjoint(data["dev_sequence_ids"])
        assert training["num_workers"] == 0
        assert training["seed"] == 7
        assert training["initialization_checkpoint"] == PARENT_CHECKPOINT
        assert training["initialization_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256
        assert decision["public_validation_used_for_selection"] is False
        assert decision["private_test_remains_closed"] is True
        assert config["model_config"].endswith("_causal.yaml")
        if name.startswith("a8_0"):
            assert "dual_transport" in config["model_config"]
            assert decision["dual_stream_contract"]["transport_encoder_trainable"] is True
        else:
            assert "transport_adapter" in config["model_config"]


def test_a8_configs_reject_protocol_that_opens_private_test() -> None:
    protocol = copy.deepcopy(_json(PROTOCOL_PATH))
    protocol["checks"]["private_test_opened"] = True
    sign_artifact(protocol)

    with pytest.raises(ValueError, match="private/test must remain closed"):
        build_configs(protocol, _yaml(A6_SOURCE), _yaml(A8_SOURCE))

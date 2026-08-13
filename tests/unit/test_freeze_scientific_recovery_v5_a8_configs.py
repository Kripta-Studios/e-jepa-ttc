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
    PROTOCOL_PATH,
    build_configs,
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_a8_configs_are_train_only_causal_and_use_fixed_parent() -> None:
    parents = {
        fold: {
            "checkpoint": f"artifacts/runs/fold{fold}/model_best.pt",
            "checkpoint_sha256": str(fold) * 64,
        }
        for fold in range(3)
    }
    configs = build_configs(_json(PROTOCOL_PATH), _yaml(A6_SOURCE), _yaml(A8_SOURCE), parents)

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
        fold = int(name.split("fold", 1)[1].split("_", 1)[0])
        assert training["initialization_checkpoint"] == parents[fold]["checkpoint"]
        assert training["initialization_checkpoint_sha256"] == parents[fold]["checkpoint_sha256"]
        assert decision["public_validation_used_for_selection"] is False
        assert decision["private_test_remains_closed"] is True
        assert decision["require_finite_metrics_for_all_dev_sequences"] is True
        assert "require_finite_metrics_for_all_validation_sequences" not in decision
        assert "primary_baseline" not in decision
        assert "frozen_validation_rows" not in decision
        assert config["model_config"].endswith("_causal.yaml")
        if name.startswith("a8_0"):
            assert "dual_transport" in config["model_config"]
            assert decision["dual_stream_contract"]["transport_encoder_trainable"] is True
            assert decision["a8_0_gate"]["comparator"].startswith("A6_causal")
        else:
            assert "transport_adapter" in config["model_config"]
            assert "a8_0_gate" not in decision


def test_a8_configs_reject_protocol_that_opens_private_test() -> None:
    protocol = copy.deepcopy(_json(PROTOCOL_PATH))
    protocol["checks"]["private_test_opened"] = True
    sign_artifact(protocol)

    with pytest.raises(ValueError, match="private/test must remain closed"):
        build_configs(protocol, _yaml(A6_SOURCE), _yaml(A8_SOURCE), {})


def test_a8_configs_reject_missing_fold_parent() -> None:
    with pytest.raises(ValueError, match="fold 2 lacks"):
        build_configs(
            _json(PROTOCOL_PATH),
            _yaml(A6_SOURCE),
            _yaml(A8_SOURCE),
            {
                0: {"checkpoint": "f0.pt", "checkpoint_sha256": "0" * 64},
                1: {"checkpoint": "f1.pt", "checkpoint_sha256": "1" * 64},
            },
        )

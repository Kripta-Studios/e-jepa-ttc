from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
from scripts.freeze_causal_hardening_configs import _base_mutate
from scripts.freeze_causal_hardening_configs import main as freeze_causal_main
from scripts.freeze_scientific_recovery_s1_configs import (
    ROOT,
    model_config_kwargs,
    parameter_count,
)

MODEL_CASES = (
    (
        "configs/model/e_jepa_causal_scale_event_v10_transport_adapter_r1_t002_legacy.yaml",
        498_130,
    ),
    (
        "configs/model/e_jepa_causal_scale_event_v10_transport_adapter_r1_t002_causal.yaml",
        498_130,
    ),
    (
        "configs/model/e_jepa_causal_scale_event_v11_dual_transport_r1_t002_legacy.yaml",
        627_827,
    ),
    (
        "configs/model/e_jepa_causal_scale_event_v11_dual_transport_r1_t002_causal.yaml",
        627_827,
    ),
)


@pytest.mark.parametrize(("relative_path", "expected_parameter_count"), MODEL_CASES)
def test_a6_a7_model_yaml_normalizes_risk_thresholds_and_instantiates(
    relative_path: str,
    expected_parameter_count: int,
) -> None:
    model_path = ROOT / relative_path
    raw = model_config_kwargs(model_path)

    assert raw["risk_thresholds_s"] == (0.5, 1.0, 2.0, 4.0)
    CausalScaleTTCConfig(**raw)
    assert parameter_count(model_path) == expected_parameter_count


@pytest.mark.parametrize(
    "invalid_thresholds",
    (
        (0.5, 1.0, 1.0, 4.0),
        (0.5, 2.0, 1.0, 4.0),
    ),
)
def test_normalization_does_not_disable_risk_threshold_guardrail(
    invalid_thresholds: tuple[float, ...],
) -> None:
    model_path = ROOT / MODEL_CASES[0][0]
    raw = model_config_kwargs(model_path)
    raw["risk_thresholds_s"] = invalid_thresholds

    with pytest.raises(ValueError, match="risk thresholds must be unique and strictly increasing"):
        CausalScaleTTCConfig(**raw)


def test_causal_a6_replication_declares_fixed_parent_seed_semantics() -> None:
    source = {
        "model_config": "old.yaml",
        "training": {"seed": 7, "num_workers": 8, "prefetch_factor": 4},
        "experiment": {"name": "a6", "protocol_version": "a6_v1"},
        "decision_contract": {},
    }
    frozen = _base_mutate(source, "a6", 23, 0, 2)
    decision = frozen["decision_contract"]

    assert decision["temporal_smoothing_mode"] == "causal_left"
    # The parent binding is added by the freezer's A6/A7 checkpoint branch.
    # This base mutation must preserve the transport seed without creating a
    # seed-specific parent path.
    assert frozen["training"]["seed"] == 23
    assert "initialization_checkpoint" not in frozen["training"]


def test_causal_a6_freezer_binds_every_transport_seed_to_fixed_seed7_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = ROOT / "artifacts/runs/scientific_recovery_a4_causal_left_seed7/model_best.pt"
    if not parent.is_file():
        pytest.skip("local scientific-recovery parent checkpoint is unavailable")
    a4_source = tmp_path / "a4.yaml"
    winner_source = tmp_path / "a6.yaml"
    output = tmp_path / "frozen"
    common = {
        "experiment": {"name": "arm", "protocol_version": "arm_v1"},
        "model_config": "old.yaml",
        "training": {"seed": 7, "num_workers": 0, "prefetch_factor": 2},
        "decision_contract": {},
    }
    a4_source.write_text(yaml.safe_dump(common), encoding="utf-8")
    winner = yaml.safe_load(yaml.safe_dump(common))
    winner["decision_contract"]["adapter_contract"] = {}
    winner_source.write_text(yaml.safe_dump(winner), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_causal_hardening_configs.py",
            "--a4-source-config",
            str(a4_source),
            "--winner-source-config",
            str(winner_source),
            "--winner-stage",
            "a6",
            "--causal-a4-checkpoint",
            str(parent),
            "--output-dir",
            str(output),
            "--num-workers",
            "0",
            "--prefetch-factor",
            "2",
        ],
    )

    assert freeze_causal_main() == 0
    parent_paths = set()
    for seed in (7, 13, 23):
        frozen = yaml.safe_load(
            (output / f"a6_s1_causal_left_seed{seed}.yaml").read_text(encoding="utf-8")
        )
        parent_paths.add(frozen["training"]["initialization_checkpoint"])
        decision = frozen["decision_contract"]
        assert decision["replication_parent_policy"] == "fixed_a4_causal_seed7"
        assert decision["replication_transport_seed"] == seed
    assert parent_paths == {"artifacts/runs/scientific_recovery_a4_causal_left_seed7/model_best.pt"}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["winner_parent_policy"] == "fixed_a4_causal_seed7"

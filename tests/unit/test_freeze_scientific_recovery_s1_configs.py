from __future__ import annotations

import pytest

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
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

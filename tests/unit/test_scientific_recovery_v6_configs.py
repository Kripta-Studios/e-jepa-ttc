from __future__ import annotations

import copy

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCConfig
from scripts.freeze_scientific_recovery_v6_configs import build_configs
from scripts.train_causal_scale_eap_screen import _validate_v6_d0_payload


def _base() -> dict:
    return {
        "experiment": {},
        "model_config": "old",
        "training": {
            "foreground_warmup_epochs": 0,
            "initialization_checkpoint": "parent.pt",
            "initialization_checkpoint_sha256": "a" * 64,
            "initialization_mode": "shape_compatible",
            "freeze_encoder": True,
        },
        "decision_contract": {
            "expected_parameter_count": 627827,
            "representation_change": {
                "type": "a4_frozen_geometry_plus_trainable_transport_encoder",
                "transport_radius": 1,
                "transport_candidates_per_position": 9,
            },
            "dual_stream_contract": {"transport_radius": 1},
            "preflight_contract": {},
            "a8_0_gate": {},
        },
    }


def _d0() -> dict:
    value = {
        "artifact_type": "scientific_recovery_v6_d0_a8_oof_failure_modes_v1",
        "status": "completed_exploratory_outer_dev_diagnostic",
        "decision": {
            "selected_family": "motion_scale",
            "selected_branch": "V6.1_MULTI_SCALE_TRANSPORT",
        },
        "contracts": {
            "public_validation_opened": False,
            "private_test_opened": False,
            "promotion_authorized": False,
        },
    }
    return sign_artifact(value)


def test_v6_freezer_builds_one_change_radius2_and_causal_a5(tmp_path) -> None:
    d0_path = tmp_path / "d0.json"
    d0_path.write_text("{}", encoding="utf-8")
    configs = build_configs([copy.deepcopy(_base()) for _ in range(3)], d0_path, _d0())

    assert len(configs) == 6
    a5 = configs["a5_causal_fold0"]
    assert a5["training"]["freeze_encoder"] is False
    assert a5["training"]["initialization_mode"] == "none"
    assert a5["decision_contract"]["geometry_preservation_required"] is False
    v6 = configs["v6_1_r2_fold0"]
    assert v6["training"]["freeze_encoder"] is True
    assert v6["decision_contract"]["representation_change"]["transport_radius"] == 2
    assert "preflight_contract" not in v6["decision_contract"]


def test_v6_d0_payload_authorizes_only_radius2_motion_scale() -> None:
    config = CausalScaleTTCConfig(
        transport_enabled=True,
        transport_radius=2,
        transport_encoder_copy_enabled=True,
    )
    contract = {
        "source_radius": 1,
        "candidate_radius": 2,
        "single_scientific_difference": "transport_radius_1_to_2",
    }

    result = _validate_v6_d0_payload(_d0(), contract, config)

    assert result["selected_family"] == "motion_scale"
    assert result["candidate_radius"] == 2

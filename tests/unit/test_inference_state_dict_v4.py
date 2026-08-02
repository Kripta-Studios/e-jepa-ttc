from __future__ import annotations

from e_jepa_ttc.models.eap_lhr_jepa_ttc import EAPLHRJEPATTC, EAPLHRJEPATTCConfig


def test_inference_state_dict_excludes_target_and_predictor_training_branches() -> None:
    model = EAPLHRJEPATTC(
        EAPLHRJEPATTCConfig(
            endpoint_event_channels=4,
            observable_motion_dim=3,
            geometry_target_dim=5,
            category_count=2,
            dim=32,
        )
    )
    state = model.inference_state_dict()
    assert state
    assert not any(key.startswith("target_roi_encoder.") for key in state)
    assert not any(key.startswith("jepa_predictor.") for key in state)
    assert any(key.startswith("roi_encoder.") for key in state)

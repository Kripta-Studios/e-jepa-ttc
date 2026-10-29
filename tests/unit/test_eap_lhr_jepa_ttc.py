from __future__ import annotations

import torch

from e_jepa_ttc.data.garlttc_lhr_cache import FORBIDDEN_MODEL_INPUT_KEYS
from e_jepa_ttc.models.eap_lhr_jepa_ttc import EAPLHRJEPATTC, EAPLHRJEPATTCConfig
from e_jepa_ttc.training.eap_lhr_jepa_ttc import _assert_model_inputs_are_causal


def test_lhr_model_preserves_ttc_head_and_discards_only_training_auxiliaries() -> None:
    model = EAPLHRJEPATTC(EAPLHRJEPATTCConfig(dim=64))
    state = model.inference_state_dict()
    assert any(key.startswith("height_head.") for key in state)
    assert any(key.startswith("ttc_residual_head.") for key in state)
    assert any(key.startswith("motion_encoder.") for key in state)
    assert not any(key.startswith("target_roi_encoder.") for key in state)
    assert not any(key.startswith("jepa_predictor.") for key in state)


def test_lhr_forward_is_finite_and_uses_only_observable_motion() -> None:
    model = EAPLHRJEPATTC(EAPLHRJEPATTCConfig(dim=64))
    batch = 2
    output = model(
        full_frame_events=torch.randn(batch, 2, 21, 90, 160),
        event_roi_pair=torch.randn(batch, 40, 128, 128),
        delta_t_s=torch.full((batch,), 0.1),
        observable_motion=torch.zeros(batch, 18),
    )
    assert output.ttc_seconds.shape == (batch,)
    assert output.predicted_heights.shape == (batch, 2)
    assert output.geometry_prediction.shape == (batch, 20)
    assert torch.isfinite(output.ttc_seconds).all()
    assert (output.predicted_heights > 0).all()


def test_target_encoder_updates_by_ema() -> None:
    model = EAPLHRJEPATTC(EAPLHRJEPATTCConfig(dim=64))
    online = next(model.roi_encoder.parameters())
    target = next(model.target_roi_encoder.parameters())
    with torch.no_grad():
        online.add_(1.0)
        before = target.clone()
    model.update_target(0.5)
    assert not torch.equal(before, target)


def test_privileged_fields_are_rejected_as_model_inputs() -> None:
    for key in FORBIDDEN_MODEL_INPUT_KEYS:
        try:
            _assert_model_inputs_are_causal({"garl_event_roi", key})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Expected privileged input {key!r} to be rejected")

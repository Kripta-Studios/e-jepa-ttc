from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.training.causal_scale_eap import (
    CausalScaleEAPTrainingConfig,
    _module_tensor_sha256,
    _shape_compatible_initialize,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shape_compatible_initialization_recovers_complete_a4_encoder(tmp_path: Path) -> None:
    source = CausalScaleTTC(CausalScaleTTCConfig())
    with torch.no_grad():
        for parameter in source.encoder.parameters():
            parameter.add_(0.123)
    checkpoint = tmp_path / "a4.pt"
    torch.save(
        {
            "artifact_type": "causal_scale_eap_public_validation_checkpoint_v1",
            "model_config": source.checkpoint_config(),
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )
    target = CausalScaleTTC(
        CausalScaleTTCConfig(transport_enabled=True, transport_radius=1, transport_temperature=0.02)
    )
    report = _shape_compatible_initialize(target, checkpoint)
    assert report["complete_encoder_loaded"] is True
    assert report["mismatched_tensor_count"] > 0  # transport-expanded fusion layers
    source_encoder = source.encoder.state_dict()
    target_encoder = target.encoder.state_dict()
    assert source_encoder.keys() == target_encoder.keys()
    for name in source_encoder:
        torch.testing.assert_close(source_encoder[name], target_encoder[name], rtol=0, atol=0)
    assert _module_tensor_sha256(source.encoder) == _module_tensor_sha256(target.encoder)


def test_dual_transport_initializes_as_exact_primary_encoder_copy(tmp_path: Path) -> None:
    source = CausalScaleTTC(CausalScaleTTCConfig())
    checkpoint = tmp_path / "a4.pt"
    torch.save(
        {
            "artifact_type": "causal_scale_eap_grouped_dev_checkpoint_v1",
            "model_config": source.checkpoint_config(),
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )
    target = CausalScaleTTC(
        CausalScaleTTCConfig(
            transport_enabled=True,
            transport_encoder_copy_enabled=True,
            transport_radius=1,
            transport_temperature=0.02,
        )
    )

    report = _shape_compatible_initialize(target, checkpoint)

    assert report["transport_encoder_initialized_from_primary"] is True
    assert target.transport_encoder is not None
    assert _module_tensor_sha256(target.encoder) == _module_tensor_sha256(target.transport_encoder)


def test_anchor_training_config_requires_zero_warmup_and_checkpoint(tmp_path: Path) -> None:
    fake = tmp_path / "a4.pt"
    fake.write_bytes(b"checkpoint")
    digest = _sha(fake)
    with pytest.raises(ValueError, match="foreground_warmup_epochs=0"):
        CausalScaleEAPTrainingConfig(
            initialization_checkpoint=str(fake),
            initialization_checkpoint_sha256=digest,
            initialization_mode="shape_compatible",
            freeze_encoder=True,
            foreground_warmup_epochs=3,
        )
    cfg = CausalScaleEAPTrainingConfig(
        initialization_checkpoint=str(fake),
        initialization_checkpoint_sha256=digest,
        initialization_mode="shape_compatible",
        freeze_encoder=True,
        foreground_warmup_epochs=0,
    )
    assert cfg.freeze_encoder is True


def test_baseline_defaults_do_not_initialize_or_freeze() -> None:
    cfg = CausalScaleEAPTrainingConfig()
    assert cfg.initialization_mode == "none"
    assert cfg.initialization_checkpoint is None
    assert cfg.freeze_encoder is False

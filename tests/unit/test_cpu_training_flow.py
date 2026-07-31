"""Test CPU training flow for EAP TTC."""

import torch
from torch.optim import AdamW

from e_jepa_ttc.training.eap_jepa import (
    build_eap_jepa_models,
    compute_eap_jepa_objective,
    update_eap_jepa_ema,
)


def test_cpu_training_flow():
    """Verify that gradients propagate to the encoder and EMA updates the target."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Need EAPJEPATrainerConfig
    from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig

    config = EAPJEPATrainerConfig(
        width=160,
        height=90,
        bins=5,
    )

    # Use real models via builder
    models = build_eap_jepa_models(
        config=config,
        device=device,
    )
    encoder = models.online_encoder
    target_encoder = models.target_encoder
    predictor = models.predictor

    # Dummy inputs
    batch_size = 2

    # 1. Forward pass
    # Raw events tensors.
    # The models are event-tubelet-transformers which take in_channels=21.
    context_voxels = torch.randn(batch_size, 21, 90, 160, device=device, requires_grad=True)
    future_voxels = torch.randn(batch_size, 3, 21, 90, 160, device=device)
    valid_mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)

    # Optimizer
    optimizer = AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)

    # Pass through objective function
    jepa_output = compute_eap_jepa_objective(
        online_encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        context=context_voxels,
        futures=future_voxels,
        future_valid=valid_mask,
        config=config,
    )

    loss = jepa_output.loss
    # Add fake TTC loss (normally done in eap_ttc by returning embeddings)
    # But since we just want to test gradient flow:
    ttc_loss = loss * 0.0 + sum(p.mean() for p in encoder.parameters()) * 0.0

    total_loss = loss + ttc_loss
    total_loss.backward()

    # 2. Check gradients
    # Check encoder parameters directly.
    has_enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert has_enc_grad, "encoder did not receive gradients!"

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    assert has_grad, "predictor did not receive gradients!"

    # 3. Update EMA
    # Record old target weights
    old_target_weights = {n: p.clone() for n, p in target_encoder.named_parameters()}

    optimizer.step()

    # Update EMA
    update_eap_jepa_ema(
        target_encoder=target_encoder,
        online_encoder=encoder,
        optimizer_step=1,
        total_optimizer_steps=100,
        config=config,
    )

    # Check that target weights moved towards encoder weights
    moved = False
    for (name, target_p), (name_enc, enc_p) in zip(
        target_encoder.named_parameters(), encoder.named_parameters(), strict=True
    ):
        assert name == name_enc
        old_p = old_target_weights[name]
        if not torch.allclose(target_p, old_p):
            moved = True

            # verify it moved towards the new encoder weights
            expected = 0.99 * old_p + 0.01 * enc_p
            assert torch.allclose(target_p, expected, atol=1e-6)

    assert moved, "Target encoder weights did not update via EMA"

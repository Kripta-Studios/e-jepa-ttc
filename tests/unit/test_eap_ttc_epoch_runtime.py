"""Unit tests to ensure the TTC training epoch runs on CPU and GPU."""

import math
import time

import pytest
import torch
from torch.optim import AdamW

from e_jepa_ttc.data.garlttc_eap import GarlTTCBatch
from e_jepa_ttc.models import infer_tubelet_token_geometry
from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig, build_eap_jepa_models
from e_jepa_ttc.training.eap_ttc import EAPSignedTTCHead, run_ttc_epoch

PRECISION_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@pytest.mark.parametrize(
    ("device_type", "precision"),
    [
        ("cpu", "fp32"),
        pytest.param("cuda", "fp32", marks=pytest.mark.cuda_fp32),
        pytest.param("cuda", "fp16", marks=pytest.mark.cuda_amp),
    ],
)
def test_run_ttc_epoch_integration(device_type: str, precision: str):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device_type)

    config = EAPJEPATrainerConfig(
        event_window_ms=100,
        horizons_ms=(100,),
        max_windows_per_sequence=10,
        precision=precision,
        geometry_loss_weight=0.0,
    )

    # 1. Models
    models = build_eap_jepa_models(config=config, device=device)
    encoder = models.online_encoder
    target_encoder = models.target_encoder
    predictor = models.predictor

    ttc_head = EAPSignedTTCHead(embed_dim=int(encoder.output_dim)).to(device)

    geometry = infer_tubelet_token_geometry(encoder, input_height=90, input_width=160)
    assert geometry.grid_t == 5
    assert geometry.grid_h == 5
    assert geometry.grid_w == 10
    assert geometry.token_count == 250

    # Check device placement
    assert all(p.device.type == device_type for p in encoder.parameters())
    assert all(p.device.type == device_type for p in predictor.parameters())
    assert all(p.device.type == device_type for p in ttc_head.parameters())

    # 2. Batch
    batch_size = 2
    context = torch.rand(batch_size, 21, 90, 160, device=device)
    futures = torch.rand(batch_size, 1, 21, 90, 160, device=device)
    assert context.device.type == device_type
    assert futures.device.type == device_type

    ttc_targets = [
        torch.tensor([0.15, -0.20], dtype=torch.float32, device=device),
        torch.tensor([0.30], dtype=torch.float32, device=device),
    ]
    batch = GarlTTCBatch(
        context=context,
        futures=futures,
        future_valid=torch.ones(batch_size, 1, dtype=torch.bool, device=device),
        bbox_masks=[
            torch.ones(2, 5, 10, dtype=torch.bool, device=device),
            torch.ones(1, 5, 10, dtype=torch.bool, device=device),
        ],
        target_ttc=ttc_targets,
        sequence_ids=["seq1", "seq2"],
        track_ids=[["t1", "t2"], ["t3"]],
        timestamp_us=torch.tensor([1000, 2000], device=device),
        events_paths=["path1", "path2"],
        original_bboxes=[
            torch.tensor([[10, 20, 30, 40], [5, 5, 20, 20]]),
            torch.tensor([[50, 50, 60, 60]]),
        ],
        transformed_bboxes=[
            torch.tensor([[10, 20, 30, 40], [5, 5, 20, 20]]),
            torch.tensor([[50, 50, 60, 60]]),
        ],
        context_event_counts=torch.tensor([100, 200]),
        future_event_counts=torch.tensor([[50], [60]]),
    )
    dataloader = [batch]

    # 3. Scaler and Optimizer
    if precision in ("fp16", "bf16") and device_type == "cuda":
        scaler = torch.amp.GradScaler(device_type, enabled=True)
    else:
        scaler = None

    params = list(encoder.parameters()) + list(predictor.parameters()) + list(ttc_head.parameters())
    optimizer = AdamW(params, lr=1e-3)

    # Save parameters to check changes
    encoder_before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    target_before = {name: p.detach().clone() for name, p in target_encoder.named_parameters()}
    predictor_before = {name: p.detach().clone() for name, p in predictor.named_parameters()}
    ttc_head_before = {name: p.detach().clone() for name, p in ttc_head.named_parameters()}

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- Train phase ---
    train_metrics, optimizer_step, train_median_ttc = run_ttc_epoch(
        epoch=1,
        dataloader=dataloader,
        target_encoder=target_encoder,
        online_encoder=encoder,
        predictor=predictor,
        ttc_head=ttc_head,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=None,
        geometry=geometry,
        device=device,
        config=config,
        start_time=time.time(),
        train=True,
        total_optimizer_steps=1,
        optimizer_step=0,
        train_median_ttc=None,
    )

    # Train asserts
    assert optimizer_step == 1
    assert math.isfinite(train_metrics["loss_jepa"])
    assert math.isfinite(train_metrics["loss_ttc"])
    assert math.isfinite(train_metrics["loss_total"])
    assert train_metrics["optimizer_update_count"] == 1.0

    # Check that parameters changed
    def module_changed(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> bool:
        return any(
            not torch.equal(parameter.detach(), before[name])
            for name, parameter in module.named_parameters()
        )

    assert module_changed(encoder, encoder_before), "Online encoder did not change"
    assert module_changed(predictor, predictor_before), "Predictor did not change"
    assert module_changed(ttc_head, ttc_head_before), "TTC head did not change"
    assert module_changed(target_encoder, target_before), "Target encoder did not change"

    # Check no nan or inf in encoder
    assert not any(torch.isnan(p).any() for p in encoder.parameters())
    assert not any(torch.isinf(p).any() for p in encoder.parameters())

    # Check no grads on target_encoder
    assert all(parameter.grad is None for parameter in target_encoder.parameters())

    if device_type == "cuda":

        def get_grad_norm(module):
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm**0.5

        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
        print("device: GPU")
        print(f"loss_jepa: {train_metrics['loss_jepa']}")
        print(f"loss_ttc: {train_metrics['loss_ttc']}")
        print(f"loss_total: {train_metrics['loss_total']}")
        print(f"encoder_grad_norm: {get_grad_norm(encoder)}")
        print(f"predictor_grad_norm: {get_grad_norm(predictor)}")
        print(f"ttc_head_grad_norm: {get_grad_norm(ttc_head)}")
        print(f"target_grad_norm: {get_grad_norm(target_encoder)}")
        print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"VRAM reserved: {peak_reserved_mb:.2f} MB")
        print(f"VRAM peak allocated: {peak_allocated_mb:.2f} MB")

    # --- Validation phase ---
    encoder_before_val = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    target_before_val = {name: p.detach().clone() for name, p in target_encoder.named_parameters()}

    validation_metrics, validation_step, _ = run_ttc_epoch(
        epoch=1,
        dataloader=dataloader,
        target_encoder=target_encoder,
        online_encoder=encoder,
        predictor=predictor,
        ttc_head=ttc_head,
        scaler=None,
        optimizer=optimizer,
        scheduler=None,
        geometry=geometry,
        device=device,
        config=config,
        start_time=time.time(),
        train=False,
        total_optimizer_steps=1,
        optimizer_step=optimizer_step,
        train_median_ttc=train_median_ttc,
    )

    assert validation_step == optimizer_step
    assert math.isfinite(validation_metrics["loss_jepa"])
    assert math.isfinite(validation_metrics["loss_ttc"])
    assert math.isfinite(validation_metrics["loss_total"])
    assert validation_metrics["optimizer_update_count"] == 0.0
    assert validation_metrics.get("ema_momentum", 0.0) == 0.0

    # Assert nothing changed in val
    assert not module_changed(encoder, encoder_before_val)
    assert not module_changed(target_encoder, target_before_val)

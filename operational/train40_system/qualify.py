"""Real TRAIN40 forward/backward admission with zero optimizer updates."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from operational.efficient_context.common import Lease, atomic_json, digest
from operational.train40_system.contracts import read
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.models import Inputs, resource_guard


def run(output: Path) -> None:
    """Exercise both original losses and all encoder gradients without changing learned weights."""
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    from e_jepa_ttc.reproducibility import seed_everything
    from e_jepa_ttc.training.causal_scale_eap import (
        _autocast,
        _foreground_only_loss_config,
        _loss,
    )

    allowed, resource = resource_guard(output)
    if not allowed:
        raise RuntimeError(f"Real TRAIN admission cannot own GPU: {resource}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_everything(7, deterministic=True)
    protocol = read(output / "TRAINING_PROTOCOL.json")
    data = Inputs(output)
    try:
        ids = list(range(32))
        batch = data.batch(ids).to(torch.device("cuda"))
        rows = []
        for arm in ("a5", "c2f"):
            recipe = protocol["producers"][arm]
            model = CausalScaleTTC(CausalScaleTTCConfig(**recipe["model_config"])).to("cuda")
            model.train()
            losses = CausalScaleTTCLossConfig(**recipe["loss_config"])
            for iteration in range(4):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with _autocast(torch.device("cuda"), "bf16"):
                    total, components, _ = _loss(
                        model,
                        batch,
                        _foreground_only_loss_config(losses) if iteration == 0 else losses,
                        mask_t0_as_proxy=True,
                        foreground_supervision="bbox_geometry",
                        representation_supervision="dinov3_local_relational",
                        representation_distillation_weight=8.0,
                    )
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite real TRAIN admission loss")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                encoder_gradients = [p.grad for p in model.encoder.parameters()]
                if not all(g is not None and torch.isfinite(g).all() for g in encoder_gradients):
                    raise ValueError("Encoder gradients absent or nonfinite")
                rows.append(
                    {
                        "arm": arm,
                        "iteration": iteration,
                        "loss_mode": "warmup" if iteration == 0 else "full",
                        "seconds_forward_backward_clip": elapsed,
                        "loss": float(total.detach().float().cpu()),
                        "gradient_norm": float(norm.detach().cpu()),
                        "components": {
                            k: float(v.detach().float().cpu()) for k, v in components.items()
                        },
                        "all_encoder_gradients_finite": True,
                    }
                )
            del model
        atomic_json(
            output / "REAL_MODEL_ADMISSION.json",
            {
                "status": "PASSED",
                "source_sha256": digest(Path(__file__)),
                "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
                "batch_rows": ids,
                "batch_size": 32,
                "precision": "bf16_original_recipe",
                "optimizer_updates": 0,
                "weights_retained_from_admission": False,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "observations": rows,
                "timing_limitation": (
                    "No optimizer or full-population I/O; admission timing is not training ETA."
                ),
            },
        )
    finally:
        data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve())

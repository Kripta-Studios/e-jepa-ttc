"""Zero-update CUDA parity audit on the sealed H fusion training cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion, fusion_loss
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port.train_heads import SealedCache

from .fusion import deterministic_forward


def main() -> None:
    """Compare reference and deterministic output/loss/gradients without optimizer steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    cache_path = run / "features/H/FUSION.npz"
    cache = SealedCache(cache_path, "F_TRUE")
    ids = torch.randint(cache.population, (128,), generator=torch.Generator().manual_seed(7))
    device = torch.device("cuda")
    inputs = tuple(
        cache.tensor(name, ids, device)
        for name in (
            "event_features",
            "event_timing",
            "event_valid",
            "event_expert_phase",
            "rgb_features",
            "rgb_timing",
            "rgb_valid",
        )
    )
    truth = cache.tensor("target_phase", ids, device)
    mass = cache.tensor("mass", ids, device)
    results = {}
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        for mode in ("F_TRUE", "F_ZERO"):
            checkpoint = run / "fits" / mode / "checkpoint_last.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = DualClockFusion(mode).to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            # Exercise nonzero gradients throughout both towers, not only zero-initialized heads.
            torch.manual_seed(7)
            with torch.no_grad():
                model.location.weight.normal_(std=0.01)
                model.width.weight.normal_(std=0.01)
            torch.use_deterministic_algorithms(False)  # Only to evaluate historical reference.
            old = model(*inputs)
            old_loss = fusion_loss(old, truth, mass, cache.population)
            old_loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None
                gradients[name] = parameter.grad.clone()
            model.zero_grad(set_to_none=True)
            torch.use_deterministic_algorithms(True)
            new = deterministic_forward(model, *inputs)
            loss = fusion_loss(new, truth, mass, cache.population)
            loss.backward()
            for key in old:
                torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
            torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
            for name, p in model.named_parameters():
                torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
            assert torch.are_deterministic_algorithms_enabled()
            results[mode] = {
                "output_loss_parameter_gradients_exact": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_tensors": len(gradients),
            }
    finally:
        torch.use_deterministic_algorithms(enabled)
    atomic_write_json(
        args.output,
        {
            "schema": "rgb_port_fusion_cuda_parity_v1",
            "status": "PASSED",
            "optimizer_updates": 0,
            "batch_size": 128,
            "seed": 7,
            "role": "H",
            "cache_sha256": sha256_file(cache_path),
            "device": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()

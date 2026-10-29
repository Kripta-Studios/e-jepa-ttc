"""Zero-update real-P CUDA contract at the complete registered effective batch."""

from __future__ import annotations

import time

import torch

from operational.range_transfer.common import digest, freeze, utc, write

from .common import RUN, config, gpu_charge, require_resources
from .data import SensorDataset
from .qa import subset
from .train import backward, initialize, setup, sources


def run() -> None:
    """Check memory, finite gradients and fixed batch32 without changing weights."""
    binding = {
        "source_hash": digest(sources()),
        "role": "P",
        "indices": list(range(32)),
        "effective_batch": 32,
        "microbatch": 4,
        "optimizer_updates": 0,
        "purpose": "input/runtime/memory contract; no accuracy or recipe selection",
    }
    freeze(RUN / "PREFLIGHT32_REGISTRATION.json", binding)
    require_resources(starting=True)
    dataset = SensorDataset("P", require_complete=False)
    batch = dataset.batch(binding["indices"])
    results = {}
    try:
        for fit_id in config()["fits"]:
            setup(7)
            with gpu_charge("real_P_zero_update_preflight/" + fit_id, reserve=300):
                model = initialize(fit_id).cuda().train()
                torch.cuda.reset_peak_memory_stats()
                batches = [
                    subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                    for i in range(0, 32, 4)
                ]
                torch.cuda.synchronize()
                start = time.perf_counter()
                backward(
                    model, batches, torch.full((32,), 1 / dataset.population), dataset.population
                )
                torch.cuda.synchronize()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                results[fit_id] = {
                    "finite": True,
                    "preclip_gradient_norm": float(norm),
                    "forward_backward_s": time.perf_counter() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "optimizer_updates": 0,
                }
                print(fit_id, results[fit_id], flush=True)
                del model, batches
                torch.cuda.empty_cache()
        write(RUN / "PREFLIGHT32.json", {"utc": utc(), "binding": binding, "results": results})
    finally:
        dataset.close()


if __name__ == "__main__":
    run()

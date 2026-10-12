"""Zero-update comparison of numerically equivalent native sampling runtimes."""

from __future__ import annotations

import time

import torch
from torch.nn import functional

from e_jepa_ttc.models import local_transport
from operational.range_transfer.common import sha, utc, write

from .common import ROOT, RUN, gpu_charge, require_resources
from .data import SensorDataset
from .qa import subset
from .train import backward, initialize, setup


def separable_sample(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Two one-dimensional linear bases and GEMM, without atomic scatter gradients."""
    batch, channels, height, width = field.shape
    x = ((grid[..., 0] + 1) * 0.5 * (width - 1)).flatten(1)
    y = ((grid[..., 1] + 1) * 0.5 * (height - 1)).flatten(1)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    columns = torch.arange(width, device=field.device, dtype=field.dtype)
    rows = torch.arange(height, device=field.device, dtype=field.dtype)
    wx = (columns == x0[..., None]) * (1 - dx[..., None]) + (columns == (x0 + 1)[..., None]) * dx[
        ..., None
    ]
    wy = (rows == y0[..., None]) * (1 - dy[..., None]) + (rows == (y0 + 1)[..., None]) * dy[
        ..., None
    ]
    intermediate = torch.bmm(field.reshape(batch, channels * height, width), wx.transpose(1, 2))
    intermediate = intermediate.reshape(batch, channels, height, -1)
    result = (intermediate * wy.transpose(1, 2)[:, None]).sum(dim=2)
    return result.reshape(batch, channels, *grid.shape[1:3])


def legacy_sample(field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Unmodified kernel, measured only as a reference; not admitted for V18 training."""
    return functional.grid_sample(field, grid, align_corners=True, padding_mode="zeros")


def run() -> None:
    """Profile only private model copies, with no optimizer or evaluation labels."""
    require_resources(starting=True)
    dataset = SensorDataset("P", require_complete=False)
    batch = dataset.batch(list(range(32)))
    original = local_transport.deterministic_bilinear_sample
    functions = {"gather": original, "separable": separable_sample, "legacy": legacy_sample}
    results, gradients = {}, {}
    try:
        with gpu_charge("zero_update_native_sampler_profile", reserve=600):
            for name, sample in functions.items():
                # Only this diagnostic process changes the function reference.
                local_transport.deterministic_bilinear_sample = sample
                setup(7)
                model = initialize("NATIVE-P2-FT-S7").cuda().train()
                batches = [
                    subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                    for i in range(0, 32, 4)
                ]
                timings = []
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(3):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    backward(model, batches, None, dataset.population, recompute=False)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)
                    if repeat == 0:
                        gradients[name] = {
                            n: p.grad.cpu().clone()
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        }
                results[name] = {
                    "seconds": timings,
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved(),
                    "optimizer_updates": 0,
                }
                print(name, results[name], flush=True)
                del model, batches
                torch.cuda.empty_cache()
        for name in ("separable", "legacy"):
            gaps = {
                n: float((v - gradients[name][n]).abs().max())
                for n, v in gradients["gather"].items()
            }
            results[name]["gradient_maxabs_vs_gather"] = max(gaps.values())
        write(
            RUN / "NATIVE_SAMPLER_PROFILE.json",
            {
                "utc": utc(),
                "results": results,
                "recompute": False,
                "scope": "real P32, zero updates; implementation cost, not accuracy selection",
                "source_sha256": sha(ROOT / "operational/joint_measurement/profile_sampling.py"),
            },
        )
    finally:
        local_transport.deterministic_bilinear_sample = original
        dataset.close()


if __name__ == "__main__":
    run()

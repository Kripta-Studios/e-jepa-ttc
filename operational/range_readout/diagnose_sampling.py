"""H-only fixed-batch variance/exposure audit; no optimization or V access."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.range_transfer.cache import HeadCache
from operational.range_transfer.train_heads import loss_on_batch as uniform_loss

from .common import HISTORY, RUN, csv_write, read, sha, write
from .train_heads import initialize, loss_on_batch


def gradient(model: torch.nn.Module, value: torch.Tensor) -> np.ndarray:
    grads = torch.autograd.grad(value, tuple(model.parameters()))
    return torch.cat([g.flatten() for g in grads]).detach().numpy().astype(np.float64)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    binding = read(RUN / "ADMISSION.json")
    cache = HeadCache(Path(binding["cache_path"]))
    model, _, _, _ = initialize(7)
    # Native TemporalRefiner has no stochastic dropout; retain its training mode.
    rows = []
    for mass in (0, 1):
        exact = None
        for begin in range(0, cache.population, 128):
            ids = torch.arange(begin, min(begin + 128, cache.population))
            g = (
                gradient(model, uniform_loss(model, cache, ids, mass, 0))
                * len(ids)
                / cache.population
            )
            exact = g if exact is None else exact + g
        assert exact is not None
        for mode in ("uniform", "importance"):
            rng = torch.Generator().manual_seed(12345)
            p = torch.from_numpy(cache.plans[mass].astype(np.float64))
            p /= p.sum()
            if mode == "uniform":
                p = torch.full_like(p, 1 / cache.population)
            samples = []
            norms = []
            squared_errors = []
            rare = []
            for _ in range(32):
                ids = (
                    torch.randint(cache.population, (128,), generator=rng)
                    if mode == "uniform"
                    else torch.multinomial(p, 128, replacement=True, generator=rng)
                )
                value = (
                    uniform_loss(model, cache, ids, mass, 0)
                    if mode == "uniform"
                    else loss_on_batch(model, cache, ids, mass, 0)
                )
                g = gradient(model, value)
                samples.append(g)
                norms.append(float(np.linalg.norm(g)))
                squared_errors.append(float(np.square(g - exact).sum()))
                rare.append(
                    int(
                        (
                            (cache.target_ttc[ids.numpy()] > 0)
                            & (cache.target_ttc[ids.numpy()] <= 1)
                        ).sum()
                    )
                )
            stack = np.stack(samples)
            rare_mass = float(
                p[torch.from_numpy((cache.target_ttc > 0) & (cache.target_ttc <= 1))].sum()
            )
            rows.append(
                {
                    "mass": f"M{mass}",
                    "sampling": mode,
                    "batches": 32,
                    "seed": 12345,
                    "batch_size": 128,
                    "trace_sample_covariance": float(stack.var(axis=0, ddof=1).sum()),
                    "mean_squared_error_to_exact_H_gradient": float(np.mean(squared_errors)),
                    "exact_H_gradient_l2": float(np.linalg.norm(exact)),
                    "mean_batch_gradient_l2": float(np.mean(norms)),
                    "clip_fraction_at_initialization": float(np.mean(np.asarray(norms) > 1)),
                    "observed_zero_rare_batches": int(np.sum(np.asarray(rare) == 0)),
                    "probability_zero_rare_batch": (1 - rare_mass) ** 128,
                }
            )
    csv_write(RUN / "H_GRADIENT_VARIANCE.csv", rows)
    exposure = []
    clipping = []
    for mass in (0, 1):
        folder = RUN / "fits" / f"S-M{mass}-L0-S7"
        checkpoint = torch.load(
            folder / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        counts = np.asarray(checkpoint["cursor"]["exposure"])
        for group in sorted(set(cache.values["group_id"])):
            for band, mask in {
                "crucial": (cache.target_ttc > 0.1) & (cache.target_ttc <= 3),
                "small": (cache.target_ttc > 3) & (cache.target_ttc <= 6),
                "large": cache.target_ttc > 6,
                "negative": cache.target_ttc < 0,
                "diagnostic_le1": (cache.target_ttc > 0.1) & (cache.target_ttc <= 1),
            }.items():
                selected = mask & (cache.values["group_id"] == group)
                exposure.append(
                    {
                        "fit": folder.name,
                        "group": str(group),
                        "band": band,
                        "rows": int(selected.sum()),
                        "draws": int(counts[selected].sum()),
                        "mass": float(cache.plans[mass][selected].astype(np.float64).sum()),
                    }
                )
        for source in (folder, HISTORY / "fits" / f"RT-M{mass}-L0-S7"):
            frame = pd.read_csv(source / "CURVE.csv")
            clipping.append(
                {
                    "fit": source.name,
                    "updates": len(frame),
                    "fraction_clipped": float((frame.gradient_l2_before_clip > 1).mean()),
                    "last250_fraction_clipped": float(
                        (frame.gradient_l2_before_clip.tail(250) > 1).mean()
                    ),
                }
            )
    csv_write(RUN / "SAMPLER_EXPOSURE.csv", exposure)
    csv_write(RUN / "CLIPPING.csv", clipping)
    write(
        RUN / "SAMPLING_DIAGNOSTICS.json",
        {
            "status": "COMPLETE",
            "role": "H",
            "optimizer_updates": 0,
            "stage": "fixed_initial_seed7_model_before_optimization",
            "gradient_reference": "exact_all_H_risk",
            "batches_per_estimator": 32,
            "seed": 12345,
            "source_sha256": sha(Path(__file__)),
            "warning": "Monte Carlo variance estimate; repeated queries are not independent scenes",
        },
    )


if __name__ == "__main__":
    main()

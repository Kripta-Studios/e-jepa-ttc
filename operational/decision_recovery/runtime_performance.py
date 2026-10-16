"""Lossless whole-batch transfers; enable only after paired native admission."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement


class SensorSource(Protocol):
    def batch(self, indices: list[int]) -> ObjectEventV4Batch: ...
    def counters(self) -> dict: ...


MIN_SPEEDUP_RATIO = 0.98
POLICY = {
    "candidate": "one default-stream B32 transfer, then eight contiguous microbatch views",
    "mass_transfer": "one FP32 B32 transfer, unchanged ordered microbatch reductions",
    "reference": "eight independent microbatch transfers and eight mass transfers",
    "native_trials": 5,
    "warm_trials": 4,
    "minimum_speedup_ratio": MIN_SPEEDUP_RATIO,
    "maximum_peak_device_fraction": 0.90,
    "fallback": "legacy if exact candidate has no measured >=2-percent prepared improvement",
    "no_extra_gpu_prefetch": True,
    "no_extra_compression": True,
}


class BatchDeviceSensors:
    """Preserve the source's CPU prefetch and sampled identities; transfer once."""

    def __init__(self, source: SensorSource, device: str = "cuda") -> None:
        self.source = source
        self.device = device
        self.device_batches = 0
        self.source_wait_s = 0.0
        self.transfer_submit_s = 0.0

    def __getattr__(self, name: str) -> object:
        return getattr(self.source, name)

    def batch(self, indices: list[int]) -> ObjectEventV4Batch:
        import torch

        tick = time.perf_counter()
        batch = self.source.batch(indices)
        ready = time.perf_counter()
        transferred = batch.to(torch.device(self.device), non_blocking=True)
        self.source_wait_s += ready - tick
        self.transfer_submit_s += time.perf_counter() - ready
        self.device_batches += 1
        return transferred

    def counters(self) -> dict:
        values = self.source.counters() if hasattr(self.source, "rows_cached") else {}
        return {
            **values,
            "device_batch_transfer": self.device,
            "device_batches": self.device_batches,
            "source_wait_s": self.source_wait_s,
            "whole_batch_transfer_submit_s": self.transfer_submit_s,
            "interval_scope": "Host intervals; no claim of isolated CUDA kernel time",
        }


def batch_device_backward(
    model: GeometricPairMeasurement, batches: list, weights: Tensor, population: int
) -> dict:
    """Copy all unchanged FP32 weights once; retain the admitted graph and arithmetic."""
    from operational.joint_measurement.captured_phase_loss import backward

    weights = weights.to(device=batches[0].events.device, non_blocking=True)
    return backward(model, batches, weights, population)


def select_transfer_mode(
    trials: list[dict], *, exact_candidate: bool, device_total_bytes: int
) -> dict:
    """Keep legacy unless warm paired evidence shows both speed and memory admission."""
    import statistics

    if len(trials) != POLICY["native_trials"] or not exact_candidate:
        raise ValueError("Complete exact native candidate comparison required")
    valid_times = all(
        0 < row[key] < float("inf") for row in trials for key in ("batch_seconds", "legacy_seconds")
    )
    if not valid_times or device_total_bytes <= 0:
        raise ValueError("Finite positive native timings and device capacity required")
    warm = trials[1:]
    ratio = statistics.median(row["batch_seconds"] / row["legacy_seconds"] for row in warm)
    peak = max(row["batch_peak_allocated_bytes"] for row in trials)
    selected = (
        "batch"
        if ratio <= MIN_SPEEDUP_RATIO
        and peak <= POLICY["maximum_peak_device_fraction"] * device_total_bytes
        else "legacy"
    )
    return {
        "selected_transfer": selected,
        "warm_median_paired_ratio": ratio,
        "candidate_peak_allocated_bytes": peak,
        "device_total_bytes": device_total_bytes,
        "exact_candidate": exact_candidate,
        "trials": trials,
        "scope": "Prepared B32 transfer plus forward/backward; excludes sensor IO and optimizer",
        "end_to_end_speedup_claim": False,
    }


def native_transfer_probe(
    batch: ObjectEventV4Batch,
    weights: Tensor,
    population: int,
    factory: Callable[[], GeometricPairMeasurement],
) -> dict:
    """Compare exact transfers and derivatives on the same native batch, without updates."""
    import gc

    import torch

    from operational.joint_measurement.captured_phase_loss import backward
    from operational.joint_measurement.geometry_capture import contiguous_subset
    from operational.joint_measurement.qa import exact
    from operational.joint_measurement.train import setup

    setup(7)
    models = {name: factory().cuda().train() for name in ("legacy", "batch")}
    initial = {k: v.detach().cpu().clone() for k, v in models["legacy"].state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    trials, checks = [], []
    try:
        for iteration in range(POLICY["native_trials"]):
            observed, row = {}, {"iteration": iteration}
            order = ("legacy", "batch") if iteration % 2 == 0 else ("batch", "legacy")
            row["order"] = list(order)
            for name in order:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                model = models[name]
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                device_batch = batch.to(torch.device("cuda")) if name == "batch" else batch
                batches = [
                    contiguous_subset(device_batch, list(range(i, i + 4))).to(torch.device("cuda"))
                    for i in range(0, 32, 4)
                ]
                loss = (batch_device_backward if name == "batch" else backward)(
                    model, batches, weights, population
                )
                torch.cuda.synchronize()
                row[f"{name}_seconds"] = time.perf_counter() - started
                row[f"{name}_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                observed[name] = {
                    "loss": loss,
                    "gradients": {
                        key: value.grad.detach().cpu().clone()
                        for key, value in model.named_parameters()
                        if value.grad is not None
                    },
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "batch": [
                        (
                            part.events.detach().cpu(),
                            part.delta_t_s.detach().cpu(),
                            part.target_ttc_s.detach().cpu(),
                            part.sample_tokens,
                        )
                        for part in batches
                    ],
                }
                del batches, device_batch
            checks.append(exact(observed["legacy"], observed["batch"]))
            trials.append(row)
        weights_unchanged = all(
            exact(initial, {k: v.detach().cpu() for k, v in model.state_dict().items()})
            for model in models.values()
        )
        result = select_transfer_mode(
            trials,
            exact_candidate=all(checks) and weights_unchanged,
            device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
        result.update(per_trial_exact=checks, weights_unchanged=weights_unchanged)
        return result
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()

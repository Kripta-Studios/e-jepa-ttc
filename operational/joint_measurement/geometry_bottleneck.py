"""Bounded zero-update timing of the admitted C1 execution path at a durable stop."""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import psutil
import torch

from operational.range_transfer.common import read, sha, utc, write

from .common import ROOT, RUN, gpu_charge
from .fast_runtime import PersistentSensors
from .geometry_graph_probe import no_live_training_driver
from .geometry_model import GeometricPairMeasurement
from .qa import subset
from .train import backward, initialize, setup

OUT = RUN / "runtime/C1_bottleneck_20261014"


def run() -> dict:
    """Measure a fixed within-pack cohort, not a new scientific training trajectory."""
    no_live_training_driver()
    origin = read(OUT / "DURABLE_ORIGIN.json")
    checkpoint = OUT / "origin_checkpoint.pt"
    if sha(checkpoint) != origin["checkpoint_sha256"]:
        raise ValueError("C1 profiling origin changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    setup(7)
    dataset = PersistentSensors(limit_bytes=128 * 1024**2)
    ids = list(range(32))
    loader = []
    cpu_profile = cProfile.Profile()
    try:
        dataset.pinned = False
        for number in range(4):
            if number == 3:
                dataset.rows_cached.clear()
                dataset.retained_bytes = 0
            tick = time.perf_counter()
            if number == 3:
                cpu_profile.enable()
            batch = dataset.load(ids)
            if number == 3:
                cpu_profile.disable()
            loader.append({"call": number, "seconds": time.perf_counter() - tick})
        cpu_text = io.StringIO()
        pstats.Stats(cpu_profile, stream=cpu_text).sort_stats("cumulative").print_stats(25)
        (OUT / "CPU_LOAD_PROFILE.txt").write_text(cpu_text.getvalue(), encoding="utf-8")
        cache = dataset.counters()
    finally:
        dataset.shutdown()
    model = GeometricPairMeasurement(initialize("A5-P2-FROZEN"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    weights = torch.full((32,), 1 / 53890)
    with gpu_charge("C1_zero_update_bottleneck_audit", reserve=600):
        model = model.cuda().train()
        batch.events = batch.events.pin_memory()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        cpu_batches = [subset(batch, list(range(i, i + 4))) for i in range(0, 32, 4)]
        subset_s = time.perf_counter() - tick
        tick = time.perf_counter()
        batches = [part.to(torch.device("cuda")) for part in cpu_batches]
        torch.cuda.synchronize()
        transfer_s = time.perf_counter() - tick
        prepared = []
        for number in range(2):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss = backward(model, batches, weights, 53890)
            torch.cuda.synchronize()
            prepared.append({"call": number, "seconds": time.perf_counter() - tick, "loss": loss})
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            backward(model, batches, weights, 53890)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(OUT / "TRACE.json"))
        averages = profile.key_averages()
        (OUT / "CPU_OPERATORS.txt").write_text(
            averages.table(sort_by="self_cpu_time_total", row_limit=30), encoding="utf-8"
        )
        (OUT / "CUDA_OPERATORS.txt").write_text(
            averages.table(sort_by="self_device_time_total", row_limit=30), encoding="utf-8"
        )
        # One extra pass times forward and backward separately with explicit synchronization.
        original = model.forward
        forward_times = []

        def timed_forward(inputs: torch.Tensor, delta_t_s: torch.Tensor) -> dict[str, torch.Tensor]:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            value = original(inputs, delta_t_s)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - tick)
            return value

        model.forward = timed_forward
        model.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        backward(model, batches, weights, 53890)
        torch.cuda.synchronize()
        instrumented_total = time.perf_counter() - tick
        result = {
            "utc": utc(),
            "origin_updates": origin["origin_updates"],
            "checkpoint_sha256": origin["checkpoint_sha256"],
            "optimizer_updates": 0,
            "scope": "32 fixed P rows within pack0; loader cold/warm separately; prepared GPU work",
            "not_measured": "Full-population cache hit rate; optimizer; scientific quality",
            "loader_calls": loader,
            "loader_call_modes": [
                "cold first-use SHA and decode",
                "warm rows",
                "warm rows",
                "decode admitted archive",
            ],
            "cache": cache,
            "parent_events_pinned": batch.events.is_pinned(),
            "subset_events_pinned": [part.events.is_pinned() for part in cpu_batches],
            "CPU_subset_s": subset_s,
            "transfer_s": transfer_s,
            "prepared_forward_backward": prepared,
            "instrumented_forward_s": sum(forward_times),
            "instrumented_total_s": instrumented_total,
            "operator_totals": {
                "cpu_self_s": sum(v.self_cpu_time_total for v in averages) / 1e6,
                "device_self_s": sum(v.self_device_time_total for v in averages) / 1e6,
            },
            "cpu_available_GiB": psutil.virtual_memory().available / 1024**3,
            "source_sha256": sha(ROOT / "operational/joint_measurement/geometry_bottleneck.py"),
            "finite_loss": bool(np.isfinite([prepared[-1]["loss"]["loss"]]).all()),
        }
    write(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    print(run())

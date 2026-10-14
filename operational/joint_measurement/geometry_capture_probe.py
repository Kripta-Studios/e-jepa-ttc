"""Zero-update C1 paired probe for complete autograd-aware CUDA capture."""

from __future__ import annotations

import argparse
import copy
import time
import traceback

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, gpu_charge, require_resources
from .geometry_capture import contiguous_microbatch, install_capture
from .geometry_graph_probe import no_live_training_driver
from .geometry_model import GeometricPairMeasurement
from .geometry_train import REGISTRATION, source_binding
from .qa import exact, fixture, subset
from .resume_fast_proof import close
from .train import backward, committed_sources, initialize, setup

OUT = RUN / "runtime/geometry_capture_v1"
ORIGIN = RUN / "runtime/C1_bottleneck_20261014/origin_checkpoint.pt"


def register() -> dict:
    """Record fixed numerical gates before CUDA results; no scientific update authority."""
    sources = source_binding()
    for name in ("geometry_capture.py", "geometry_capture_probe.py"):
        path = "operational/joint_measurement/" + name
        sources[path] = sha(ROOT / path)
    committed_sources(sources)
    record = {
        "schema": "v18_C1_complete_capture_probe_v1",
        "sources": sources,
        "C1_registration_sha256": sha(REGISTRATION),
        "prior_failure_sha256": sha(RUN / "runtime/geometry_graph_probe_v3/RESULT.json"),
        "origin_checkpoint_sha256": sha(ORIGIN),
        "origin_updates": 247,
        "population": "synthetic8_128x128",
        "seed": 7,
        "effective_batch": 32,
        "microbatch": 4,
        "optimizer_updates": 0,
        "precision": "FP32_no_TF32",
        "capture": "torch.cuda.make_graphed_callables; training phase and validity guard",
        "warmup_backward_calls": 1,
        "paired_timing_rounds": 4,
        "numerical_rtol": 1e-5,
        "numerical_atol": 1e-7,
        "maximum_graph_to_eager_median_ratio": 0.9,
        "require_rng_exact": True,
        "require_weights_unchanged": True,
        "scientific_runtime_admitted": False,
        "passing_means": "Eligible for separate optimizer-trajectory and exact-resume QA",
        "scope": "Prepared GPU forward/loss/backward; excludes input loading and optimizer",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run() -> dict:
    """Use the preserved real C1 weights on synthetic input without an optimizer."""
    record = register()
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    no_live_training_driver()
    require_resources(starting=True)
    with gpu_charge("C1_complete_capture_probe", reserve=900):
        result = {"utc": utc(), "registration_sha256": sha(OUT / "REGISTRATION.json")}
        try:
            setup(7)
            initial = GeometricPairMeasurement(initialize("A5-P2-FROZEN"))
            payload = torch.load(ORIGIN, map_location="cpu", weights_only=False)
            initial.load_state_dict(payload["model_state_dict"], strict=True)
            del payload
            baseline = copy.deepcopy(initial).cuda().train()
            candidate = copy.deepcopy(initial).cuda().train()
            sampler = torch.Generator().manual_seed(7)
            ids = torch.randint(8, (32,), generator=sampler).tolist()
            data = subset(fixture(128), ids)
            data.events = data.events.pin_memory()
            parts = [contiguous_microbatch(data, i, i + 4) for i in range(0, 32, 4)]
            batches = [part.to(torch.device("cuda")) for part in parts]
            core = install_capture(candidate, batches[0].events, batches[0].delta_t_s)
            weights = torch.full((32,), 1 / 8)
            for model in (baseline, candidate):
                model.zero_grad(set_to_none=True)
                backward(model, batches, weights, 8)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            outputs = {}
            for name, model in (("eager", baseline), ("capture", candidate)):
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                model.zero_grad(set_to_none=True)
                losses = backward(model, batches, weights, 8)
                outputs[name] = {
                    "losses": losses,
                    "gradients": {
                        key: p.grad.detach().cpu().clone() if p.grad is not None else None
                        for key, p in model.named_parameters()
                    },
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                }
            checks = {
                key: close(
                    outputs["eager"][key],
                    outputs["capture"][key],
                    rtol=record["numerical_rtol"],
                    atol=record["numerical_atol"],
                )
                for key in ("losses", "gradients")
            }
            checks["rng_exact"] = all(
                exact(outputs["eager"][key], outputs["capture"][key])
                for key in ("cpu_rng", "cuda_rng")
            )
            checks["weights_unchanged"] = all(
                torch.equal(value, model.state_dict()[key].cpu())
                for key, value in initial.state_dict().items()
                for model in (baseline, candidate)
            )
            checks["pinned_microbatch_views"] = all(part.events.is_pinned() for part in parts)
            checks["parameter_identity"] = {
                key: id(p) for key, p in candidate.named_parameters()
            } == {key: id(p) for key, p in core.named_parameters()}
            timings = []
            for number in range(record["paired_timing_rounds"]):
                order = (("eager", baseline), ("capture", candidate))
                for name, model in order if number % 2 == 0 else order[::-1]:
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    backward(model, batches, weights, 8)
                    torch.cuda.synchronize()
                    timings.append({"round": number, "mode": name, "s": time.perf_counter() - tick})
            medians = {
                name: float(np.median([r["s"] for r in timings if r["mode"] == name]))
                for name in ("eager", "capture")
            }
            ratio = medians["capture"] / medians["eager"]
            result.update(
                status="CANDIDATE_FOR_RESUME_QA"
                if all(checks.values()) and 0 < ratio <= 0.9
                else "REJECTED",
                checks=checks,
                timings=timings,
                medians=medians,
                ratio=ratio,
                eager_bitwise={
                    key: exact(outputs["eager"][key], outputs["capture"][key])
                    for key in ("losses", "gradients")
                },
            )
        except Exception:
            result.update(status="REJECTED", traceback=traceback.format_exc())
        result.update(optimizer_updates=0, scientific_runtime_admitted=False)
        freeze(OUT / "RESULT.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())

"""Prospective zero-update C1 graph probe; passing still requires full resume QA.

Never install this optimization in a scientific trainer from probe evidence
alone. Run only after the active RGB driver has exited and before C1 starts.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, gpu_charge, require_resources
from .fast_runtime import install_graph
from .geometry_model import GeometricPairMeasurement
from .geometry_train import REGISTRATION, registered, source_binding
from .qa import exact, fixture, subset
from .resume_fast_proof import close
from .train import backward, committed_sources, initialize, setup

PREVIOUS = RUN / "runtime/geometry_graph_probe_v1"
OUT = RUN / "runtime/geometry_graph_probe_v2"


def register() -> dict:
    """Freeze tolerances and speed gate before looking at any graph comparison."""
    registered()
    if (PREVIOUS / "RESULT.json").exists():
        raise ValueError("Review the prior graph result before any probe revision")
    paths = [
        "geometry_graph_probe.py",
        "resume_fast_proof.py",
        "fast_runtime.py",
    ]
    sources = {
        **source_binding(),
        **{
            f"operational/joint_measurement/{name}": sha(
                ROOT / "operational/joint_measurement" / name
            )
            for name in paths
        },
    }
    committed_sources(sources)
    record = {
        "supersedes_unexecuted_registration_sha256": sha(PREVIOUS / "REGISTRATION.json"),
        "revision_reason": (
            "Recognize the admitted rgb_cached_run GPU owner; numerical recipe unchanged"
        ),
        "sources": sources,
        "C1_registration_sha256": sha(REGISTRATION),
        "population": "synthetic8_128x128",
        "seed": 7,
        "effective_batch": 32,
        "microbatch": 4,
        "optimizer_updates": 0,
        "backend": "cudagraphs",
        "precision": "FP32_no_TF32",
        "warmup_backward_calls": 3,
        "paired_timing_rounds": 6,
        "numerical_rtol": 1e-5,
        "numerical_atol": 1e-7,
        "require_rng_exact": True,
        "require_weights_unchanged": True,
        "require_recorded_graph": True,
        "maximum_graph_to_eager_median_ratio": 0.9,
        "timing_scope": (
            "prepared CUDA effective-batch forward/loss/backward; "
            "excludes optimizer, loader and capture"
        ),
        "passing_means": "eligible for separate optimizer-trajectory and exact-resume QA",
        "scientific_runtime_admitted": False,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def no_live_training_driver() -> None:
    """Do not steal a gap between the active RGB or C1 driver's GPU scopes."""
    current = psutil.Process()
    parent = current.parent()
    own_launcher = (
        parent.pid if parent is not None and parent.cmdline()[1:] == current.cmdline()[1:] else None
    )
    names = {
        "operational.joint_measurement.rgb_run",
        "operational.joint_measurement.rgb_cached_run",
        "operational.joint_measurement.geometry_run",
        "operational.joint_measurement.geometry_capture_run",
    }
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] in {os.getpid(), own_launcher}:
            continue
        if names.intersection(process.info["cmdline"] or []):
            try:
                if Path(process.cwd()).resolve() == ROOT:
                    raise RuntimeError(f"GPU_OWNER_WAIT: live training driver {process.pid}")
            except psutil.NoSuchProcess:
                continue


def decision(checks: dict[str, bool], ratio: float, record: dict) -> str:
    """A numerical probe can enable further QA, never training admission."""
    required = {
        "losses",
        "gradients",
        "rng_exact",
        "actual_cuda_graphs",
        "parameter_identity_unchanged",
        "weights_unchanged",
    }
    if (
        set(checks) == required
        and all(checks.values())
        and np.isfinite(ratio)
        and 0 < ratio <= record["maximum_graph_to_eager_median_ratio"]
    ):
        return "CANDIDATE_FOR_RESUME_QA"
    return "REJECTED"


def run() -> dict:
    """Compare the same seeded initial model without executing optimizer.step."""
    record = register()
    if (OUT / "RESULT.json").exists():
        result = read(OUT / "RESULT.json")
        if result["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
            raise ValueError("C1 graph probe registration changed")
        return result
    no_live_training_driver()
    require_resources(starting=True)
    with gpu_charge("C1_zero_update_graph_probe", reserve=1800):
        result = {
            "utc": utc(),
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "optimizer_updates": 0,
            "scientific_runtime_admitted": False,
        }
        try:
            setup(7)
            initial = GeometricPairMeasurement(initialize("A5-P2-FROZEN"))
            baseline = copy.deepcopy(initial).cuda().train()
            candidate = copy.deepcopy(initial).cuda().train()
            original_parameter_ids = {name: id(p) for name, p in candidate.named_parameters()}
            install_graph(candidate)
            sampler = torch.Generator().manual_seed(7)
            indices = torch.randint(8, (32,), generator=sampler).tolist()
            data = subset(fixture(128), indices)
            batches = [
                subset(data, list(range(i, i + 4))).to(torch.device("cuda"))
                for i in range(0, 32, 4)
            ]
            weights = torch.full((32,), 1 / 8)
            for model in (baseline, candidate):
                for _ in range(record["warmup_backward_calls"]):
                    model.zero_grad(set_to_none=True)
                    backward(model, batches, weights, 8)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            outputs = {}
            for name, model in (("eager", baseline), ("graph", candidate)):
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
            from torch._inductor.cudagraph_trees import get_manager

            manager = get_manager(0, create_if_none_exists=False)
            recorded = manager.graph_counter.__reduce__()[1][0] if manager is not None else 0
            checks = {
                key: close(
                    outputs["eager"][key],
                    outputs["graph"][key],
                    rtol=record["numerical_rtol"],
                    atol=record["numerical_atol"],
                )
                for key in ("losses", "gradients")
            }
            checks.update(
                rng_exact=all(
                    exact(outputs["eager"][key], outputs["graph"][key])
                    for key in ("cpu_rng", "cuda_rng")
                ),
                actual_cuda_graphs=recorded > 0,
                parameter_identity_unchanged=original_parameter_ids
                == {name: id(p) for name, p in candidate.named_parameters()},
            )
            timings = []
            for round_index in range(record["paired_timing_rounds"]):
                order = (("eager", baseline), ("graph", candidate))
                for name, model in order if round_index % 2 == 0 else order[::-1]:
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    backward(model, batches, weights, 8)
                    torch.cuda.synchronize()
                    timings.append(
                        {"round": round_index, "mode": name, "seconds": time.perf_counter() - tick}
                    )
            checks["weights_unchanged"] = all(
                torch.equal(value, model.state_dict()[key].cpu())
                for key, value in initial.state_dict().items()
                for model in (baseline, candidate)
            )
            medians = {
                name: float(np.median([row["seconds"] for row in timings if row["mode"] == name]))
                for name in ("eager", "graph")
            }
            ratio = medians["graph"] / medians["eager"]
            result.update(
                status=decision(checks, ratio, record),
                checks=checks,
                timings=timings,
                graph_to_eager_ratio=ratio,
                medians=medians,
                recorded_graphs=recorded,
                eager_bitwise_equivalence={
                    key: exact(outputs["eager"][key], outputs["graph"][key])
                    for key in ("losses", "gradients")
                },
                loss_values={name: value["losses"] for name, value in outputs.items()},
            )
        except Exception:
            result.update(status="REJECTED", traceback=traceback.format_exc())
        freeze(OUT / "RESULT.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())

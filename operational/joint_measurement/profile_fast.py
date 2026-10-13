"""Bounded optimization admission from the durable current native checkpoint."""

from __future__ import annotations

import argparse
import copy
import gc
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from . import train
from .common import ROOT, RUN, accounting, gpu_charge, require_resources
from .data import SensorDataset
from .fast_runtime import PersistentSensors, batched_backward, install_graph
from .qa import exact, subset

OUT = RUN / "runtime/optimization_20261013"
DIAG = RUN / "diagnostics/fast_runtime_20261013"
FIT = "NATIVE-P2-FT-S7"
STATE_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_generator_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_random_state",
    "python_random_state",
    "cursor",
)


def rng(payload: dict) -> None:
    """Restore every stochastic stream without an optimizer update."""
    torch.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    np.random.set_state(payload["numpy_random_state"])
    random.setstate(payload["python_random_state"])


def micros(batch: object) -> list:
    return [subset(batch, list(range(i, i + 4))).to(torch.device("cuda")) for i in range(0, 32, 4)]


def origin() -> dict:
    """Require exact safe-stop evidence before touching CUDA or private trajectories."""
    stopped = read(RUN / "scheduling/RUNTIME_OPTIMIZATION/STOP_RESULT.json")
    path = RUN / "fits" / FIT / "checkpoint_last.pt"
    if stopped["checkpoint_sha256"] != sha(path):
        raise ValueError("Primary origin changed after safe stop")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["completed_updates"] != stopped["durable_updates"]
        or payload["identity"]["sources"] != train.sources()
    ):
        raise ValueError("Origin cursor or scientific sources differ")
    return payload


def model_from(payload: dict, *, graph: bool = False) -> torch.nn.Module:
    model = train.initialize(FIT).cuda().train()
    model.load_state_dict(payload["model_state_dict"])
    if graph:
        install_graph(model)
    return model


def graph_probe() -> dict:
    """Compare full-batch stochastic losses/gradients, with zero optimizer updates."""
    target = OUT / "GRAPH_PROBE.json"
    if target.exists():
        return read(target)
    payload = origin()
    dataset = PersistentSensors(limit_bytes=512 * 1024**2)
    sampler = torch.Generator().set_state(payload["sampler_generator_state"])
    ids = torch.randint(dataset.population, (32,), generator=sampler).tolist()
    batch = dataset.load(ids)
    result = {
        "utc": utc(),
        "optimizer_updates": 0,
        "ids": ids,
        "origin_updates": payload["completed_updates"],
        "backend": "cudagraphs",
    }
    try:
        with gpu_charge("primary_runtime_zero_update_graph_admission", reserve=1800):
            train.setup(7)
            batches = micros(batch)
            reference = model_from(payload)
            rng(payload)
            baseline_losses = train.backward(reference, batches, None, dataset.population)
            baseline_gradients = {
                k: p.grad.detach().cpu().clone()
                for k, p in reference.named_parameters()
                if p.grad is not None
            }
            baseline_rng = torch.cuda.get_rng_state_all()
            del reference
            gc.collect()
            torch.cuda.empty_cache()
            candidate = model_from(payload, graph=True)
            # Compile/capture at the admitted signature, then erase warmup RNG effects.
            for _ in range(3):
                candidate.zero_grad(set_to_none=True)
                batched_backward(candidate, batches, None, dataset.population)
            candidate.load_state_dict(payload["model_state_dict"])
            rng(payload)
            candidate.zero_grad(set_to_none=True)
            losses = batched_backward(candidate, batches, None, dataset.population)
            gradients = {
                k: p.grad.detach().cpu()
                for k, p in candidate.named_parameters()
                if p.grad is not None
            }
            checks = {
                "losses": exact(baseline_losses, losses),
                "gradients": exact(baseline_gradients, gradients),
                "cuda_rng": exact(baseline_rng, torch.cuda.get_rng_state_all()),
            }
            result.update(
                status="PASS" if all(checks.values()) else "REJECTED",
                checks=checks,
                gradient_maxabs=max(
                    float((v - gradients[k]).abs().max()) for k, v in baseline_gradients.items()
                ),
            )
    except Exception:
        result.update(status="REJECTED", traceback=traceback.format_exc())
    finally:
        dataset.shutdown()
    write(target, result)
    print(result, flush=True)
    return result


def trajectory(mode: str, limit: int, payload: dict, *, graph: bool) -> dict:
    """Compare sixteen steps with eight+restore+eight on a private current-state copy."""
    require_resources(starting=True)
    train.setup(7)
    fast = mode == "fast"
    dataset = PersistentSensors() if fast else SensorDataset("P")
    directory = DIAG / mode
    identity = {
        "fit_id": "FAST_RUNTIME_DIAG-" + mode,
        "updates": 16,
        "origin": read(RUN / "scheduling/RUNTIME_OPTIMIZATION/STOP_RESULT.json")[
            "checkpoint_sha256"
        ],
        "registration": sha(OUT / "REGISTRATION.json"),
        "graph": graph if fast else False,
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    state = ProducerCheckpoint(directory, identity, 16)
    timings = []
    with gpu_charge("primary_runtime_real_P_admission/" + mode, reserve=900):
        model = model_from(payload, graph=graph and fast)
        optimizer, scheduler = train.make_optimizer(model, updates=12000)
        sampler = torch.Generator()
        if state.pointer.exists():
            curve = state.restore(model, optimizer, scheduler, sampler)["curve"]
        else:
            optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
            scheduler.load_state_dict(copy.deepcopy(payload["scheduler_state_dict"]))
            sampler.set_state(payload["sampler_generator_state"])
            rng(payload)
            curve = []
            state.save(model, optimizer, scheduler, sampler, {"curve": curve}, status="RUNNING")
        if graph and fast:
            saved = torch.load(state.path, map_location="cpu", weights_only=False)
            shadow = torch.Generator().set_state(sampler.get_state())
            ids = torch.randint(dataset.population, (32,), generator=shadow).tolist()
            batches = micros(dataset.load(ids))
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                batched_backward(model, batches, None, dataset.population)
            state.restore(model, optimizer, scheduler, sampler)
            del saved, batches
        if fast:
            dataset.prepare(sampler.get_state())
        try:
            while state.completed < limit:
                tick = time.perf_counter()
                ids = torch.randint(dataset.population, (32,), generator=sampler).tolist()
                batch = dataset.batch(ids)
                loaded = time.perf_counter()
                batches = micros(batch)
                optimizer.zero_grad(set_to_none=True)
                losses = (batched_backward if fast else train.backward)(
                    model, batches, None, dataset.population
                )
                backward_done = time.perf_counter()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                state.begin()
                optimizer.step()
                scheduler.step()
                state.commit()
                curve.append({"ids": ids, "losses": losses, "norm": float(norm)})
                torch.cuda.synchronize()
                timings.append(
                    {
                        "update": state.completed,
                        "seconds": time.perf_counter() - tick,
                        "loader_s": loaded - tick,
                        "h2d_backward_s": backward_done - loaded,
                    }
                )
            state.save(
                model,
                optimizer,
                scheduler,
                sampler,
                {"curve": curve},
                status="COMPLETE" if limit == 16 else "PAUSED",
            )
        finally:
            if fast:
                counters = dataset.counters()
                dataset.shutdown()
            else:
                counters = {"verified_packs": len(dataset.verified)}
                dataset.close()
    write(directory / f"TIMING_TO_{limit:02d}.json", {"timings": timings, "cache": counters})
    accounting()
    return torch.load(state.path, map_location="cpu", weights_only=False)


def validate() -> dict:
    """Keep all failed evidence; admit only exact state and faster observed steps."""
    if (OUT / "ADMISSION.json").exists():
        return read(OUT / "ADMISSION.json")
    probe = read(OUT / "GRAPH_PROBE.json")
    graph = probe["status"] == "PASS"
    payload = origin()
    baseline = trajectory("serial", 16, payload, graph=False)
    trajectory("fast", 8, payload, graph=graph)
    candidate = trajectory("fast", 16, payload, graph=graph)
    checks = {key: exact(baseline[key], candidate[key]) for key in STATE_KEYS}
    timing = {
        mode: [
            row
            for p in sorted((DIAG / mode).glob("TIMING_TO_*.json"))
            for row in read(p)["timings"]
        ]
        for mode in ("serial", "fast")
    }
    assert all(len(rows) == 16 for rows in timing.values())
    means = {mode: sum(row["seconds"] for row in rows) / len(rows) for mode, rows in timing.items()}
    result = {
        "utc": utc(),
        "status": "PASS"
        if all(checks.values()) and means["fast"] < means["serial"]
        else "REJECTED",
        "checks": checks,
        "graph": graph,
        "optimizer_updates": 32,
        "mean_seconds": means,
        "ratio": means["fast"] / means["serial"],
        "timing": timing,
        "origin_updates": payload["completed_updates"],
        "runtime_sources": {
            p.name: sha(p)
            for p in (Path(__file__), ROOT / "operational/joint_measurement/fast_runtime.py")
        },
        "scope": (
            "16 future real-P updates per private trajectory; cold input costs retained; "
            "no primary updates consumed or lost"
        ),
    }
    write(OUT / "ADMISSION.json", result)
    print({k: v for k, v in result.items() if k != "timing"}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["graph", "validate"])
    args = parser.parse_args()
    print(graph_probe() if args.action == "graph" else validate())

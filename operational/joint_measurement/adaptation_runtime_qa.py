"""Prospective exact-data, loss, persistent-graph and resume admission for A1 runtime."""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from dataclasses import fields

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, write

from . import adaptation_depth as a1
from .common import ROOT, RUN, accounting, config, gpu_charge
from .compute_guard import guard_idle
from .correspondence_model import step_backward
from .fast_runtime import PersistentSensors
from .geometry_capture import contiguous_subset
from .geometry_capture_qa import STATE_KEYS
from .parallel_training_sensor import ParallelTrainingSensors
from .persistent_adaptation import Session
from .phase_transport_loss import backward
from .qa import exact
from .train import committed_sources

OUT = RUN / "runtime/A1_parallel_persistent_v1"
QA = RUN / "qa/A1_parallel_persistent_v1"


def sources() -> dict:
    return {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "parallel_training_sensor",
            "prefetched_sensor",
            "phase_transport_loss",
            "persistent_adaptation",
            "adaptation_runtime_qa",
            "adaptation_runtime",
        )
    }


def register() -> dict:
    """Fix exact gates and technical updates before benchmark or optimizer execution."""
    a1.registered()
    committed_sources(sources())
    stopped = read(OUT / "QUIESCED.json")
    if stopped["checkpoint_sha256"] != sha(OUT / "ORIGIN.pt"):
        raise ValueError("Original A1 checkpoint witness changed")
    record = {
        "sources": sources(),
        "scientific_registration_sha256": sha(a1.REGISTRATION),
        "original_QA_sha256": sha(a1.QA / "RESULT.json"),
        "origin_sha256": stopped["checkpoint_sha256"],
        "origin_updates": 1086,
        "cpu": (
            "12 next native P batches from saved sampler plus duplicate draw; "
            "two alternating-order passes, no row cache"
        ),
        "loader_gate": (
            "All batch fields bitwise exact; second-pass median paired parallel/serial <=0.85"
        ),
        "gpu": (
            "Both arms: original captured4, persistent2+2, persistent2 then fresh-session resume2;"
            "24 physical technical updates"
        ),
        "gpu_gate": (
            "Exact model, Adam, scheduler, sampler, all RNG, loss/gradient norm/ID curves; "
            "only wall timings excluded"
        ),
        "persistent_graph_gate": (
            "Same parameter identities and one capture across both bounded scopes"
        ),
        "native_gpu": (
            "Next real P B32 at saved1086 weights/RNG: original versus combined loss, exact "
            "losses/gradients/RNG, zero optimizer updates, median warm ratio<=1.10"
        ),
        "physical_technical_updates": 24,
        "scientific_updates_from_QA": 0,
        "loss": (
            "Identical FP32 ordered arithmetic; combine guards before backward and transfer all "
            "per-microbatch diagnostics once; original Python-double aggregation order"
        ),
        "preserved": (
            "Weights, complete optimizer/scheduler/sampler/RNG/cursor, original journal, source "
            "hashes and update accounting; no precision or recipe change"
        ),
        "runtime": (
            "One fresh child per arm; retain model, CUDA graph and bounded sensor cache "
            "between max900s charged scopes"
        ),
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def check_registration() -> dict:
    record = read(OUT / "REGISTRATION.json")
    if record["sources"] != sources() or record["scientific_registration_sha256"] != sha(
        a1.REGISTRATION
    ):
        raise ValueError("Runtime admission sources changed")
    a1.registered()
    return record


def cpu() -> dict:
    """Measure exact random P batch transport independently of CUDA training."""
    guard_idle()
    check_registration()
    torch.set_num_threads(1)
    payload = torch.load(OUT / "ORIGIN.pt", map_location="cpu", weights_only=False)
    serial = PersistentSensors(limit_bytes=0)
    parallel = ParallelTrainingSensors(limit_bytes=0, workers=8)
    sampler = torch.Generator().set_state(payload["sampler_generator_state"])
    batches = [
        torch.randint(serial.population, (32,), generator=sampler).tolist() for _ in range(12)
    ]
    batches.append([batches[0][0], batches[0][0], *batches[0][2:]])
    before_rng = torch.get_rng_state().clone()
    measurements = []
    try:
        for iteration in range(2):
            for number, ids in enumerate(batches):
                order = (("serial", serial), ("parallel", parallel))
                if (number + iteration) % 2:
                    order = order[::-1]
                outputs, elapsed = {}, {}
                for name, dataset in order:
                    begin = time.perf_counter()
                    outputs[name] = dataset.load(ids)
                    elapsed[name] = time.perf_counter() - begin
                checks = {}
                for field in fields(outputs["serial"]):
                    left = getattr(outputs["serial"], field.name)
                    right = getattr(outputs["parallel"], field.name)
                    checks[field.name] = (
                        left.dtype == right.dtype and torch.equal(left, right)
                        if isinstance(left, torch.Tensor)
                        else left == right
                    )
                if not all(checks.values()):
                    raise ValueError(f"Native transport parity failed: {checks}")
                measurements.append(
                    {
                        "pass": iteration,
                        "batch": number,
                        "ids": ids,
                        "seconds": elapsed,
                        "checks": checks,
                        "order": [x[0] for x in order],
                    }
                )
                write(OUT / "CPU_PROGRESS.json", {"completed": len(measurements), "total": 26})
        ratios = [
            r["seconds"]["parallel"] / r["seconds"]["serial"]
            for r in measurements
            if r["pass"] == 1
        ]
        ratio = statistics.median(ratios)
        checks = {
            "all_fields_exact": True,
            "global_rng_unchanged": torch.equal(before_rng, torch.get_rng_state()),
            "cuda_uninitialized": not torch.cuda.is_initialized(),
            "speed_gate": ratio <= 0.85,
        }
        result = {
            "status": "PASS" if all(checks.values()) else "REJECTED",
            "checks": checks,
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "paired_median_ratio": ratio,
            "warm_pass_median_seconds": {
                name: statistics.median(r["seconds"][name] for r in measurements if r["pass"] == 1)
                for name in ("serial", "parallel")
            },
            "measurements": measurements,
            "optimizer_updates": 0,
            "scope": (
                "Quiescent CPU B32 random reads, no decoded-row caching; pass0 includes "
                "first-use SHA, pass1 uses admitted archives; OS cache uncontrolled"
            ),
        }
        freeze(OUT / "CPU.json", result)
        return result
    finally:
        serial.shutdown()
        parallel.shutdown()


def gpu(mode: str) -> dict:
    """Run one arm in a fresh CUDA process; no scientific checkpoint is mutated."""
    guard_idle()
    check_registration()
    if read(OUT / "CPU.json")["status"] != "PASS":
        raise ValueError("Native CPU transport must pass first")
    if accounting()["technical"] + 12 > config()["caps"]["technical"]:
        raise RuntimeError("Technical update reserve exhausted")
    target = QA / mode
    if (target / "RESULT.json").exists():
        raise ValueError("Do not repeat completed QA updates")
    masses = np.full(8, 1 / 8)
    with gpu_charge(f"A1_persistent_QA_{mode}", reserve=900) as deadline:
        with a1.engine_context():
            a1.engine.segment(
                target / "original",
                a1.geometry_train.SyntheticSensors(),
                masses,
                mode,
                2,
                limit=4,
                deadline=deadline,
                capture=True,
                synthetic=True,
            )
        gc.collect()
        torch.cuda.empty_cache()
        sensors = a1.geometry_train.SyntheticSensors()
        session = Session(target / "persistent", mode, masses, synthetic=True)
        session.advance(sensors, limit=2, deadline=deadline)
        parameter_ids = [id(p) for p in session.model.parameters()]
        session.advance(sensors, limit=4, deadline=deadline)
        retained = session.capture_count == 1 and parameter_ids == [
            id(p) for p in session.model.parameters()
        ]
        del session
        gc.collect()
        torch.cuda.empty_cache()
        session = Session(target / "resumed", mode, masses, synthetic=True)
        session.advance(sensors, limit=2, deadline=deadline)
        del session
        gc.collect()
        torch.cuda.empty_cache()
        session = Session(target / "resumed", mode, masses, synthetic=True)
        session.advance(sensors, limit=4, deadline=deadline)
        del session
        sensors.close()
    payloads = [
        torch.load(target / name / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        for name in ("original", "persistent", "resumed")
    ]
    checks = {}
    for name, other in zip(("persistent", "resumed"), payloads[1:], strict=True):
        checks[name] = {key: exact(payloads[0][key], other[key]) for key in STATE_KEYS}
        checks[name]["curve"] = exact(payloads[0]["cursor"]["curve"], other["cursor"]["curve"])
    passed = retained and all(all(v.values()) for v in checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "same_parameters_one_capture_two_scopes": retained,
        "physical_technical_updates": 12,
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
    }
    freeze(target / "RESULT.json", result)
    if not passed:
        raise ValueError("Persistent runtime must match original and resume exactly")
    return result


def native() -> dict:
    """Compare original and optimized derivatives at the actual saved scientific checkpoint."""
    guard_idle()
    check_registration()
    if read(OUT / "CPU.json")["status"] != "PASS":
        raise ValueError("CPU data admission required")
    payload = torch.load(OUT / "ORIGIN.pt", map_location="cpu", weights_only=False)
    sensors = ParallelTrainingSensors(workers=8, limit_bytes=0)
    try:
        sampler = torch.Generator().set_state(payload["sampler_generator_state"])
        ids = torch.randint(sensors.population, (32,), generator=sampler).tolist()
        batch = sensors.load(ids)
        from e_jepa_ttc.joint_measurement.loss import training_mass

        masses = training_mass(
            sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
        )
        checks, timings = [], []
        with gpu_charge("A1_native_loss_transport_zero_updates", reserve=900):
            a1.engine.setup(7)
            batches = [
                contiguous_subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                for i in range(0, 32, 4)
            ]
            models = {
                name: a1.make_model("LAST-BLOCK")[0].cuda().train()
                for name in ("original", "candidate")
            }
            for model in models.values():
                model.load_state_dict(payload["model_state_dict"])
            for iteration in range(4):
                order = (
                    ("original", "candidate") if iteration % 2 == 0 else ("candidate", "original")
                )
                observed = {}
                row = {"iteration": iteration, "order": order, "seconds": {}}
                for name in order:
                    torch.set_rng_state(payload["torch_rng_state"])
                    torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
                    np.random.set_state(payload["numpy_random_state"])
                    random.setstate(payload["python_random_state"])
                    model = models[name]
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    weights = torch.from_numpy(masses[ids]).float()
                    losses = (
                        step_backward(
                            model,
                            batches,
                            weights,
                            sensors.population,
                            correspondence=False,
                            capture=True,
                        )
                        if name == "original"
                        else backward(model, batches, weights, sensors.population)
                    )
                    torch.cuda.synchronize()
                    row["seconds"][name] = time.perf_counter() - start
                    observed[name] = {
                        "loss": losses,
                        "gradients": {
                            n: p.grad.detach().cpu().clone()
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        },
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    }
                check = {
                    key: exact(observed["original"][key], observed["candidate"][key])
                    for key in observed["original"]
                }
                checks.append(check)
                timings.append(row)
                if not all(check.values()):
                    break
            ratio = (
                statistics.median(
                    row["seconds"]["candidate"] / row["seconds"]["original"] for row in timings[1:]
                )
                if len(timings) > 1
                else float("inf")
            )
        result = {
            "status": "PASS"
            if len(checks) == 4 and all(all(c.values()) for c in checks) and ratio <= 1.10
            else "FAIL",
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "checks": checks,
            "origin_updates": payload["completed_updates"],
            "ids": ids,
            "optimizer_updates": 0,
            "timings": timings,
            "median_warm_ratio": ratio,
            "scope": "Prepared real P B32 forward/backward; not end-to-end production throughput",
        }
        freeze(OUT / "NATIVE.json", result)
        return result
    finally:
        sensors.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "cpu", "gpu", "native"))
    parser.add_argument("--arm", choices=a1.MODES, default=a1.MODES[0])
    args = parser.parse_args()
    result = (
        gpu(args.arm)
        if args.action == "gpu"
        else {"register": register, "cpu": cpu, "native": native}[args.action]()
    )
    print({key: result.get(key) for key in ("status", "checks", "paired_median_ratio")}, flush=True)

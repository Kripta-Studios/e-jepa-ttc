"""One registered, bounded real-P diagnostic: serial10 versus prefetched5+restore5."""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc
from operational.rgb_port.train_producers import ProducerCheckpoint

from .common import ROOT, RUN, accounting, config, gpu_charge, require_resources
from .data import SensorDataset
from .prefetch import UniformPrefetch
from .qa import exact, subset
from .train import backward, initialize, make_optimizer, setup, sources

DIRECTORY = RUN / "diagnostics/prefetch_runtime"


def registration() -> dict:
    """Freeze the sole real-P optimization diagnostic before any of its updates."""
    record = {
        "experiment_id": "V18_INPUT_PREFETCH_P_DIAGNOSTIC",
        "registration_revision": 4,
        "purpose": "Execution equivalence, exact resume and step timing; no H/V model selection",
        "population": "P",
        "fits": config()["fits"],
        "updates_per_fit": {"serial": 10, "prefetch": 10},
        "maximum_physical_updates": 80,
        "counts_as": "scientific_diagnostic_within_single_P_preflight_max200",
        "initialization": "Private admitted P-only parent, seed7, same new modules",
        "warmup": "one forward/backward on first batch, zero optimizer updates, both paths",
        "optimizer_schedule": "unchanged registered 12000-update schedule; run stops after ten",
        "comparison": "serial10 versus prefetched5+checkpoint_restore+5",
        "acceptance": "all model/optimizer/scheduler/RNG/sampler/cursor fields bit exact",
        "timing_scope": (
            "batch consumption, H2D, backward, gradient clipping, optimizer and journal"
        ),
        "resource_checks": (
            "Before each trajectory of at most ten updates; outside prefetch context"
        ),
        "performance_gate": "native mean measured step time at least10% lower; all paths reported",
        "sources": {
            **sources(),
            "operational/joint_measurement/prefetch.py": sha(
                ROOT / "operational/joint_measurement/prefetch.py"
            ),
            "operational/joint_measurement/profile_prefetch.py": sha(Path(__file__)),
        },
    }
    freeze(DIRECTORY / "REGISTRATION.json", record)
    return record


def trajectory(
    fit_id: str, mode: str, limit: int, dataset: SensorDataset, masses: np.ndarray
) -> dict:
    """Run only the declared ten-step private trajectory, retaining all physical work."""
    require_resources()
    setup(7)
    folder = DIRECTORY / fit_id / mode
    identity = {
        "fit_id": f"PREFETCH_DIAG-{fit_id}-{mode}",
        "registration": sha(DIRECTORY / "REGISTRATION.json"),
        "seed": 7,
        "updates": 10,
        "mode": mode,
    }
    identity["identity_sha256"] = digest(identity)
    freeze(folder / "IDENTITY.json", identity)
    model = initialize(fit_id).cuda().train()
    optimizer, scheduler = make_optimizer(model, updates=config()["updates"])
    sampler = torch.Generator().manual_seed(7)
    state = ProducerCheckpoint(folder, identity, 10)
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, sampler)
        accounting()
        curve = cursor["curve"]
    else:
        # Both trajectories receive the same zero-update warmup before their initial state.
        probe = torch.Generator().set_state(sampler.get_state())
        ids = torch.randint(dataset.population, (32,), generator=probe).tolist()
        batch = dataset.batch(ids)
        micros = [
            subset(batch, list(range(i, i + 4))).to(torch.device("cuda")) for i in range(0, 32, 4)
        ]
        backward(model, micros, torch.from_numpy(masses[ids]).float(), dataset.population)
        optimizer.zero_grad(set_to_none=True)
        curve = []
        state.save(model, optimizer, scheduler, sampler, {"curve": curve}, status="RUNNING")
    timings = []
    # Retain verified pack hashes, but reset archive-handle locality for each trajectory segment.
    dataset.close()
    with (
        UniformPrefetch(dataset.batch, dataset.population, 32, sampler)
        if mode == "prefetch"
        else nullcontext()
    ) as loader:
        while state.completed < limit:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            if loader is None:
                ids = torch.randint(dataset.population, (32,), generator=sampler).tolist()
                batch = dataset.batch(ids)
            else:
                ids, batch = loader.next()
            micros = [
                subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                for i in range(0, 32, 4)
            ]
            optimizer.zero_grad(set_to_none=True)
            loss = backward(
                model, micros, torch.from_numpy(masses[ids]).float(), dataset.population
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            state.begin()
            optimizer.step()
            scheduler.step()
            state.commit()
            torch.cuda.synchronize()
            timings.append({"update": state.completed, "seconds": time.perf_counter() - tick})
            curve.append({"ids": ids, "losses": loss, "norm": float(norm)})
            if state.completed % 5 == 0:
                state.save(
                    model,
                    optimizer,
                    scheduler,
                    sampler,
                    {"curve": curve},
                    status="COMPLETE" if state.completed == 10 else "PAUSED",
                )
    path = folder / f"TIMING_TO_{state.completed:02d}.csv"
    if timings:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["update", "seconds"])
            writer.writeheader()
            writer.writerows(timings)
    accounting()
    return torch.load(folder / "checkpoint_last.pt", map_location="cpu", weights_only=False)


def run() -> dict:
    """Execute exactly eighty scientific diagnostic updates once, with separate weights."""
    registration()
    if (DIRECTORY / "COMPLETE.json").exists():
        return read(DIRECTORY / "COMPLETE.json")
    require_resources(starting=True)
    dataset = SensorDataset("P")
    masses = training_mass(
        dataset.rows.target_ttc.to_numpy(), dataset.rows.group_id.to_numpy(), role="P"
    )
    try:
        # Admit identical sensor pages before timing either implementation.
        probe = torch.Generator().manual_seed(7)
        for _ in range(11):
            dataset.batch(torch.randint(dataset.population, (32,), generator=probe).tolist())
        checks = {}
        for fit_id in config()["fits"]:
            receipt = DIRECTORY / fit_id / "COMPARISON.json"
            if receipt.exists():
                checks[fit_id] = read(receipt)
                if checks[fit_id]["status"] != "PASS":
                    raise AssertionError(f"Retained failed diagnostic requires review: {fit_id}")
                continue
            with gpu_charge("real_P_prefetch_diagnostic/" + fit_id, reserve=600):
                control = trajectory(fit_id, "serial", 10, dataset, masses)
                trajectory(fit_id, "prefetch", 5, dataset, masses)
                candidate = trajectory(fit_id, "prefetch", 10, dataset, masses)
                keys = [
                    "model_state_dict",
                    "optimizer_state_dict",
                    "scheduler_state_dict",
                    "sampler_generator_state",
                    "torch_rng_state",
                    "cuda_rng_state_all",
                    "numpy_random_state",
                    "python_random_state",
                    "cursor",
                ]
                equality = {key: exact(control[key], candidate[key]) for key in keys}
            times = {}
            for mode in ("serial", "prefetch"):
                measured = []
                for path in sorted((DIRECTORY / fit_id / mode).glob("TIMING_TO_*.csv")):
                    with path.open(newline="") as stream:
                        measured.extend(float(row["seconds"]) for row in csv.DictReader(stream))
                if len(measured) != 10:
                    raise ValueError("Incomplete timing evidence; do not adopt the optimization")
                times[mode] = {"steps": len(measured), "mean_s": float(np.mean(measured))}
            comparison = {
                "status": "PASS" if all(equality.values()) else "FAIL",
                "exact": equality,
                "times": times,
                "prefetch_to_serial_mean_ratio": times["prefetch"]["mean_s"]
                / times["serial"]["mean_s"],
            }
            freeze(receipt, comparison)
            checks[fit_id] = comparison
            if comparison["status"] != "PASS":
                raise AssertionError(f"Prefetch equivalence differs: {fit_id}")
        result = {
            "status": "PASS",
            "checks": checks,
            "optimizer_updates": 80,
            "native_performance_gate": checks["NATIVE-P2-FT-S7"]["prefetch_to_serial_mean_ratio"]
            <= 0.9,
            "installed_in_primary": False,
            "utc": utc(),
        }
        freeze(DIRECTORY / "COMPLETE.json", result)
        return result
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(registration() if parser.parse_args().register_only else run())

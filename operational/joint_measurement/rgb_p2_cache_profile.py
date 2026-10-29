"""Prospective CPU comparison of current versus compact native RGB shard residency."""

from __future__ import annotations

import gc
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import psutil
import torch

from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, require_resources
from .qa import exact
from .rgb_cached_data import CachedRGBPairDataset
from .rgb_p2_cached_data import P2CachedRGBPairDataset
from .rgb_train import REGISTRATION, registered
from .train import committed_sources

OUT = RUN / "runtime/rgb_p2_profile_v1"


def register() -> dict:
    """Fix order, workload, numerical gate and speed threshold before measurement."""
    recipe = registered()
    paths = (
        "rgb_p2_cache_profile.py",
        "rgb_p2_cached_data.py",
        "rgb_cached_data.py",
        "rgb_data.py",
    )
    sources = {
        f"operational/joint_measurement/{name}": sha(ROOT / "operational/joint_measurement" / name)
        for name in paths
    }
    sources["src/e_jepa_ttc/rgb_port/data.py"] = sha(ROOT / "src/e_jepa_ttc/rgb_port/data.py")
    committed_sources(sources)
    generator = torch.Generator().manual_seed(20261014)
    population = recipe["input_identity"]["population"]
    record = {
        "sources": sources,
        "RGB_registration_sha256": sha(REGISTRATION),
        "input_identity": recipe["input_identity"],
        "warmup_batches": 128,
        "measured_batches": 16,
        "batch_size": 32,
        "seed": 20261014,
        "ids": [torch.randint(population, (32,), generator=generator).tolist() for _ in range(144)],
        "cache_limit_bytes": 6 * 1024**3,
        "maximum_median_ratio": 0.9,
        "order": ["current6GiB", "compact6GiB"],
        "order_limitation": (
            "Sequential phases to bound RAM; second phase can benefit from OS cache"
        ),
        "required": ["all measured batch fields exact", "CPU RNG unchanged", "CUDA uninitialized"],
        "scope": "CPU input decode/collation, not full training throughput",
        "optimizer_updates": 0,
        "scientific_runtime_admitted": False,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def no_live_driver() -> None:
    """Require a saved pause because two resident input caches need private CPU headroom."""
    names = {
        "operational.joint_measurement.rgb_run",
        "operational.joint_measurement.rgb_cached_run",
        "operational.joint_measurement.geometry_run",
    }
    for process in psutil.process_iter(["pid", "cmdline"]):
        if names.intersection(process.info["cmdline"] or []):
            try:
                if Path(process.cwd()).resolve() == ROOT:
                    raise RuntimeError(f"RESOURCE_WAIT: live scientific driver {process.pid}")
            except psutil.NoSuchProcess:
                pass


def run() -> dict:
    """Measure identical fixed P batches sequentially, retaining exact reference tensors."""
    record = register()
    if (OUT / "RESULT.json").exists():
        result = read(OUT / "RESULT.json")
        if result["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
            raise ValueError("Compact RGB profile registration differs")
        return result
    no_live_driver()
    require_resources(starting=True)
    if torch.cuda.is_initialized():
        raise ValueError("CPU profile must not initialize CUDA")
    rng = torch.get_rng_state().clone()
    references, measurements, counters, checks = [], [], {}, []
    started = time.perf_counter()
    for name, factory in (
        ("current6GiB", CachedRGBPairDataset),
        ("compact6GiB", P2CachedRGBPairDataset),
    ):
        dataset = factory("P", limit_bytes=record["cache_limit_bytes"])
        try:
            for index, ids in enumerate(record["ids"]):
                if index % 16 == 0:
                    require_resources()
                tick = time.perf_counter()
                batch = dataset.batch(ids)
                elapsed = time.perf_counter() - tick
                warmup = index < record["warmup_batches"]
                measurements.append(
                    {"mode": name, "batch": index, "warmup": warmup, "seconds": elapsed}
                )
                if not warmup:
                    if name == "current6GiB":
                        references.append(batch)
                    else:
                        reference = references[index - record["warmup_batches"]]
                        checks.append(
                            {
                                field.name: exact(
                                    getattr(reference, field.name), getattr(batch, field.name)
                                )
                                for field in fields(batch)
                            }
                        )
                if (index + 1) % 32 == 0:
                    print(f"{name} {index + 1}/{len(record['ids'])}", flush=True)
            counters[name] = dataset.counters()
        finally:
            dataset.shutdown()
            del dataset
            gc.collect()
    medians = {
        name: float(
            np.median(
                [
                    row["seconds"]
                    for row in measurements
                    if row["mode"] == name and not row["warmup"]
                ]
            )
        )
        for name in counters
    }
    ratio = medians["compact6GiB"] / medians["current6GiB"]
    rng_exact = torch.equal(rng, torch.get_rng_state())
    passed = all(all(check.values()) for check in checks) and len(checks) == 16 and rng_exact
    passed = passed and not torch.cuda.is_initialized() and ratio <= record["maximum_median_ratio"]
    result = {
        "status": "CANDIDATE_FOR_DURABLE_RUNTIME_TRANSITION" if passed else "REJECTED",
        "utc": utc(),
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "measurements": measurements,
        "checks": checks,
        "counters": counters,
        "median_seconds": medians,
        "compact_to_current_ratio": ratio,
        "cpu_rng_exact": rng_exact,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
        "wall_seconds": time.perf_counter() - started,
        "limitations": [record["scope"], record["order_limitation"]],
        "scientific_runtime_admitted": False,
    }
    freeze(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register()) if args.register_only else print(run())

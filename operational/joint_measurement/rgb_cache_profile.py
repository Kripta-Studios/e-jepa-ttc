"""CPU-only paired lossless RGB loader measurement, without touching the live trainer."""

from __future__ import annotations

import time
from dataclasses import fields

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, require_resources
from .qa import exact
from .rgb_cached_data import CachedRGBPairDataset
from .rgb_data import RGBPairDataset
from .rgb_train import REGISTRATION, registered
from .train import committed_sources

OUT = RUN / "runtime/rgb_input_cache_profile_v1"


def run() -> dict:
    """Warm eight batches, then compare eight new paired batches and every native field."""
    require_resources(starting=True)
    record = registered()
    paths = ["rgb_cache_profile.py", "rgb_cached_data.py", "rgb_data.py"]
    sources = {
        f"operational/joint_measurement/{name}": sha(ROOT / "operational/joint_measurement" / name)
        for name in paths
    }
    sources["src/e_jepa_ttc/rgb_port/data.py"] = sha(ROOT / "src/e_jepa_ttc/rgb_port/data.py")
    commit = committed_sources(sources)
    population = record["input_identity"]["population"]
    sampler = torch.Generator().manual_seed(20261014)
    ids = [torch.randint(population, (32,), generator=sampler).tolist() for _ in range(16)]
    registration = {
        "sources": sources,
        "RGB_registration_sha256": sha(REGISTRATION),
        "input_identity": record["input_identity"],
        "seed": 20261014,
        "ids": ids,
        "warmup_batches": 8,
        "measured_batches": 8,
        "cache_limit_bytes": 6 * 1024**3,
        "baseline_limit_bytes": 256 * 1024**2,
        "scope": "CPU sensor batch load/collation only",
        "all_batch_fields_exact_required": True,
        "optimizer_updates": 0,
        "GPU_work": False,
        "maximum_loader_ratio": 0.85,
        "scientific_runtime_admitted": False,
    }
    freeze(OUT / "REGISTRATION.json", registration)
    if (OUT / "RESULT.json").exists():
        result = read(OUT / "RESULT.json")
        if result["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
            raise ValueError("RGB loader profile registration changed")
        return result
    cpu_rng = torch.get_rng_state().clone()
    baseline = RGBPairDataset("P")
    candidate = CachedRGBPairDataset("P", limit_bytes=registration["cache_limit_bytes"])
    measurements, checks = [], []
    try:
        for index, indices in enumerate(ids):
            require_resources()
            order = (("baseline", baseline), ("cached", candidate))
            values, elapsed = {}, {}
            for name, dataset in order if index % 2 == 0 else order[::-1]:
                tick = time.perf_counter()
                values[name] = dataset.batch(indices)
                elapsed[name] = time.perf_counter() - tick
            check = {
                field.name: exact(
                    getattr(values["baseline"], field.name), getattr(values["cached"], field.name)
                )
                for field in fields(values["baseline"])
            }
            checks.append(check)
            measurements.append({"batch": index, "warmup": index < 8, "seconds": elapsed})
            print(
                {
                    "batch": index + 1,
                    "all_fields_exact": all(check.values()),
                    "seconds": elapsed,
                    "cache": candidate.counters(),
                },
                flush=True,
            )
            if not all(check.values()):
                raise ValueError("Lossless RGB residency changed native batch fields")
        medians = {
            name: float(
                np.median([row["seconds"][name] for row in measurements if not row["warmup"]])
            )
            for name in ("baseline", "cached")
        }
        ratio = medians["cached"] / medians["baseline"]
        result = {
            "status": "CANDIDATE_FOR_DURABLE_RUNTIME_TRANSITION"
            if ratio <= 0.85
            else "NO_LOADER_GAIN",
            "utc": utc(),
            "git_commit": commit,
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "measurements": measurements,
            "checks": checks,
            "all_fields_exact": True,
            "cpu_rng_exact": torch.equal(cpu_rng, torch.get_rng_state()),
            "median_seconds": medians,
            "cached_to_baseline_ratio": ratio,
            "cache": candidate.counters(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "optimizer_updates": 0,
            "scientific_runtime_admitted": False,
            "limitation": "Loader-only timing under concurrent training; not training throughput",
        }
        if not result["cpu_rng_exact"] or result["cuda_initialized"]:
            raise ValueError("CPU loader test changed RNG or initialized CUDA")
        freeze(OUT / "RESULT.json", result)
        return result
    finally:
        candidate.shutdown()
        baseline.close()


if __name__ == "__main__":
    print(run())

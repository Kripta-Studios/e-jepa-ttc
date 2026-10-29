"""Preregistered eighty-update real-P storage equivalence and resume diagnostic."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc

from . import profile_prefetch
from .common import ROOT, RUN, config, gpu_charge, require_resources
from .data import SensorDataset
from .mapped_data import mapped_root
from .mapped_runtime import MappedSensorDataset, admit_mapped
from .qa import exact
from .train import sources

DIRECTORY = RUN / "diagnostics/mapped_runtime"


def registration() -> dict:
    """Register exact-state and speed requirements before any diagnostic updates."""
    extra = ("mapped_data.py", "mapped_runtime.py", "profile_mapped.py", "profile_prefetch.py")
    record = {
        "experiment_id": "V18_UNCOMPRESSED_INPUT_DIAGNOSTIC",
        "population": "P",
        "seed": 7,
        "fits": config()["fits"],
        "updates_per_fit": {"serial": 10, "mapped": 10},
        "maximum_physical_updates": 80,
        "counts_as": "scientific diagnostic; cumulative input diagnostics 160 of preflight max200",
        "purpose": "Storage optimization only; no H/V or checkpoint selection",
        "comparison": "Original serial10 versus uncompressed5 + full restore +5",
        "warmup": "Identical zero-update forward/backward before initial state",
        "order": "Alternate which trajectory runs first across registered fit order",
        "data_admission": (
            "Complete P byte hashes before GPU scope; all eleven sampled batches exact"
        ),
        "acceptance": (
            "All weights, optimizer, scheduler, random states and sample/loss cursor bit exact"
        ),
        "speed_gate": {"each_fit_ratio_max": 0.9, "arithmetic_mean_fit_ratio_max": 0.8},
        "timing_scope": (
            "Batch read, H2D, backward, clipping, optimizer, scheduler and update journal"
        ),
        "not_measured": "Entire cold campaign preparation or final end-to-end inference",
        "scientific_sources": sources(),
        "runtime_sources": {
            f"operational/joint_measurement/{name}": sha(
                ROOT / "operational/joint_measurement" / name
            )
            for name in extra
        },
        "new_cache_binding": sha(mapped_root() / "BINDING.json"),
        "installed_in_primary": False,
    }
    freeze(DIRECTORY / "REGISTRATION.json", record)
    return record


def run() -> dict:
    """Retain all diagnostic outcomes and never modify primary checkpoint files."""
    registration()
    if (DIRECTORY / "COMPLETE.json").exists():
        return read(DIRECTORY / "COMPLETE.json")
    require_resources(starting=True)
    reader = admit_mapped()
    original = SensorDataset("P")
    mapped = MappedSensorDataset("P", admission=reader)
    masses = training_mass(
        original.rows.target_ttc.to_numpy(), original.rows.group_id.to_numpy(), role="P"
    )
    try:
        probe = torch.Generator().manual_seed(7)
        for _ in range(11):
            ids = torch.randint(original.population, (32,), generator=probe).tolist()
            left, right = original.batch(ids), mapped.batch(ids)
            if not all(exact(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)):
                raise AssertionError("Storage adapters produce different real-P batches")
        freeze(
            DIRECTORY / "INPUT_PARITY.json",
            {
                "batches": 11,
                "rows_with_duplicates": 352,
                "all_fields_bit_exact": True,
                "complete_cache_sha256": sha(mapped_root() / "COMPLETE.json"),
            },
        )
        checks = {}
        with patch.object(profile_prefetch, "DIRECTORY", DIRECTORY):
            for number, fit_id in enumerate(config()["fits"]):
                path = DIRECTORY / fit_id / "COMPARISON.json"
                if path.exists():
                    checks[fit_id] = read(path)
                    if checks[fit_id]["status"] != "PASS":
                        raise AssertionError("Retained storage diagnostic failure requires review")
                    continue
                order = ("serial", "mapped") if number % 2 == 0 else ("mapped", "serial")
                endpoints = {}
                with gpu_charge("real_P_mapped_diagnostic/" + fit_id, reserve=600):
                    for mode in order:
                        dataset = original if mode == "serial" else mapped
                        if mode == "mapped":
                            profile_prefetch.trajectory(fit_id, mode, 5, dataset, masses)
                        endpoints[mode] = profile_prefetch.trajectory(
                            fit_id, mode, 10, dataset, masses
                        )
                    keys = (
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
                    equality = {
                        key: exact(endpoints["serial"][key], endpoints["mapped"][key])
                        for key in keys
                    }
                timing = {}
                for mode in ("serial", "mapped"):
                    measured = []
                    for csv_path in sorted((DIRECTORY / fit_id / mode).glob("TIMING_TO_*.csv")):
                        with csv_path.open(newline="") as stream:
                            measured.extend(float(row["seconds"]) for row in csv.DictReader(stream))
                    if len(measured) != 10:
                        raise ValueError("Incomplete timing; refuse optimization installation")
                    timing[mode] = float(np.mean(measured))
                comparison = {
                    "status": "PASS" if all(equality.values()) else "FAIL",
                    "exact": equality,
                    "order": order,
                    "mean_seconds": timing,
                    "mapped_to_serial_ratio": timing["mapped"] / timing["serial"],
                }
                freeze(path, comparison)
                checks[fit_id] = comparison
                if comparison["status"] != "PASS":
                    raise AssertionError("Storage adapter changed training or exact resume")
        ratios = [r["mapped_to_serial_ratio"] for r in checks.values()]
        result = {
            "status": "PASS",
            "checks": checks,
            "optimizer_updates": 80,
            "performance_gate_pass": all(r <= 0.9 for r in ratios)
            and float(np.mean(ratios)) <= 0.8,
            "installed_in_primary": False,
            "registration_sha256": sha(DIRECTORY / "REGISTRATION.json"),
            "input_parity_sha256": sha(DIRECTORY / "INPUT_PARITY.json"),
            "scientific_source_hash": digest(sources()),
            "utc": utc(),
        }
        freeze(DIRECTORY / "COMPLETE.json", result)
        return result
    finally:
        original.close()
        mapped.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(registration() if parser.parse_args().register_only else run())

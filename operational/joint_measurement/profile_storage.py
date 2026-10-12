"""Bounded CPU-only comparison of identical sensor packs on USB and local NVMe."""

from __future__ import annotations

import shutil
import time
from dataclasses import fields

import numpy as np

from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, require_resources
from .data import SensorDataset, cache_root
from .qa import exact


def run() -> dict:
    """Compare eight interleaved P batches; never construct an optimizer or CUDA model."""
    directory = RUN / "diagnostics/NVME_STORAGE_PROBE"
    target = ROOT / "data/cache/v18_nvme_storage_probe"
    generator = np.random.default_rng(20261012)
    packs = [0, 70, 140, 210]
    batches = []
    for _ in range(8):
        indices = []
        for number in packs:
            end = min((number + 1) * 256, 53890)
            indices.extend(
                generator.choice(np.arange(number * 256, end), 8, replace=False).tolist()
            )
        generator.shuffle(indices)
        batches.append(indices)
    orders = [["usb", "nvme"] if i % 2 == 0 else ["nvme", "usb"] for i in range(8)]
    registration = {
        "schema": "v18_cpu_storage_probe_v1",
        "source_sha256": sha(ROOT / "operational/joint_measurement/profile_storage.py"),
        "source_root": str(cache_root("P")),
        "target_root": str(target),
        "packs": packs,
        "batches": batches,
        "orders": orders,
        "optimizer_updates": 0,
        "gpu_work": False,
        "gate": {"mean_time_ratio_max": 0.75, "paired_ratio_max": 0.9, "min_pairs": 6},
        "scope": (
            "Warm CPU loader only; four packs, same data and original loader; concurrent trainer"
        ),
        "migration": "not automatic; requires complete-copy admission and a durable transition",
    }
    freeze(directory / "REGISTRATION.json", registration)
    if (directory / "RESULT.json").exists():
        return read(directory / "RESULT.json")
    require_resources(starting=True)
    total = sum((cache_root("P") / f"pack_{number:05d}.zip").stat().st_size for number in packs)
    if shutil.disk_usage(ROOT).free - total < 11_000_000_000:
        raise ValueError("Insufficient local NVMe reserve for bounded storage probe")
    target.mkdir(parents=True, exist_ok=True)
    admitted = {}
    for number in packs:
        name = f"pack_{number:05d}.zip"
        source, destination = cache_root("P") / name, target / name
        receipt = read(source.with_suffix(".json"))
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha(destination) != receipt["sha256"]:
            raise ValueError("Copied sensor pack differs; retain it for investigation")
        if not destination.with_suffix(".json").exists():
            shutil.copyfile(source.with_suffix(".json"), destination.with_suffix(".json"))
        if read(destination.with_suffix(".json")) != receipt:
            raise ValueError("Copied sensor receipt differs")
        admitted[name] = {"bytes": destination.stat().st_size, "sha256": receipt["sha256"]}
    freeze(directory / "COPY_ADMISSION.json", {"packs": admitted, "total_bytes": total})
    datasets = {name: SensorDataset("P") for name in ("usb", "nvme")}
    datasets["nvme"].folder = target
    try:
        # Both use the original full-file admission before the timed comparison.
        for dataset in datasets.values():
            dataset.batch(batches[0])
        timings = []
        for indices, order in zip(batches, orders, strict=True):
            elapsed, output = {}, {}
            for name in order:
                tick = time.perf_counter()
                output[name] = datasets[name].batch(indices)
                elapsed[name] = time.perf_counter() - tick
            checks = {
                field.name: exact(
                    getattr(output["usb"], field.name), getattr(output["nvme"], field.name)
                )
                for field in fields(output["usb"])
            }
            if not all(checks.values()):
                raise ValueError("NVMe and USB loader inputs differ")
            timings.append({"indices": indices, "seconds": elapsed, "fields_bit_exact": checks})
            del output
        ratios = [row["seconds"]["nvme"] / row["seconds"]["usb"] for row in timings]
        ratio = sum(row["seconds"]["nvme"] for row in timings) / sum(
            row["seconds"]["usb"] for row in timings
        )
        result = {
            "status": "PASS",
            "utc": utc(),
            "timings": timings,
            "paired_ratios": ratios,
            "mean_time_ratio": ratio,
            "performance_gate_pass": ratio <= 0.75 and sum(value <= 0.9 for value in ratios) >= 6,
            "input_parity": "all batch fields bit exact",
            "optimizer_updates": 0,
            "production_cache_changed": False,
            "not_end_to_end_training_timing": True,
        }
        freeze(directory / "RESULT.json", result)
        return result
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    result = run()
    print({key: value for key, value in result.items() if key != "timings"}, flush=True)

"""Finite CPU-only test of uncompressed row access, with exact field parity."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc, write

from .common import ROOT, RUN, require_resources
from .data import SensorDataset


def main() -> dict:
    """Measure a fixed small P cohort without changing the production loader."""
    torch.set_num_threads(1)
    directory = RUN / "diagnostics/UNCOMPRESSED_ROW_PROBE"
    destination = ROOT / "data/cache/v18_uncompressed_row_probe"
    indices = [pack * 256 + offset for pack in (0, 70, 140, 210) for offset in range(32)]
    registration = {
        "schema": "v18_uncompressed_row_probe_v1",
        "indices": indices,
        "pairs": 8,
        "batch_size": 32,
        "seed": 20261012,
        "source_sha256": sha(Path(__file__)),
        "original_loader_sha256": sha(ROOT / "operational/joint_measurement/data.py"),
        "gate": {"mean_ratio_max": 0.5, "paired_ratio_max": 0.75, "required_pairs": 6},
        "scope": "warm CPU row reads; concurrent primary trainer; not end-to-end training",
        "optimizer_updates": 0,
        "production_changed": False,
    }
    freeze(directory / "REGISTRATION.json", registration)
    if (directory / "RESULT.json").exists():
        return read(directory / "RESULT.json")
    require_resources(disk_reserve=400_000_000)
    if destination.exists():
        raise ValueError("Probe destination already exists; inspect incomplete evidence first")
    destination.mkdir(parents=True)
    dataset = SensorDataset("P")
    mapping = {index: position for position, index in enumerate(indices)}
    arrays = {}
    try:
        records = [dataset._row(index) for index in indices]
        for key in records[0]:
            values = np.stack([row[key] for row in records])
            path = destination / f"{key}.npy"
            np.save(path, values, allow_pickle=False)
            arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        for index in indices:
            position = mapping[index]
            if not all(
                np.array_equal(arrays[k][position], v) for k, v in records[position].items()
            ):
                raise ValueError("Uncompressed serialization changed a field")
        del records
        admission = {
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(destination.glob("*.npy"))
        }
        freeze(directory / "CACHE_ADMISSION.json", admission)
        generator = np.random.default_rng(registration["seed"])
        pairs = []
        for trial in range(registration["pairs"]):
            selected = generator.choice(
                indices, size=registration["batch_size"], replace=True
            ).tolist()
            positions = [mapping[i] for i in selected]
            outputs, elapsed = {}, {}
            order = ("zip", "uncompressed") if trial % 2 == 0 else ("uncompressed", "zip")
            for mode in order:
                started = time.perf_counter()
                if mode == "zip":
                    rows = [dataset._row(i) for i in selected]
                    outputs[mode] = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
                else:
                    outputs[mode] = {
                        key: np.array(value[positions], copy=True) for key, value in arrays.items()
                    }
                elapsed[mode] = time.perf_counter() - started
            if not all(
                np.array_equal(outputs["zip"][key], outputs["uncompressed"][key])
                for key in outputs["zip"]
            ):
                raise ValueError("Paired batch mismatch")
            pairs.append(
                {
                    "trial": trial,
                    "indices": selected,
                    "order": order,
                    "seconds": elapsed,
                    "ratio": elapsed["uncompressed"] / elapsed["zip"],
                    "all_fields_exact": True,
                }
            )
            write(directory / "PROGRESS.json", {"utc": utc(), "pairs": pairs})
        ratio = sum(p["seconds"]["uncompressed"] for p in pairs) / sum(
            p["seconds"]["zip"] for p in pairs
        )
        gate = ratio <= 0.5 and sum(p["ratio"] <= 0.75 for p in pairs) >= 6
        result = {
            "utc": utc(),
            "pairs": pairs,
            "mean_ratio": ratio,
            "performance_gate_pass": gate,
            "all_fields_exact": True,
            "optimizer_updates": 0,
            "production_changed": False,
            "next": "Further integration and exact resume QA required"
            if gate
            else "Do not install",
            "scope": registration["scope"],
        }
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU probe unexpectedly initialized CUDA")
        write(directory / "RESULT.json", result)
        return result
    finally:
        arrays.clear()
        dataset.close()


if __name__ == "__main__":
    print(json.dumps(main()))

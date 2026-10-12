"""Measure CPU forward only on portable prepared inputs, with an explicit scope."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.export_weights import load_export


def main(folder: Path) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    for directory in sorted(folder.iterdir()):
        if not directory.is_dir():
            continue
        metadata = read(directory / "MODEL_METADATA.json")
        receipt = read(directory / "RECEIPT.json")
        if sha(directory / "WEIGHTS.pt") != receipt["weights_sha256"]:
            raise ValueError("Portable weights changed")
        if sha(directory / "EXAMPLE_INPUT.npz") != receipt["example_input_sha256"]:
            raise ValueError("Prepared input changed")
        model = load_export(directory / "WEIGHTS.pt")
        with np.load(directory / "EXAMPLE_INPUT.npz", allow_pickle=False) as archive:
            values = {
                key: torch.from_numpy(archive[key][:1].copy()) for key in archive.files
            }
        if any(key.startswith(("target_", "dino_")) for key in values):
            raise ValueError("Training target in inference example")
        durations = []
        with torch.inference_mode():
            for iteration in range(220):
                started = time.perf_counter_ns()
                if metadata["input_style"] == "positional":
                    model(*(values[key] for key in metadata["input_order"]))
                else:
                    model(values)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if iteration >= 20:
                    durations.append(elapsed)
        rows.append(
            {
                "fit": directory.name,
                "kind": metadata["kind"],
                "batch": 1,
                "cpu_threads": 1,
                "warmup": 20,
                "measured": 200,
                "median_ms": float(np.median(durations)),
                "p95_ms": float(np.quantile(durations, 0.95)),
                "weights_bytes": (directory / "WEIGHTS.pt").stat().st_size,
                "weights_sha256": receipt["weights_sha256"],
                "durations_ms": durations,
            }
        )
    scope = {
        "scope": "CPU batch1 forward over prepared portable inputs",
        "excluded": [
            "raw event reading and representations",
            "encoder and transport extraction",
            "history construction and normalization",
            "detector or tracker",
            "model loading and input file loading",
        ],
        "end_to_end_latency": False,
        "optimizer_updates": 0,
        "device": "cpu",
        "dtype": "float32",
        "host": platform.node(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "source_sha256": sha(Path(__file__)),
        "warning": "Shared-host wall timings; no guarantee for other hardware or software",
        "models": rows,
    }
    write(folder.parent / "PORTABLE_CPU_FORWARD_PROFILE.json", scope)
    pd.DataFrame([{k: v for k, v in row.items() if k != "durations_ms"} for row in rows]).to_csv(
        folder.parent / "PORTABLE_CPU_FORWARD_PROFILE.csv", index=False
    )
    print(f"Profiled {len(rows)} portable models; CPU prepared-input forward only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    main(parser.parse_args().folder)

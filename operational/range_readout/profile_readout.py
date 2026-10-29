"""Warm CPU A5-only raw forward cost; excludes C2F, PAIR, H8 and sensor IO."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from operational.rgb_port.infer_experts import delta_matrix
from operational.rgb_port.train_producers import make_event_producer_source

from .cache_readout import parent
from .common import OWNER, RUN, read, sha, utc, write
from .train_readout import load_endpoint


def main() -> None:
    """Measure the same one P input, three models, no optimizer and no V labels."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    source = make_event_producer_source(OWNER / "P_MANIFEST.json", event_source_root=source_root)
    source._inputs.limit = 128 * 1024**2
    batch = source.batch([0], "event")
    events = batch.events.float()
    delta = delta_matrix(batch.delta_t_s, events.shape[1])
    native, _ = parent()
    models = {"historical_A5": native}
    hashes = {"historical_A5": sha(OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt")}
    for bound in (0.05, 0.20):
        name = f"R-C{round(bound * 100):02d}-S7"
        path = RUN / "fits" / name / "checkpoint_last.pt"
        models[name] = load_endpoint(path, ResidualSpec(bound)).model
        hashes[name] = sha(path)
    durations: dict[str, list[float]] = {name: [] for name in models}
    with torch.inference_mode():
        for model in models.values():
            for _ in range(3):
                model(events, delta)
        # Interleave order to avoid assigning all time drift to one endpoint.
        for repeat in range(30):
            names = list(models)
            names = names[repeat % 3 :] + names[: repeat % 3]
            for name in names:
                started = time.perf_counter()
                output = models[name](events, delta)
                elapsed = time.perf_counter() - started
                if not torch.isfinite(output.ttc_mean_seconds).all():
                    raise ValueError("Nonfinite raw producer output during profiling")
                durations[name].append(elapsed * 1000)
    write(
        RUN / "A5_RAW_FORWARD_PROFILE.json",
        {
            "utc": utc(),
            "scope": "CPU_FP32_A5_raw_forward_only_batch1_T3_warm",
            "end_to_end": False,
            "full_system": False,
            "excludes": ["sensor_reading", "voxel_preparation", "C2F", "PAIR", "H8", "history"],
            "host_shared_with_other_processes": True,
            "numeric_threads": 1,
            "P_query_index": 0,
            "input_shape": list(events.shape),
            "warmups_each": 3,
            "repeats_each": 30,
            "checkpoint_sha256": hashes,
            "optimizer_updates": 0,
            "results": {
                name: {
                    "samples_ms": values,
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.quantile(values, 0.95)),
                }
                for name, values in durations.items()
            },
        },
    )


if __name__ == "__main__":
    main()

"""Portable P2 TTC exports for both adapted replica encoders, with original-forward parity."""

from __future__ import annotations

import torch

from operational.range_transfer.common import freeze, sha

from .common import ROOT, require_resources
from .data import SensorDataset
from .fusion_replication_pipeline import DIRECTORY, freeze_all, producer
from .fusion_replication_train import SEEDS
from .geometry_export import trace_portable
from .train import committed_sources


def export() -> dict:
    """Keep real P examples, complete weights and signed output checks for each producer."""
    endpoints = freeze_all()
    require_resources()
    torch.set_num_threads(1)
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in ("fusion_replication_export", "geometry_export", "geometry_model")
    }
    committed_sources(sources)
    indices = [0, 17, 83, 159, 255]
    source = SensorDataset("P")
    try:
        examples = []
        for index in indices:
            batch = source.batch([index])
            examples.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        results = {}
        for seed in SEEDS:
            model, proof = producer(seed)
            binding = {
                "seed": seed,
                "endpoint_sha256": endpoints["producers"][str(seed)],
                "proof": proof,
                "P_rows": indices,
                "rtol": 1e-5,
                "atol_seconds": 1e-5,
                "scope": "Native prepared P2 pixels to signed TTC; not RGB fusion or raw preprocessing",
                "sources": sources,
            }
            results[str(seed)] = trace_portable(
                DIRECTORY / "exports" / f"C1-S{seed}", model, examples, binding
            )
        freeze(DIRECTORY / "exports/PRODUCER_EXPORTS.json", results)
        return results
    finally:
        source.close()


if __name__ == "__main__":
    print(export())

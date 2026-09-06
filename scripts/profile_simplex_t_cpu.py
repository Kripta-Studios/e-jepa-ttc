"""Accounted 500-update synthetic CPU resource profile; no scientific evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import fit


class ProfileSource:
    """Generated inputs for the largest registered GRU projection; no real rows."""

    population = 256

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(918)
        self.features = torch.randn(256, 8, 145, generator=generator)
        self.experts = 0.025 + 0.01 * torch.randn(256, 3, generator=generator)
        self.truth = self.experts.median(-1).values + 0.009 * torch.tanh(self.features[:, -3, 0])
        self.identity_sha256 = hashlib.sha256(
            self.features.numpy().tobytes()
            + self.experts.numpy().tobytes()
            + self.truth.numpy().tobytes()
        ).hexdigest()

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        size = len(query_ids)
        timing = torch.zeros(size, 8, 4)
        timing[:, :, :2] = torch.arange(7, -1, -1)[None, :, None] * 0.1
        timing[:, 1:, 2] = 0.1
        return (
            self.features[query_ids],
            timing,
            torch.ones(size, 8, dtype=torch.bool),
            self.experts[query_ids],
            self.truth[query_ids],
            torch.full((size,), 1 / self.population),
        )


def main() -> None:
    """Profile one immutable output directory and retain exact resource evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(
        args.output / "RESERVATION.json",
        {
            "scope": "SYNTHETIC_CPU_PROFILE_ONLY",
            "reserved_technical_updates": 500,
            "previous_technical_updates": 85,
            "technical_cap": 1000,
            "registered_scientific_updates": 0,
        },
    )
    started = time.perf_counter()
    observations = []

    def resource_ok() -> bool:
        snapshot = admitted([args.output.resolve()])
        snapshot["elapsed_seconds"] = time.perf_counter() - started
        observations.append(snapshot)
        return bool(snapshot["has_headroom"])

    result = fit(
        ProfileSource(),
        TemporalConfig(feature_count=145, hidden=160),
        args.output,
        seed=7,
        freeze_sha256=hashlib.sha256(b"synthetic_resource_profile_only_v1").hexdigest(),
        resource_ok=resource_ok,
        stop_after=500,
    )
    elapsed = time.perf_counter() - started
    write_new_json(
        args.output / "RESOURCE_PROFILE.json",
        {
            "scope": "SYNTHETIC_CPU_PROFILE_ONLY",
            "result": result,
            "elapsed_seconds": elapsed,
            "physical_batch": 128,
            "history": 8,
            "features": 145,
            "hidden": 160,
            "device": "cpu",
            "dtype": "float32",
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "peak_process_tree_rss_bytes": max(
                o.get("process_tree_rss_bytes", 0) for o in observations
            ),
            "minimum_host_available_bytes": min(
                o.get("host_available_bytes", 0) for o in observations
            ),
            "observations": observations,
            "scientific_fits": 0,
            "total_technical_updates": 85 + result["completed_updates"],
            "scientific_recipe_selected_from_scores": False,
        },
    )
    print(
        json.dumps(
            {
                "completed_updates": result["completed_updates"],
                "elapsed_seconds": elapsed,
                "status": result["status"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

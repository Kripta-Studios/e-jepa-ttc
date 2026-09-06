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
from e_jepa_ttc.simplex_t.lifecycle import TechnicalBudget, admitted
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
    if args.output.exists():
        raise FileExistsError("profile output already exists; inspect its exact saved state")
    budget_path = Path(__file__).resolve().parents[1] / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    if not budget_path.is_file():
        raise FileNotFoundError("reconcile historical technical receipts before profiling")
    budget = TechnicalBudget(budget_path).reserve("cpu_profile_500", 500)
    previous_updates = sum(budget["reservations"].values()) - 500
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(
        args.output / "RESERVATION.json",
        {
            "scope": "SYNTHETIC_CPU_PROFILE_ONLY",
            "reserved_technical_updates": 500,
            "previous_reserved_technical_updates": previous_updates,
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
            "profile_executed_updates": result["completed_updates"],
            "total_reserved_technical_updates": previous_updates + 500,
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

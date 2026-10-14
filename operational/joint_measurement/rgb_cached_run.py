"""Resume the registered RGB trainer with admitted persistent native input shards."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from . import rgb_run, rgb_train
from .common import ROOT, RUN
from .qa import exact
from .rgb_cached_data import CachedRGBPairDataset
from .train import committed_sources

OUT = RUN / "runtime/rgb_resident_v1"
PROFILE = RUN / "runtime/rgb_input_cache_profile_v1"


def cpu(value: object) -> object:
    """Copy nested tensors to CPU without changing dtype or RNG."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu(item) for item in value)
    return value


class VerifiedCheckpoint(ProducerCheckpoint):
    """Witness the original restore before the unchanged trainer can take an update."""

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        generator: torch.Generator,
    ) -> dict:
        version, pointer = self._recover_version()
        payload = torch.load(version, map_location="cpu", weights_only=False)
        cursor = super().restore(model, optimizer, scheduler, generator)
        actual = {
            "model_state_dict": cpu(model.state_dict()),
            "optimizer_state_dict": cpu(optimizer.state_dict()),
            "scheduler_state_dict": scheduler.state_dict(),
            "sampler_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if payload["cuda_rng_state_all"] is not None
                else None
            ),
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "cursor": cursor,
        }
        checks = {key: exact(value, payload[key]) for key, value in actual.items()}
        checks["durable_cursor"] = self.completed == self.durable == pointer["completed_updates"]
        checks["no_new_recovery"] = self.recovery_upper == payload["recovery_upper"]
        result = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "fit_id": self.identity["fit_id"],
            "updates": self.completed,
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "optimizer_updates_in_check": 0,
            "utc": utc(),
        }
        write(OUT / "restores" / f"{self.identity['fit_id']}_{self.completed}.json", result)
        if result["status"] != "PASS":
            raise ValueError("Resident RGB runtime full-state restore failed")
        return cursor


def admission() -> dict:
    """Admit unchanged native numerics from the prospective paired input profile."""
    record = rgb_train.registered()
    profile = read(PROFILE / "RESULT.json")
    registration = read(PROFILE / "REGISTRATION.json")
    if (
        profile["registration_sha256"] != sha(PROFILE / "REGISTRATION.json")
        or profile["status"] != "CANDIDATE_FOR_DURABLE_RUNTIME_TRANSITION"
        or not profile["cpu_rng_exact"]
        or profile["optimizer_updates"] != 0
        or registration["RGB_registration_sha256"] != sha(rgb_train.REGISTRATION)
        or registration["input_identity"] != record["input_identity"]
    ):
        raise ValueError("Native RGB input profile admission failed")
    for path, expected in registration["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Profiled RGB loader source changed")
    sources = {
        f"operational/joint_measurement/{name}": sha(ROOT / "operational/joint_measurement" / name)
        for name in ("rgb_cached_run.py", "rgb_cached_data.py", "rgb_run.py")
    }
    sources["operational/rgb_port/train_producers.py"] = sha(
        ROOT / "operational/rgb_port/train_producers.py"
    )
    committed_sources(sources)
    binding = {
        "sources": sources,
        "RGB_registration_sha256": sha(rgb_train.REGISTRATION),
        "profile_sha256": sha(PROFILE / "RESULT.json"),
        "cache_limit_bytes": registration["cache_limit_bytes"],
        "numerical_operations": "Unchanged native decode, collation, model, backward and optimizer",
        "checkpoint": "Original ProducerCheckpoint plus exact state witness before updates",
        "scope": "Training input residency only; evaluation uses original data route",
        "cuda_graph": False,
        "new_optimizer_qa_updates": 0,
    }
    freeze(OUT / "REGISTRATION.json", binding)
    return binding


def run(*, once: bool = False) -> None:
    """Use the existing sole-writer lock and charged scopes, retaining only private RAM."""
    binding = admission()
    rgb_run.require_authorization()
    original_dataset = rgb_train.RGBPairDataset
    original_checkpoint = rgb_train.ProducerCheckpoint
    original_fit = rgb_run.fit
    dataset = None

    def factory(role: str) -> CachedRGBPairDataset:
        nonlocal dataset
        if role != "P":
            raise ValueError("Resident training adapter is P-only")
        if dataset is None:
            dataset = CachedRGBPairDataset(role, limit_bytes=binding["cache_limit_bytes"])
        return dataset

    def fit(fit_id: str) -> dict:
        nonlocal dataset
        receipt = original_fit(fit_id)
        if dataset is not None:
            write(
                OUT / "CACHE_STATE.json",
                {
                    "utc": utc(),
                    "pid": os.getpid(),
                    "fit_id": fit_id,
                    "updates": receipt["completed_updates"],
                    **dataset.counters(),
                },
            )
            if receipt["status"] == "COMPLETE":
                dataset.shutdown()
                dataset = None
        return receipt

    rgb_train.RGBPairDataset = factory
    rgb_train.ProducerCheckpoint = VerifiedCheckpoint
    rgb_run.fit = fit
    write(
        OUT / "ACTIVE.json",
        {
            "pid": os.getpid(),
            "utc": utc(),
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "status": "STARTING_EXISTING_RGB_DRIVER_WITH_INPUT_RESIDENCY",
        },
    )
    try:
        rgb_run.run(once=once)
    finally:
        if dataset is not None:
            dataset.shutdown()
        rgb_train.RGBPairDataset = original_dataset
        rgb_train.ProducerCheckpoint = original_checkpoint
        rgb_run.fit = original_fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admit-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    print(admission()) if args.admit_only else run(once=args.once)

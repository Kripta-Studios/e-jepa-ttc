"""Recover repeated capture invalidation by ending the CUDA process at durable boundaries."""

from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from . import geometry_train
from .common import ROOT, RUN, accounting, gpu_charge, require_resources
from .data import cache_root
from .fast_runtime import PersistentSensors
from .fusion_replication_train import DIRECTORY, REGISTRATION, SEEDS, qa, registered, segment
from .geometry_graph_probe import no_live_training_driver
from .rgb_run import require_authorization
from .train import committed_sources

OUT = RUN / "runtime/fusion_replication_isolated_v1"


def admission() -> dict:
    """Keep every registered scientific/runtime byte and the existing full-state QA."""
    require_authorization()
    registered()
    proof = qa()
    path = Path(__file__)
    sources = {path.relative_to(ROOT).as_posix(): sha(path)}
    committed_sources(sources)
    record = {
        "replication_registration_sha256": sha(REGISTRATION),
        "sources": sources,
        "exact_resume_qa": proof,
        "policy": "One unchanged segment per fresh CUDA process; at most900s reserved scope",
        "reason": "Repeated CUDA capture invalidation during a later segment in one process",
        "numerical_changes": [],
        "checkpoint": "Original complete model/Adam/scheduler/sampler/RNG/cursor; no reset",
        "cache": "Same lossless8GiB LRU/prefetch, rebuilt in each child process",
        "additional_optimizer_updates": 0,
        "retry_policy": "Stop on child failure; never skip an update or failed receipt",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def child(seed: int) -> None:
    """Advance one original segment and terminate its CUDA context after durable save."""
    admission()
    record = registered()
    if seed not in SEEDS:
        raise ValueError("Unregistered replica seed")
    require_resources(starting=True)
    sensors = PersistentSensors()
    folder = DIRECTORY / "fits" / f"C1-FUSION-REPLICA-S{seed}"
    try:
        if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
            raise ValueError("P sensor identity changed")
        masses = training_mass(
            sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
        )
        if digest(masses.tolist()) != record["mass_sha256"]:
            raise ValueError("P sampling masses changed")
        with gpu_charge(f"fusion_replication_isolated_seed{seed}", reserve=900) as deadline:
            provenance = folder / "RUN_PROVENANCE.json"
            if not provenance.exists():
                freeze(
                    provenance,
                    {
                        "experiment_id": "V18_FUSION_C1_REPLICATION",
                        "run_name": folder.name,
                        "seed": seed,
                        "git_commit": committed_sources(record["sources"]),
                        "config_hash": digest(record),
                        "dataset_manifest_hash": record["P_manifest_sha256"],
                        "split_version": "unchanged native P/H/V",
                        "host": socket.gethostname(),
                        "python_version": platform.python_version(),
                        "torch_version": str(torch.__version__),
                        "cuda_version": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "start_time": utc(),
                        "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                        "checkpoint_path": str(folder / "checkpoint_last.pt"),
                        "process_boundary_runtime_sha256": sha(OUT / "REGISTRATION.json"),
                    },
                )
            result = segment(folder, sensors, masses, seed, limit=12000, deadline=deadline)
            write(
                OUT / f"SEGMENT_{seed}_{result['completed_updates']:06d}.json",
                {
                    "utc": utc(),
                    "pid": os.getpid(),
                    "receipt": result,
                    "accounting": accounting(),
                    "runtime_registration_sha256": sha(OUT / "REGISTRATION.json"),
                },
            )
            print(result, flush=True)
    finally:
        sensors.shutdown()


def run() -> None:
    """Replace the failed driver with one coordinator and one GPU child at a time."""
    no_live_training_driver()
    admission()
    python = ROOT.parent / "e-jepa-ttc/.venv/Scripts/python.exe"
    with patch.object(geometry_train, "DIRECTORY", DIRECTORY), geometry_train.exclusive_driver():
        try:
            for seed in SEEDS:
                folder = DIRECTORY / "fits" / f"C1-FUSION-REPLICA-S{seed}"
                while True:
                    path = folder / "CHECKPOINT_RECEIPT.json"
                    if path.exists() and read(path)["status"] == "COMPLETE":
                        receipt = read(path)
                        if receipt["completed_updates"] != 12000 or receipt[
                            "checkpoint_sha256"
                        ] != sha(folder / "checkpoint_last.pt"):
                            raise ValueError("Replica endpoint changed")
                        break
                    before = read(path)["completed_updates"] if path.exists() else 0
                    command = [
                        str(python),
                        "-u",
                        "-B",
                        "-m",
                        "operational.joint_measurement.fusion_replication_isolated",
                        "--segment",
                        "--seed",
                        str(seed),
                    ]
                    with subprocess.Popen(
                        command, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW
                    ) as process:
                        write(
                            DIRECTORY / "RUNNER_STATE.json",
                            {
                                "status": "TRAINING",
                                "seed": seed,
                                "pid": os.getpid(),
                                "child_launcher_pid": process.pid,
                                "utc": utc(),
                                "runtime": "fresh CUDA process per durable segment",
                            },
                        )
                        code = process.wait()
                    if code != 0:
                        raise RuntimeError(f"Isolated replica child exited with code {code}")
                    if read(path)["completed_updates"] <= before:
                        raise RuntimeError("Isolated replica segment made no progress")
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "REPLICA_PRODUCERS_COMPLETE",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "full_objective_complete": False,
                    "next": "New H features/normalizers and four heads before V",
                },
            )
        except Exception as error:
            write(
                DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "ACTION_REQUIRED",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=13)
    args = parser.parse_args()
    child(args.seed) if args.segment else run()

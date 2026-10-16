"""Run each unchanged B1 segment in a fresh CUDA process after capture invalidation."""

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

from . import correspondence_model as numerical
from . import correspondence_train as training
from . import geometry_train
from .common import ROOT, RUN, accounting, gpu_charge, require_resources
from .correspondence_run import BufferedCorrespondence
from .correspondence_run import admission as buffered_admission
from .data import cache_root
from .fast_runtime import PersistentSensors
from .geometry_graph_probe import no_live_training_driver
from .rgb_run import require_authorization
from .train import committed_sources

OUT = RUN / "runtime/correspondence_isolated_v1"


def admission() -> dict:
    """Bind allocation and process isolation without changing scientific sources or QA."""
    require_authorization()
    buffered_admission()
    proof_path = training.QA / "RESULT.json"
    proof = read(proof_path)
    if (
        proof["status"] != "PASS"
        or proof["registration_sha256"] != sha(training.REGISTRATION)
        or proof["physical_technical_updates"] != 90
        or set(proof["checks"]) != {"CORRESPONDENCE/1", "TTC-CONTROL/1", "TTC-CONTROL/2"}
        or not all(
            all(row[kind].values())
            for row in proof["checks"].values()
            for kind in ("exact_resume", "eager_numerical")
        )
    ):
        raise ValueError("B1 isolation requires original complete CUDA QA")
    own = Path(__file__).relative_to(ROOT).as_posix()
    sources = {own: sha(Path(__file__))}
    committed_sources(sources)
    record = {
        "sources": sources,
        "scientific_registration_sha256": sha(training.REGISTRATION),
        "buffered_runtime_sha256": sha(
            RUN / "runtime/correspondence_buffered_capture_v1/REGISTRATION.json"
        ),
        "QA_sha256": sha(proof_path),
        "policy": "One original segment per fresh CUDA process; max900s reserved scope",
        "reason": "Capture invalidated entering stage2 after both durable stage1 endpoints",
        "numerical_changes": [],
        "checkpoint": "Original complete model/Adam/scheduler/sampler/RNG/cursor; no reset",
        "cache": "Original lossless8GiB LRU and pinned prefetch rebuilt per child",
        "additional_technical_updates": 0,
        "retry_policy": "Stop on child failure; preserve logs and every durable update",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def child(mode: str, stage: int) -> None:
    """Advance one admitted segment and release its CUDA context on process exit."""
    admission()
    record = training.registered()
    target = training.folder(mode, stage)
    require_resources(starting=True)
    sensors = PersistentSensors()
    try:
        if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
            raise ValueError("P sensor identity changed")
        masses = training_mass(
            sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
        )
        if digest(masses.tolist()) != record["mass_sha256"]:
            raise ValueError("P sampling masses changed")
        provenance = target / "RUN_PROVENANCE.json"
        if not provenance.exists():
            freeze(
                provenance,
                {
                    "experiment_id": record["experiment_id"],
                    "run_name": target.name,
                    "git_commit": committed_sources(training.source_binding()),
                    "seed": 7,
                    "config_hash": digest(record),
                    "dataset_manifest_hash": record["P_manifest_sha256"],
                    "split_version": "native P/H/V",
                    "host": socket.gethostname(),
                    "python_version": platform.python_version(),
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "start_time": utc(),
                    "checkpoint_path": str(target / "checkpoint_last.pt"),
                    "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                    "process_boundary_runtime_sha256": sha(OUT / "REGISTRATION.json"),
                },
            )
        with (
            patch.object(numerical, "CorrespondenceCore", BufferedCorrespondence),
            gpu_charge(f"B1_isolated_{mode}_stage{stage}", reserve=900) as deadline,
        ):
            result = training.segment(
                target,
                sensors,
                masses,
                mode,
                stage,
                limit=training.STEPS[stage],
                deadline=deadline,
                capture=True,
            )
        write(
            OUT / f"SEGMENT_{mode}_{stage}_{result['completed_updates']:06d}.json",
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
    """Keep one coordinator and one GPU child, stopping explicitly on any failed child."""
    no_live_training_driver()
    admission()
    python = ROOT.parent / "e-jepa-ttc/.venv/Scripts/python.exe"
    directory = training.DIRECTORY
    with patch.object(geometry_train, "DIRECTORY", directory), geometry_train.exclusive_driver():
        try:
            for stage in training.STEPS:
                for mode in training.MODES:
                    target = training.folder(mode, stage)
                    path = target / "CHECKPOINT_RECEIPT.json"
                    while True:
                        receipt = read(path) if path.exists() else None
                        if receipt and receipt["status"] == "COMPLETE":
                            if receipt["completed_updates"] != training.STEPS[stage] or receipt[
                                "checkpoint_sha256"
                            ] != sha(target / "checkpoint_last.pt"):
                                raise ValueError("B1 completed endpoint changed")
                            break
                        before = receipt["completed_updates"] if receipt else 0
                        command = [
                            str(python),
                            "-u",
                            "-B",
                            "-m",
                            "operational.joint_measurement.correspondence_isolated",
                            "--segment",
                            "--mode",
                            mode,
                            "--stage",
                            str(stage),
                        ]
                        with subprocess.Popen(
                            command, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW
                        ) as process:
                            write(
                                directory / "RUNNER_STATE.json",
                                {
                                    "status": "TRAINING",
                                    "mode": mode,
                                    "stage": stage,
                                    "pid": os.getpid(),
                                    "child_launcher_pid": process.pid,
                                    "utc": utc(),
                                    "runtime": "fresh CUDA process per durable segment",
                                },
                            )
                            code = process.wait()
                        if code != 0:
                            raise RuntimeError(f"Isolated B1 child exited with code {code}")
                        if read(path)["completed_updates"] <= before:
                            raise RuntimeError("Isolated B1 segment made no progress")
            freeze(
                directory / "ENDPOINTS.json",
                {
                    training.folder(mode, stage).name: sha(
                        training.folder(mode, stage) / "checkpoint_last.pt"
                    )
                    for stage in training.STEPS
                    for mode in training.MODES
                },
            )
            write(
                directory / "RUNNER_STATE.json",
                {
                    "status": "B1_ENDPOINTS_COMPLETE",
                    "pid": os.getpid(),
                    "utc": utc(),
                    "full_objective_complete": False,
                },
            )
        except Exception as error:
            write(
                directory / "RUNNER_STATE.json",
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
    parser.add_argument("--mode", choices=training.MODES, default=training.MODES[0])
    parser.add_argument("--stage", type=int, choices=training.STEPS, default=2)
    args = parser.parse_args()
    child(args.mode, args.stage) if args.segment else run()

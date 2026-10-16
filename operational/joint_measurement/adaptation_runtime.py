"""Admitted A1 transport/loss runtime with one persistent CUDA child per arm."""

from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from . import adaptation_depth as a1
from .adaptation_runtime_qa import OUT, QA, check_registration
from .common import ROOT, accounting, gpu_charge, require_resources
from .compute_guard import guard_idle
from .data import cache_root
from .parallel_training_sensor import ParallelTrainingSensors
from .persistent_adaptation import Session
from .qa import exact
from .rgb_run import require_authorization
from .train import committed_sources


def admission() -> dict:
    """Require original scientific registration and every new runtime proof."""
    record = check_registration()
    proofs = {
        "cpu": OUT / "CPU.json",
        "native": OUT / "NATIVE.json",
        **{mode: QA / mode / "RESULT.json" for mode in a1.MODES},
    }
    for path in proofs.values():
        result = read(path)
        if result["status"] != "PASS" or result["registration_sha256"] != sha(
            OUT / "REGISTRATION.json"
        ):
            raise ValueError(f"Runtime admission incomplete: {path}")
    result = {
        "status": "ADMITTED",
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "proofs": {name: sha(path) for name, path in proofs.items()},
        "sources": record["sources"],
        "checkpoint_policy": "Original complete state, no resets",
    }
    freeze(OUT / "ADMISSION.json", result)
    return result


def verify_transition(target: Path, origin: dict, population: int) -> None:
    """Prove the original 1086-row trajectory and Adam/sampler counters survive installation."""
    current = torch.load(target / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    count = current["completed_updates"] - origin["completed_updates"]
    if count <= 0:
        return
    generator = torch.Generator().set_state(origin["sampler_generator_state"])
    for row in current["cursor"]["curve"][origin["completed_updates"] :]:
        ids = torch.randint(population, (32,), generator=generator).tolist()
        if ids != row["ids"]:
            raise ValueError("Installed runtime altered authoritative sample order")
    old_state, new_state = (
        origin["optimizer_state_dict"]["state"],
        current["optimizer_state_dict"]["state"],
    )
    checks = {
        "original_curve_prefix": exact(
            origin["cursor"]["curve"], current["cursor"]["curve"][: origin["completed_updates"]]
        ),
        "original_timing_prefix": exact(
            origin["cursor"]["timings"], current["cursor"]["timings"][: origin["completed_updates"]]
        ),
        "identity": exact(origin["identity"], current["identity"]),
        "sampler": torch.equal(generator.get_state(), current["sampler_generator_state"]),
        "Adam_advanced_without_reset": set(old_state) == set(new_state)
        and all(
            float(new_state[k]["step"]) - float(v["step"]) == count for k, v in old_state.items()
        ),
        "no_recovery": current["recovery_upper"] == origin["recovery_upper"],
        "scheduler_advanced": current["scheduler_state_dict"]["last_epoch"]
        - origin["scheduler_state_dict"]["last_epoch"]
        == count,
    }
    if not all(checks.values()):
        raise ValueError(f"Runtime transition verification failed: {checks}")
    freeze(
        OUT / "ACTIVATION_VERIFIED.json",
        {
            "utc": utc(),
            "checks": checks,
            "origin_updates": origin["completed_updates"],
            "completed_updates": current["completed_updates"],
            "new_updates": count,
            "checkpoint_sha256": sha(target / "checkpoint_last.pt"),
        },
    )


def child(mode: str) -> None:
    """Retain graph, weights, optimizer and sensory LRU across independently charged scopes."""
    guard_idle()
    proof = admission()
    record = a1.registered()
    sensors = ParallelTrainingSensors(workers=8)
    target = a1.folder(mode)
    session = None
    try:
        if sha(cache_root("P") / "COMPLETE.json") != record["sensor_binding"]:
            raise ValueError("P population changed")
        masses = training_mass(
            sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
        )
        if digest(masses.tolist()) != record["mass_sha256"]:
            raise ValueError("P masses changed")
        freeze(target / "RUNTIME_ADMISSION.json", proof)
        while session is None or session.state.completed < a1.UPDATES:
            require_resources(starting=session is None)
            with gpu_charge(f"A1_persistent_{mode}", reserve=900) as deadline:
                if session is None:
                    session = Session(target, mode, masses)
                    if mode == "LAST-BLOCK" and not (OUT / "ACTIVATION_VERIFIED.json").exists():
                        origin = torch.load(
                            OUT / "ORIGIN.pt", map_location="cpu", weights_only=False
                        )
                        if session.state.completed != origin["completed_updates"]:
                            raise ValueError("Unverified transition cursor changed")
                        session.save = transition_saver(
                            session.save, target, origin, sensors.population
                        )
                    if not (target / "RUN_PROVENANCE.json").exists():
                        freeze(
                            target / "RUN_PROVENANCE.json",
                            {
                                "experiment_id": record["experiment_id"],
                                "run_name": target.name,
                                "git_commit": committed_sources(a1.source_binding()),
                                "config_hash": digest(record),
                                "seed": 7,
                                "dataset_manifest_hash": record["P_manifest_sha256"],
                                "split_version": "native P/H/V",
                                "host": socket.gethostname(),
                                "python_version": platform.python_version(),
                                "torch_version": str(torch.__version__),
                                "cuda_version": torch.version.cuda,
                                "gpu_name": torch.cuda.get_device_name(0),
                                "start_time": utc(),
                                "checkpoint_path": str(target / "checkpoint_last.pt"),
                                "metrics_path": str(target / "CURVE.csv"),
                                "parent_sha256": record["parent_sha256"],
                                "runtime_admission_sha256": sha(OUT / "ADMISSION.json"),
                                "end_time_status": "CHECKPOINT_RECEIPT.json and RUNNER_STATE.json",
                            },
                        )
                before = session.state.completed
                result = session.advance(sensors, limit=a1.UPDATES, deadline=deadline)
                a1.verify_change(target, mode)
            write(
                a1.DIRECTORY / "segments" / f"{mode}_{result['completed_updates']:06d}.json",
                {
                    "utc": utc(),
                    "receipt": result,
                    "accounting": accounting(),
                    "runtime_admission_sha256": sha(OUT / "ADMISSION.json"),
                    "same_process_scope": session.scope_count,
                    "captures": session.capture_count,
                    "cache": sensors.counters(),
                    "pid": os.getpid(),
                },
            )
            if session.state.completed <= before:
                raise RuntimeError(
                    "Persistent scope made no progress; inspect resources before resume"
                )
    finally:
        sensors.shutdown()


def transition_saver(
    save: Callable[[str], None], target: Path, origin: dict, population: int
) -> Callable[[str], None]:
    """Bind the existing saver and transition witness once, outside the scope loop."""

    def checked_save(status: str) -> None:
        save(status)
        if not (OUT / "ACTIVATION_VERIFIED.json").exists():
            verify_transition(target, origin, population)

    return checked_save


def run() -> None:
    """Keep the existing exclusive driver lock; never retry a failed child silently."""
    guard_idle()
    require_authorization()
    admission()
    python = ROOT.parent / "e-jepa-ttc/.venv/Scripts/python.exe"
    with (
        patch.object(a1.geometry_train, "DIRECTORY", a1.DIRECTORY),
        a1.geometry_train.exclusive_driver(),
    ):
        try:
            for mode in a1.MODES:
                target = a1.folder(mode)
                path = target / "CHECKPOINT_RECEIPT.json"
                if not path.exists() or read(path)["completed_updates"] != a1.UPDATES:
                    with subprocess.Popen(
                        [
                            str(python),
                            "-u",
                            "-B",
                            "-m",
                            "operational.joint_measurement.adaptation_runtime",
                            "child",
                            "--arm",
                            mode,
                        ],
                        cwd=ROOT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    ) as process:
                        write(
                            a1.DIRECTORY / "RUNNER_STATE.json",
                            {
                                "status": "TRAINING",
                                "mode": mode,
                                "pid": os.getpid(),
                                "child_launcher_pid": process.pid,
                                "utc": utc(),
                                "runtime": "parallel_reader_persistent_graph_combined_phase_guards",
                            },
                        )
                        code = process.wait()
                    if code:
                        raise RuntimeError(f"Persistent A1 child failed: {code}")
                receipt = read(path)
                if receipt["status"] != "COMPLETE" or receipt["checkpoint_sha256"] != sha(
                    target / "checkpoint_last.pt"
                ):
                    raise ValueError("A1 endpoint incomplete or changed")
                a1.verify_change(target, mode)
            freeze(
                a1.DIRECTORY / "ENDPOINTS.json",
                {
                    a1.folder(mode).name: sha(a1.folder(mode) / "checkpoint_last.pt")
                    for mode in a1.MODES
                },
            )
            write(
                a1.DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "A1_ENDPOINTS_COMPLETE",
                    "utc": utc(),
                    "pid": os.getpid(),
                    "full_objective_complete": False,
                },
            )
        except Exception as error:
            write(
                a1.DIRECTORY / "RUNNER_STATE.json",
                {
                    "status": "ACTION_REQUIRED",
                    "utc": utc(),
                    "pid": os.getpid(),
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "child", "admit"))
    parser.add_argument("--arm", choices=a1.MODES, default=a1.MODES[0])
    args = parser.parse_args()
    child(args.arm) if args.action == "child" else {"run": run, "admit": admission}[args.action]()

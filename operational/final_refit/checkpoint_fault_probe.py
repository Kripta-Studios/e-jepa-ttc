"""Hermetic CPU fault injection for the durable producer checkpoint transaction.

This probe starts real child processes and terminates them with ``os._exit`` at
controlled Python-level transaction boundaries.  It does not emulate a machine
power loss, filesystem-controller cache loss, or torn sector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn

import operational.rgb_port.train_producers as checkpoint_module
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.recipe import canonical_sha256, file_sha256
from operational.rgb_port.train_producers import ProducerCheckpoint

UPDATES = 12
SAVE_INTERVAL = 4
FAULT_EXIT_CODE = 91
FAULTS = (
    "commit_journal_before",
    "commit_journal_after",
    "save_version_after_write",
    "save_pointer_after",
    "save_alias_after",
    "save_journal_after",
    "save_receipt_after",
)


def _identity() -> dict[str, Any]:
    payload = {
        "schema": "final_refit_checkpoint_fault_probe_identity_v1",
        "fit_id": "CPU_FAULT_PROBE",
        "technical": True,
        "seed": 7,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _components() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.Generator,
]:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Dropout(0.2), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    generator = torch.Generator(device="cpu").manual_seed(7)
    return model, optimizer, scheduler, generator


def _exit() -> None:
    os._exit(FAULT_EXIT_CODE)


def _install_fault(state: ProducerCheckpoint, fault: str, target_update: int) -> None:
    """Terminate after/before one named Python transaction boundary."""
    original_journal = state._write_journal

    def journal(self: ProducerCheckpoint) -> None:
        commit_boundary = (
            self.completed == target_update
            and not self.pending
            and self.durable < self.completed
        )
        if fault == "commit_journal_before" and commit_boundary:
            _exit()
        original_journal()
        if fault == "commit_journal_after" and commit_boundary:
            _exit()

    state._write_journal = MethodType(journal, state)  # type: ignore[method-assign]
    original_save = checkpoint_module.torch.save
    original_atomic = checkpoint_module._atomic_json
    original_replace = checkpoint_module._durable_replace

    def torch_save(value: Any, path: str | os.PathLike[str]) -> None:  # noqa: ANN401
        original_save(value, path)
        if (
            fault == "save_version_after_write"
            and isinstance(value, dict)
            and value.get("completed_updates") == target_update
            and str(path).endswith(".pending")
        ):
            _exit()

    def atomic(path: Path, value: Mapping[str, Any]) -> None:
        original_atomic(path, value)
        if int(value.get("completed_updates", -1)) != target_update:
            return
        if fault == "save_pointer_after" and path.name == "CHECKPOINT_POINTER.json":
            _exit()
        if (
            fault == "save_journal_after"
            and path.name == "UPDATE_JOURNAL.json"
            and int(value.get("durable_updates", -1)) == target_update
        ):
            _exit()
        if fault == "save_receipt_after" and path.name == "CHECKPOINT_RECEIPT.json":
            _exit()

    def durable_replace(source: Path, target: Path) -> None:
        original_replace(source, target)
        if (
            fault == "save_alias_after"
            and target.name == "checkpoint_last.pt"
            and state.completed == target_update
        ):
            _exit()

    checkpoint_module.torch.save = torch_save  # type: ignore[method-assign]
    checkpoint_module._atomic_json = atomic
    checkpoint_module._durable_replace = durable_replace


def _next_index(cursor: dict[str, Any], generator: torch.Generator) -> int:
    order = cursor["order"]
    position = int(cursor["position"])
    if position == len(order):
        order = torch.randperm(16, generator=generator)
        cursor["order"], cursor["position"] = order, 0
        position = 0
    cursor["position"] = position + 1
    return int(order[position])


def _record_save_time(run: Path, seconds: float) -> None:
    path = run / "SAVE_TIMINGS.json"
    prior = read_json_shared(path) if path.is_file() else {"seconds": []}
    atomic_write_json(path, {"seconds": [*prior["seconds"], seconds]})


def train_worker(run: Path, fault: str | None) -> dict[str, Any]:
    """Run or resume the tiny deterministic optimizer through the real checkpoint class."""
    model, optimizer, scheduler, generator = _components()
    state = ProducerCheckpoint(run, _identity(), UPDATES)
    cursor: dict[str, Any] = {
        "order": torch.randperm(16, generator=generator),
        "position": 0,
        "curve": [],
    }
    if state.pointer.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        started = time.perf_counter()
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
        _record_save_time(run, time.perf_counter() - started)
    if fault is not None:
        target = 6 if fault.startswith("commit_") else 8
        _install_fault(state, fault, target)

    data = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 31.0
    target = torch.sin(data.sum(dim=1, keepdim=True))
    while state.completed < UPDATES:
        index = _next_index(cursor, generator)
        python_scale = 0.9 + 0.2 * random.random()
        numpy_scale = 0.9 + 0.2 * float(np.random.random())
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data[index : index + 1])
        loss = ((prediction - target[index : index + 1]) ** 2).mean()
        loss = loss * python_scale * numpy_scale
        loss.backward()
        state.begin()
        optimizer.step()
        scheduler.step()
        state.commit()
        cursor["curve"].append(float(loss.detach()))
        if state.completed % SAVE_INTERVAL == 0 or state.completed == UPDATES:
            started = time.perf_counter()
            state.save(
                model,
                optimizer,
                scheduler,
                generator,
                cursor,
                status="COMPLETE" if state.completed == UPDATES else "IN_PROGRESS",
            )
            _record_save_time(run, time.perf_counter() - started)
    return read_json_shared(state.receipt)


def _run_child(run: Path, fault: str | None) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "operational.final_refit.checkpoint_fault_probe",
        "worker",
        "--run",
        str(run),
    ]
    if fault is not None:
        command.extend(("--fault", fault))
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)


def _load_final(run: Path) -> dict[str, Any]:
    receipt = read_json_shared(run / "CHECKPOINT_RECEIPT.json")
    checkpoint = Path(str(receipt["checkpoint_path"])).resolve(strict=True)
    if receipt.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("Fault-probe final receipt does not bind its checkpoint")
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def _equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def run_suite(output: Path) -> dict[str, Any]:
    """Execute baseline and every fault in separate processes, then compare final states."""
    output.mkdir(parents=True, exist_ok=False)
    baseline_run = output / "baseline"
    baseline_process = _run_child(baseline_run, None)
    if baseline_process.returncode != 0:
        raise RuntimeError(f"Baseline child failed: {baseline_process.stderr}")
    baseline = _load_final(baseline_run)
    comparisons = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "numpy_random_state",
        "python_random_state",
        "cursor",
        "completed_updates",
    )
    cases: list[dict[str, Any]] = []
    for fault in FAULTS:
        run = output / fault
        interrupted = _run_child(run, fault)
        if interrupted.returncode != FAULT_EXIT_CODE:
            raise RuntimeError(
                f"Fault {fault} exited {interrupted.returncode}: {interrupted.stderr}"
            )
        resumed = _run_child(run, None)
        if resumed.returncode != 0:
            raise RuntimeError(f"Resume {fault} failed: {resumed.stderr}")
        payload = _load_final(run)
        exact = {name: _equal(baseline[name], payload[name]) for name in comparisons}
        receipt = read_json_shared(run / "CHECKPOINT_RECEIPT.json")
        journal = read_json_shared(run / "UPDATE_JOURNAL.json")
        cases.append(
            {
                "fault": fault,
                "interrupted_returncode": interrupted.returncode,
                "resume_returncode": resumed.returncode,
                "exact": exact,
                "all_exact": all(exact.values()),
                "completed_updates": receipt["completed_updates"],
                "scientific_endpoint": receipt["scientific_endpoint"],
                "recovery_upper": journal["recovery_upper"],
                "bounded_by_save_interval": int(journal["recovery_upper"]) <= SAVE_INTERVAL,
            }
        )
    baseline_saves = read_json_shared(baseline_run / "SAVE_TIMINGS.json")["seconds"]
    payload: dict[str, Any] = {
        "schema": "final_refit_checkpoint_fault_probe_v1",
        "status": "PASS" if all(case["all_exact"] for case in cases) else "FAIL",
        "scope": "real_process_exit_at_python_transaction_boundaries_cpu_only",
        "not_tested": [
            "machine_power_loss",
            "filesystem_controller_cache_loss",
            "torn_sector_or_device_failure",
            "CUDA_state_restore",
        ],
        "updates": UPDATES,
        "save_interval_updates": SAVE_INTERVAL,
        "baseline_save_seconds": baseline_saves,
        "baseline_save_seconds_median": float(np.median(baseline_saves)),
        "cases": cases,
    }
    atomic_write_json(output / "FAULT_PROBE_REPORT.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--fault", choices=FAULTS)
    suite = commands.add_parser("suite")
    suite.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "worker":
        result = train_worker(args.run.resolve(), args.fault)
    else:
        result = run_suite(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FAULTS", "run_suite", "train_worker"]

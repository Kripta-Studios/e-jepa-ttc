"""Preserve a full native checkpoint before enabling the tested lossless cache amendment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read


def main() -> int:
    """Preserve this campaign's complete state at the next durable optimizer boundary."""
    import psutil
    import torch
    from e_jepa_ttc.simplex_t.training import state_digest

    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    c.freeze()
    qa = read(c.out / "TEST_RESULTS/parallel_inputs/QA.json")
    if qa["status"] != "PASSED":
        raise ValueError("parallel CPU input engineering QA required")
    for pin in qa["files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("compressed cache QA source changed before the restart")
    if read(c.out / "garl/PARALLEL_INPUT_QA.json")["status"] != "PASSED":
        raise ValueError("real TRAIN cache bit parity required")
    progress_path = c.out / "garl/UPDATE_PROGRESS.json"
    initial = read(progress_path)
    work_path = c.out / "garl/PHYSICAL_WORK.json"
    initial_saved = read(work_path)["fits"][initial["fit"]]["saved"]
    while True:
        progress = read(progress_path)
        if progress["fit"] != initial["fit"]:
            raise RuntimeError("fit changed before the controlled checkpoint boundary")
        saved = read(work_path)["fits"][initial["fit"]]["saved"]
        if saved > initial_saved:
            break
        owner = read(c.out / "WRITER.lock")
        if not psutil.pid_exists(owner["pid"]):
            raise RuntimeError(
                "previously sealed native cache worker stopped before a durable checkpoint"
            )
        time.sleep(2)
    owner_record = read(c.out / "WRITER.lock")
    owner = psutil.Process(owner_record["pid"])
    line = " ".join(owner.cmdline())
    if (
        owner.create_time() != owner_record["create_time"]
        or Path(owner.cwd()).resolve() != ROOT
        or "-m operational.efficient_context.garl_train_compressed --resume" not in line
    ):
        raise ValueError(
            "refusing to interrupt a process outside this previously sealed native cache worker"
        )
    owner.suspend()
    try:
        checkpoint = c.out / f"garl/fits/{progress['fit']}/checkpoint_last.pt"
        checkpoint_sha = digest(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seal = state.pop("state_sha256")
        if state_digest(state) != seal or state["completed_updates"] < 100:
            raise ValueError("complete native checkpoint is not verified")
        if state["protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json"):
            raise ValueError("native checkpoint differs from its original scientific freeze")
        required = {
            "model",
            "optimizer",
            "scheduler",
            "torch_rng",
            "cuda_rng",
            "sampler_rng",
            "python_rng",
            "numpy_rng",
            "order",
            "cursor",
            "epoch",
            "losses",
        }
        if not required.issubset(state):
            raise ValueError("native checkpoint misses complete restoration state")
        snapshot = c.out / "verification/parallel_input_engineering/checkpoint_before_amendment.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, snapshot)
        if digest(snapshot) != checkpoint_sha:
            raise ValueError("preserved native checkpoint bytes differ")
        next_ids = state["order"][state["cursor"] : state["cursor"] + 128]
        observed = read(c.out / "garl/UPDATE_PROGRESS.json")
        proof = {
            "observed_utc": datetime.now(UTC).isoformat(),
            "fit": state["key"],
            "durable_updates": state["completed_updates"],
            "checkpoint_sha256": checkpoint_sha,
            "state_sha256": seal,
            "preserved_checkpoint": str(snapshot),
            "complete_RNG_optimizer_scheduler_sampler_payloads_retained": True,
            "next_batch_indices": next_ids.tolist(),
            "confirmed_progress_before_restart": observed["confirmed_updates"],
            "confirmed_lost_updates_lower": max(
                0, observed["confirmed_updates"] - state["completed_updates"]
            ),
            "pending_updates_upper": 100,
            "optimizer_updates_for_proof": 0,
            "scientific_recipe_changed": False,
            "original_owner_pid": owner.pid,
            "original_owner_create_time": owner.create_time(),
            "cache_QA_sha256": digest(c.out / "garl/PARALLEL_INPUT_QA.json"),
            "controller_implementation_sha256": digest(Path(__file__)),
        }
        atomic_json(c.out / "garl/PARALLEL_INPUT_RESUME_PROOF.json", proof)
        del state
    except BaseException:
        owner.resume()
        raise
    owner.terminate()
    owner.wait(timeout=10)
    print("NATIVE_COMPLETE_STATE_PRESERVED", proof["durable_updates"], flush=True)
    # Let the original parent retain its nonzero exit and logs. A stale writer
    # prevents its packaging step from deleting the live lossless numeric cache.
    while True:
        parents = []
        for process in psutil.process_iter(["pid", "cmdline", "name"]):
            line = " ".join(process.info["cmdline"] or [])
            if (
                process.pid != os.getpid()
                and (process.info["name"] or "").lower().startswith("python")
                and "-m operational.efficient_context.queue all " in line
            ):
                try:
                    if Path(process.cwd()).resolve() == ROOT:
                        parents.append(process.pid)
                except psutil.Error:
                    pass
        if not parents:
            break
        time.sleep(2)
    log = c.out / "garl/PARALLEL_INPUT_RESUMED_QUEUE.txt"
    command = [
        sys.executable,
        "-m",
        "operational.efficient_context.queue",
        "all",
        "--resume",
        "--protocol",
        str(c.config_path),
    ]
    print("NATIVE_PARALLEL_INPUT_QUEUE_RESUME", flush=True)
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    atomic_json(
        c.out / "garl/PARALLEL_INPUT_QUEUE_RECEIPT.json",
        {
            "command": command,
            "returncode": result.returncode,
            "log": str(log),
            "end_utc": datetime.now(UTC).isoformat(),
        },
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

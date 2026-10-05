"""Announce the authorized next worker so the existing guard yields a complete optimizer state."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, Lease, atomic_json, digest, read


def run() -> int:
    """Use the old worker's existing one-trainer guard; never overlap heavy trainers."""
    import psutil

    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    c.freeze()
    qa = read(c.out / "TEST_RESULTS/parallel_inputs/QA.json")
    if qa["status"] != "PASSED":
        raise ValueError("CPU input handoff requires completed QA")
    for pin in qa["files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("CPU input handoff source changed since QA")
    intent = read(c.out / "garl/PARALLEL_INPUT_HANDOFF_INTENT.json")
    old_pid = intent["original_owner_pid"]
    fit = intent["fit"]
    if psutil.pid_exists(old_pid):
        owner = psutil.Process(old_pid)
        if (
            owner.create_time() != intent["original_owner_create_time"]
            or Path(owner.cwd()).resolve() != ROOT
            or "-m operational.efficient_context.garl_train_compressed --resume"
            not in " ".join(owner.cmdline())
        ):
            raise ValueError("handoff intent does not identify this campaign's old native worker")
    # This process is the announced next training owner. The previously frozen
    # GuardedCampaign detects its real module name and saves PAUSED_RESOURCE at
    # an optimizer boundary before releasing the writer. No new model exists yet.
    while psutil.pid_exists(old_pid) and (c.out / "WRITER.lock").exists():
        time.sleep(0.1)
    with Lease(c.out):
        while psutil.pid_exists(old_pid):
            time.sleep(0.1)
        import torch
        from e_jepa_ttc.simplex_t.training import state_digest

        checkpoint = c.out / f"garl/fits/{fit}/checkpoint_last.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seal = state.pop("state_sha256")
        if state_digest(state) != seal:
            raise ValueError("handoff checkpoint state digest differs")
        if state["protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json"):
            raise ValueError("handoff checkpoint differs from the original scientific freeze")
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
            raise ValueError("handoff checkpoint is not a complete restoration state")
        ledger = read(c.out / "garl/PHYSICAL_WORK.json")
        if ledger["fits"][fit]["saved"] != state["completed_updates"]:
            raise ValueError("handoff checkpoint and durable ledger differ")
        snapshot = c.out / "verification/parallel_input_engineering/checkpoint_before_amendment.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise ValueError("preserve the previous handoff snapshot; do not overwrite it")
        sha = digest(checkpoint)
        shutil.copy2(checkpoint, snapshot)
        if digest(snapshot) != sha:
            raise ValueError("preserved handoff checkpoint bytes differ")
        observed = read(c.out / "garl/UPDATE_PROGRESS.json")
        proof = {
            "observed_utc": datetime.now(UTC).isoformat(),
            "fit": fit,
            "durable_updates": state["completed_updates"],
            "checkpoint_sha256": sha,
            "state_sha256": seal,
            "preserved_checkpoint": str(snapshot),
            "complete_RNG_optimizer_scheduler_sampler_payloads_retained": True,
            "next_batch_indices": state["order"][state["cursor"] : state["cursor"] + 128].tolist(),
            "confirmed_progress_before_handoff": observed["confirmed_updates"],
            "confirmed_lost_updates_lower": max(
                0, observed["confirmed_updates"] - state["completed_updates"]
            ),
            "original_worker_checkpoint_status": state["status"],
            "cooperation": (
                "original one-trainer guard paused at optimizer boundary for next authorized owner"
            ),
            "original_owner_pid": old_pid,
            "next_owner_announcing_handoff_pid": os.getpid(),
            "new_model_constructed_before_old_owner_exit": False,
            "optimizer_updates_for_proof": 0,
            "scientific_recipe_changed": False,
            "worker_QA_sha256": digest(c.out / "garl/PARALLEL_INPUT_QA.json"),
            "controller_implementation_sha256": digest(Path(__file__)),
        }
        atomic_json(c.out / "garl/PARALLEL_INPUT_RESUME_PROOF.json", proof)
        intent.update(status="STATE_PRESERVED", checkpoint_sha256=sha)
        atomic_json(c.out / "garl/PARALLEL_INPUT_HANDOFF_INTENT.json", intent)
        del state
        # Keep the writer lease while the previous parent records its failed
        # resource status. Its partial-packaging step cannot evict the live cache.
        while True:
            parents = []
            for process in psutil.process_iter(["pid", "name", "cmdline"]):
                line = " ".join(process.info["cmdline"] or [])
                if (process.info["name"] or "").lower().startswith(
                    "python"
                ) and "-m operational.efficient_context.queue all " in line:
                    try:
                        if Path(process.cwd()).resolve() == ROOT:
                            parents.append(process.pid)
                    except psutil.Error:
                        pass
            if not parents:
                break
            time.sleep(0.2)
    print("NATIVE_COOPERATIVE_STATE_PRESERVED", proof["durable_updates"], flush=True)
    command = [
        sys.executable,
        "-m",
        "operational.efficient_context.queue",
        "all",
        "--resume",
        "--protocol",
        str(c.config_path),
    ]
    log = c.out / "garl/PARALLEL_INPUT_RESUMED_QUEUE.txt"
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    atomic_json(
        c.out / "garl/PARALLEL_INPUT_QUEUE_RECEIPT.json",
        {
            "command": command,
            "returncode": result.returncode,
            "log": str(log),
            "end_utc": datetime.now(UTC).isoformat(),
            "cooperative_handoff": True,
        },
    )
    return result.returncode

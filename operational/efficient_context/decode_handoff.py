"""Cooperative complete-state handoff for lossless input decoding and expanded disk cache."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from .common import ROOT, Campaign, Lease, atomic_json, digest, read


def handoff(c: Campaign) -> None:
    """Identify our old writer, let its existing guard save, and verify every resume payload."""
    import psutil

    owner = read(c.out / "WRITER.lock")
    pid = owner["pid"]
    process = psutil.Process(pid)
    if (
        process.create_time() != owner["create_time"]
        or Path(process.cwd()).resolve() != ROOT
        or "-m operational.efficient_context.garl_train_scandir" not in " ".join(process.cmdline())
    ):
        raise ValueError("decode handoff does not identify our original quota-guarded producer")
    intent_path = c.out / "garl/DECODE_HANDOFF_INTENT.json"
    if intent_path.exists():
        raise ValueError("preserve the previous decode handoff intent; do not repeat it")
    intent = {"status": "REQUESTED", "original_owner": owner, "next_owner_pid": os.getpid()}
    atomic_json(intent_path, intent)
    while psutil.pid_exists(pid) and (c.out / "WRITER.lock").exists():
        time.sleep(0.1)
    with Lease(c.out):
        while psutil.pid_exists(pid):
            time.sleep(0.1)
        import torch

        from e_jepa_ttc.simplex_t.training import state_digest

        ledger = read(c.out / "garl/PHYSICAL_WORK.json")
        fit = read(c.out / "garl/PROGRESS.json")["fit"]
        checkpoint = c.out / f"garl/fits/{fit}/checkpoint_last.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seal = state.pop("state_sha256")
        if (
            state_digest(state) != seal
            or state["protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
            or ledger["fits"][fit]["saved"] != state["completed_updates"]
        ):
            raise ValueError("decode handoff complete checkpoint/ledger/binding differs")
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
            raise ValueError("decode handoff checkpoint lacks complete resume payloads")
        snapshot = c.out / "verification/parallel_decode/checkpoint_before_amendment.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise ValueError("preserve the previous decode checkpoint snapshot")
        sha = digest(checkpoint)
        shutil.copy2(checkpoint, snapshot)
        if digest(snapshot) != sha:
            raise ValueError("preserved decode checkpoint bytes differ")
        observed = read(c.out / "garl/UPDATE_PROGRESS.json")
        atomic_json(
            c.out / "garl/DECODE_RESUME_PROOF.json",
            {
                "fit": fit,
                "durable_updates": state["completed_updates"],
                "checkpoint_sha256": sha,
                "state_sha256": seal,
                "preserved_checkpoint": str(snapshot),
                "complete_RNG_optimizer_scheduler_sampler_payloads_retained": True,
                "next_batch_indices": state["order"][
                    state["cursor"] : state["cursor"] + 128
                ].tolist(),
                "confirmed_lost_updates_lower": max(
                    0, observed["confirmed_updates"] - state["completed_updates"]
                ),
                "new_model_constructed_before_old_owner_exit": False,
                "optimizer_updates_for_proof": 0,
                "scientific_recipe_changed": False,
            },
        )
        intent.update(status="STATE_PRESERVED", checkpoint_sha256=sha)
        atomic_json(intent_path, intent)

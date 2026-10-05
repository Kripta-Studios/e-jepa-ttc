"""Cooperate with the frozen trainer guard to preserve a user-requested pause."""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read


def identify_owner(c: Campaign) -> tuple[dict, int]:
    """Verify the exact live writer before requesting its existing checkpoint guard."""
    import psutil

    authorization = read(c.out / "USER_STRATEGY_PAUSE_AUTHORIZATION.json")
    if (
        authorization["parent_protocol_sha256"] != digest(c.out / "PROTOCOL.json")
        or authorization["selected_answer"] != "Pausar Garl y medir el piloto (recomendado)"
    ):
        raise ValueError("user pause authorization differs from this campaign")
    owner = read(c.out / "WRITER.lock")
    process = psutil.Process(owner["pid"])
    if (
        process.create_time() != owner["create_time"]
        or Path(process.cwd()).resolve() != ROOT
        or "-m operational.efficient_context.garl_train_decode" not in " ".join(process.cmdline())
        or process.pid == os.getpid()
    ):
        raise ValueError("user pause does not identify our active native producer")
    return owner, process.pid


def run(c: Campaign, timeout_seconds: float) -> None:
    """Let the existing other-trainer guard save, then verify the durable full state."""
    import psutil

    owner, pid = identify_owner(c)
    intent = c.out / "garl/USER_PAUSE_INTENT.json"
    if intent.exists():
        raise ValueError("preserve the previous user pause intent; inspect before retry")
    observed = read(c.out / "garl/UPDATE_PROGRESS.json")
    atomic_json(
        intent,
        {
            "status": "REQUESTED",
            "requested_utc": datetime.now(UTC).isoformat(),
            "owner": owner,
            "coordinator_pid": os.getpid(),
            "observed_progress": observed,
            "mechanism": "existing GuardedCampaign other-process checkpoint boundary",
            "coordinator_constructs_model_or_optimizer": False,
        },
    )
    start = time.monotonic()
    while psutil.pid_exists(pid):
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("trainer has not finished its safe checkpoint; do not terminate")
        time.sleep(0.2)
    if (c.out / "WRITER.lock").exists():
        raise ValueError("writer lease persists after native process exit; inspect owner")

    import torch

    from e_jepa_ttc.simplex_t.training import state_digest

    fit = observed["fit"]
    checkpoint = c.out / f"garl/fits/{fit}/checkpoint_last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    seal = state.pop("state_sha256")
    ledger = read(c.out / "garl/PHYSICAL_WORK.json")
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
        "completed_updates",
        "losses",
    }
    if (
        not required.issubset(state)
        or state_digest(state) != seal
        or state["protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
        or state["key"] != fit
        or ledger["fits"][fit]["saved"] != state["completed_updates"]
        or ledger["fits"][fit]["pending"] is not None
        or state["completed_updates"] < observed["confirmed_updates"]
        or len(state["losses"]) != state["completed_updates"]
    ):
        raise ValueError("paused checkpoint state, binding or scientific ledger differs")
    receipt = {
        "status": "PAUSED_USER_STRATEGY",
        "completed_utc": datetime.now(UTC).isoformat(),
        "parent_protocol_sha256": digest(c.out / "PROTOCOL.json"),
        "native_protocol_sha256": digest(c.out / "garl/PROTOCOL.json"),
        "fit": fit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "state_sha256": seal,
        "durable_updates": state["completed_updates"],
        "saved_native_updates_all_fits": ledger["saved_updates"],
        "checkpoint_contains_optimizer_scheduler_RNG_sampler_and_full_model": True,
        "confirmed_lost_updates": 0,
        "checkpoint_rewritten_by_coordinator": False,
        "optimizer_updates_by_coordinator": 0,
        "scientific_negative": False,
        "original_resume_command": (
            "../e-jepa-ttc/.venv/Scripts/python.exe -m "
            "operational.efficient_context.garl_train_decode --resume"
        ),
        "resume_requires_user_decision_to_reactivate_original_queue": True,
    }
    atomic_json(c.out / "garl/USER_PAUSE_RECEIPT.json", receipt)
    atomic_json(intent, {**read(intent), "status": "STATE_PRESERVED"})


def main() -> int:
    """Checkpoint the authorized active producer without modifying its frozen code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    c.freeze()
    run(c, args.timeout_seconds)
    print("PAUSED_USER_STRATEGY: complete checkpoint verified; zero lost updates", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

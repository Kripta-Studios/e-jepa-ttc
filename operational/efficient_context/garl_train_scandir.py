"""Resume the original native recipes with separately sealed exact quota counting."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, Lease, atomic_json, digest, read


def handoff(c: Campaign) -> None:
    """Wait for the existing guard's complete-state save before constructing another model."""
    import psutil

    owner = read(c.out / "WRITER.lock")
    pid = owner["pid"]
    process = psutil.Process(pid)
    if (
        process.create_time() != owner["create_time"]
        or Path(process.cwd()).resolve() != ROOT
        or "-m operational.efficient_context.garl_train_parallel_authorized"
        not in " ".join(process.cmdline())
    ):
        raise ValueError("quota handoff does not identify our original parallel producer")
    intent_path = c.out / "garl/QUOTA_SCAN_HANDOFF_INTENT.json"
    intent = {"status": "REQUESTED", "original_owner": owner, "next_owner_pid": os.getpid()}
    if intent_path.exists():
        raise ValueError("preserve the previous quota handoff intent; do not repeat it")
    atomic_json(intent_path, intent)
    # The frozen original GuardedCampaign sees our legitimate module and saves at an
    # optimizer boundary. No new model exists while the previous trainer remains alive.
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
        if state_digest(state) != seal or state["protocol_sha256"] != digest(
            c.out / "garl/PROTOCOL.json"
        ):
            raise ValueError("quota handoff checkpoint digest/binding differs")
        if ledger["fits"][fit]["saved"] != state["completed_updates"]:
            raise ValueError("quota handoff ledger and checkpoint differ")
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
            raise ValueError("quota handoff checkpoint is not complete")
        snapshot = c.out / "verification/quota_scanner_engineering/checkpoint_before_amendment.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            raise ValueError("preserve the previous quota checkpoint snapshot")
        sha = digest(checkpoint)
        shutil.copy2(checkpoint, snapshot)
        if digest(snapshot) != sha:
            raise ValueError("quota handoff copy differs")
        observed = read(c.out / "garl/UPDATE_PROGRESS.json")
        atomic_json(
            c.out / "garl/QUOTA_SCANNER_RESUME_PROOF.json",
            {
                "observed_utc": datetime.now(UTC).isoformat(),
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


def main() -> int:
    """Restore identical producer states after source-verified engineering admission."""
    from . import garl_train_parallel_authorized

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--handoff-quota", action="store_true")
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args, _ = parser.parse_known_args()
    c = Campaign(args.protocol)
    c.freeze()
    if not (c.out / "garl/QUOTA_SCANNER_FREEZE.json").exists():
        raise ValueError("quota scanner must pass QA and freeze before activation")
    if args.handoff_quota:
        handoff(c)
        sys.argv.remove("--handoff-quota")
    return garl_train_parallel_authorized.main()


if __name__ == "__main__":
    raise SystemExit(main())

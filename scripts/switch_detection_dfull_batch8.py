"""Switch the authorized Dfull process at an atomic checkpoint boundary."""

# ruff: noqa: B009 -- POSIX-only signals are intentionally resolved on the Linux host.

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    """Perform a Linux-only checkpoint handover with bounded preparation rollback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    base = args.root
    project = base / "workspace/phase-ttc-s0-evaluation"
    campaign = base / "dfull_campaign_v1"
    source_run = campaign / "r2_dfull_seed7_v2"
    checkpoint = source_run / "checkpoints/last.pt"
    os.chdir(project)
    sys.path.insert(0, str(project / "src"))
    import torch
    import yaml

    from e_jepa_ttc.detection.batch_transition import build_plan
    from e_jepa_ttc.detection.campaign_training import (
        ResumablePermutationSampler,
        load_campaign_config,
        sha256_file,
    )

    def write(name: str, value: object) -> None:
        (campaign / name).write_text(json.dumps(value, indent=2) + "\n")

    command = Path(f"/proc/{args.trainer_pid}/cmdline").read_text()
    if str(source_run) not in command or "train_detection_campaign.py" not in command:
        raise ValueError("Trainer process identity differs")
    if (
        "run_detection_dfull_campaign.py"
        not in Path(f"/proc/{args.controller_pid}/cmdline").read_text()
    ):
        raise ValueError("Controller process identity differs")
    if (campaign / "BATCH_TRANSITION.json").exists():
        raise FileExistsError("Batch transition already exists")
    old = checkpoint.stat()
    initial_identity = (old.st_ino, old.st_mtime_ns)
    write(
        "BATCH8_SWITCH_STATUS.json",
        {"status": "WAITING_FOR_NEXT_ATOMIC_CHECKPOINT", "trainer_pid": args.trainer_pid},
    )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        new = checkpoint.stat()
        if (new.st_ino, new.st_mtime_ns) != initial_identity:
            break
        time.sleep(0.25)
    else:
        raise TimeoutError("No new checkpoint; original run remains active")
    os.kill(args.trainer_pid, getattr(signal, "SIGSTOP"))
    watch = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import os,signal,time;time.sleep(120);os.kill({args.trainer_pid},signal.SIGCONT)",
        ],
        start_new_session=True,
    )
    retired = False
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        update = payload["update"]
        original_config = load_campaign_config(campaign / "config_v2.yaml")
        if payload["config_sha256"] != original_config.sha256 or payload["best_metric"] is not None:
            raise ValueError("Source checkpoint/config or selection state differs")
        tokens = json.loads(Path(original_config.data.manifest).read_text())["roles"]["train"][
            "sample_tokens"
        ]
        sampler = ResumablePermutationSampler(len(tokens), original_config.seed)
        count = 0
        with (source_run / "metrics.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["update"] > update:
                    break
                count += 1
                if row["update"] != count or row["sample_tokens"] != [
                    tokens[i] for i in sampler.next_batch(4)
                ]:
                    raise ValueError("Committed source ledger differs from the registered sampler")
        state = sampler.state_dict()
        if count != update or any(
            not torch.equal(value, payload["sampler"][key])
            if isinstance(value, torch.Tensor)
            else value != payload["sampler"][key]
            for key, value in state.items()
        ):
            raise ValueError("Source sampler checkpoint differs from consumed images")
        immutable = checkpoint.with_name(f"batch8_transition_{update:06d}.pt")
        os.link(checkpoint, immutable)
        plan = build_plan(
            update, original_config.optimization.updates, list(original_config.evaluation.updates)
        )
        receipt = {
            "schema": "phase_dfull_batch_transition_v1",
            "status": "ADMITTED",
            "authorized_change": "User requested batch 8 after measured throughput gain",
            "source_run": str(source_run),
            "source_checkpoint": str(immutable),
            "source_checkpoint_sha256": sha256_file(immutable),
            "source_config": str(campaign / "config_v2.yaml"),
            "source_config_sha256": original_config.sha256,
            "source_ledger_sha256": sha256_file(source_run / "metrics.jsonl"),
            "experiment_id": "cloud-rgb-dfull-r2-warmstart-seed7-batch8-v1",
            "plan": plan,
            "state_policy": "Preserve model/AdamW/sampler/RNG; map old LR clock by consumed images",
            "precision": "FP32 highest unchanged",
            "created_utc": datetime.now(UTC).isoformat(),
        }
        write("BATCH_TRANSITION.json", receipt)
        config = yaml.safe_load((campaign / "config_v2.yaml").read_text())
        config["experiment_id"] = receipt["experiment_id"]
        config["optimization"].update(
            batch_size=8, updates=plan["total_updates"], checkpoint_interval=512
        )
        for name in ("updates", "full_selection_updates"):
            config["evaluation"][name] = [
                plan["selection_update_map"][str(u)] for u in config["evaluation"][name]
            ]
        (campaign / "config_batch8.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        load_campaign_config(campaign / "config_batch8.yaml")
        producer = project / "output/dfull_v1/producer_batch8_v1"
        subprocess.run(
            [
                sys.executable,
                str(project / "scripts/materialize_detection_batch8.py"),
                "--source",
                str(project / "output/dfull_v1/producer_v2"),
                "--destination",
                str(producer),
                "--module",
                str(project / "src/e_jepa_ttc/detection/batch_transition.py"),
                "--receipt",
                str(campaign / "BATCH_TRANSITION.json"),
            ],
            check=True,
        )
        del payload
        os.kill(args.controller_pid, signal.SIGTERM)
        os.kill(args.trainer_pid, signal.SIGTERM)
        os.kill(args.trainer_pid, getattr(signal, "SIGCONT"))
        retired = True
        watch.terminate()
        for _ in range(100):
            status = Path(f"/proc/{args.trainer_pid}/status")
            if not status.exists() or "\nState:\tZ" in status.read_text():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Original trainer has not exited")
        write(
            "BATCH4_RETIRED.json",
            {
                "status": "MIGRATED_AT_CHECKPOINT",
                "update": update,
                "controller_pid": args.controller_pid,
                "trainer_pid": args.trainer_pid,
                "checkpoint": str(immutable),
            },
        )
        command = [
            sys.executable,
            str(project / "scripts/run_detection_dfull_campaign.py"),
            "--project",
            str(project),
            "--producer",
            str(producer),
            "--campaign",
            str(campaign),
            "--root",
            str(base),
            "--verifier",
            str(
                project
                / "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_queue_v1/scripts"
                / "run_detection_campaign_queue.py"
            ),
            "--batch-transition",
            str(campaign / "BATCH_TRANSITION.json"),
        ]
        with (campaign / "controller_batch8.log").open("x") as log:
            child = subprocess.Popen(
                command, cwd=project, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(
            "BATCH8_SWITCH_STATUS.json",
            {
                "status": "LAUNCHED",
                "source_update": update,
                "controller_pid": child.pid,
                "plan": plan,
            },
        )
        print(json.dumps({"status": "LAUNCHED", "controller_pid": child.pid, "plan": plan}))
    finally:
        if not retired:
            os.kill(args.trainer_pid, getattr(signal, "SIGCONT"))
        watch.terminate()


if __name__ == "__main__":
    main()

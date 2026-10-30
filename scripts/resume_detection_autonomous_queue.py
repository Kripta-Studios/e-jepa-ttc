"""Resume the sealed RGB queue after a user pause without repeating completed jobs."""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def load_queue() -> Any:
    """Reuse the unchanged, separately pinned original supervisor utilities."""
    path = Path(__file__).with_name("run_detection_autonomous_queue.py")
    spec = importlib.util.spec_from_file_location("original_autonomous_queue", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load original supervisor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_no_other_campaign() -> None:
    """Reject duplicate supervisors/trainers by current command, not stale PID."""
    exempt = {os.getpid(), *(process.pid for process in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in exempt or "python" not in (process.info["name"] or "").lower():
            continue
        command = " ".join(process.info["cmdline"] or [])
        if any(
            name in command
            for name in (
                "train_detection_campaign.py",
                "run_detection_autonomous_queue.py",
                "resume_detection_autonomous_queue.py",
            )
        ):
            raise RuntimeError(f"Campaign process already running: {process.pid}")


def segment_identity(
    name: str, base: dict[str, Any], pause: dict[str, Any]
) -> tuple[int, str | None]:
    """Return the exact checkpoint boundary for each adopted/resumed segment."""
    if name == "r0_seed7":
        return base["existing_r0"]["start_update"], base["existing_r0"]["resume"]["sha256"]
    if name == pause["paused_job"]:
        return pause["checkpoint"]["update"], pause["checkpoint"]["sha256"]
    return 0, None


def verify_full(
    helper: Any,
    queue: Any,
    output: Path,
    summary: dict[str, Any],
    pause: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify parent 4096 and child 8192 separately for a resumed R2 segment."""
    if pause is None:
        return helper.verify_full_evaluations(output, summary)
    parent = queue.check(pause["preserved_full_evaluation_4096"])
    kwargs = {
        "protocol_sha256": summary["evaluation_protocol_sha256"],
        "expected_sample_count": summary["full_selection_count"],
    }
    return {
        "4096": helper.verify_evaluation(parent, **kwargs),
        "8192": helper.verify_evaluation(
            output / "evaluation/full_selection_update_008192.json", **kwargs
        ),
    }


def prepare(pause_path: Path, output: Path) -> Path:
    """Create a fresh continuation plan; preparing never launches training."""
    queue = load_queue()
    assert_no_other_campaign()
    pause = queue.read(pause_path)
    base = queue.read(queue.check(pause["queue_plan"]))
    if (
        pause["status"] != "PAUSED_BY_USER"
        or pause["paused_job"] != "r2_seed7"
        or pause["checkpoint"]["update"] != 5376
        or set(pause["completed_jobs"]) != {"r0_seed7", "r1_seed7"}
    ):
        raise ValueError("This continuation requires the recorded R2 pause at update 5376")
    queue.check(base["supervisor_source"])
    for key in ("checkpoint", "selected_checkpoint", "config", "producer", "metrics_at_pause"):
        queue.check(pause[key])
    jobs = copy.deepcopy(base["jobs"])
    jobs["r2_seed7"]["output"] = pause["resume_output_must_be_fresh"]
    if Path(jobs["r2_seed7"]["output"]).exists() or output.exists():
        raise FileExistsError("Continuation supervisor and resumed run outputs must be fresh")
    for name, job in jobs.items():
        if name not in pause["completed_jobs"] and Path(job["output"]).exists():
            raise FileExistsError(f"Uncompleted job output already exists: {name}")
    output.mkdir(parents=True)
    (output / "receipts").mkdir()
    plan = {
        "schema": "phase_ttc_autonomous_rgb_continuation_v1",
        "supervisor_source": queue.binding(Path(__file__)),
        "original_supervisor_source": base["supervisor_source"],
        "pause_manifest": queue.binding(pause_path),
        "base_plan": pause["queue_plan"],
        "jobs": jobs,
        "no_model_api_calls": True,
        "automatic_retries": 0,
        "authorization": "User explicitly requested reanuda todo after the recorded pause",
    }
    queue.write(output / "plan.json", plan)
    return output / "plan.json"


def run(plan_path: Path, expected_sha: str) -> None:
    """Adopt R0/R1, resume R2, then execute the original confirmation rule."""
    queue = load_queue()
    if queue.sha(plan_path) != expected_sha:
        raise ValueError("Continuation plan hash mismatch")
    plan = queue.read(plan_path)
    queue.check(plan["supervisor_source"])
    queue.check(plan["original_supervisor_source"])
    pause = queue.read(queue.check(plan["pause_manifest"]))
    base = queue.read(queue.check(plan["base_plan"]))
    helper = queue.load_helper(queue.check(base["helper"]))
    original_path = queue.check(base["original_plan"])
    original = helper.load_plan(original_path, base["original_plan"]["sha256"])
    assert_no_other_campaign()
    root = plan_path.parent
    with (root / "OWNER.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, stream)
    state: dict[str, Any] = {
        "schema": "phase_ttc_autonomous_rgb_continuation_status_v1",
        "plan_sha256": expected_sha,
        "status": "VERIFYING_COMPLETED_JOBS",
        "active_job": None,
        "completed": {},
        "started_at_utc": datetime.now(UTC).isoformat(),
        "no_model_api_calls": True,
        "automatic_retries": 0,
        "resume_from_update": pause["checkpoint"]["update"],
    }
    records: dict[str, Any] = state["completed"]
    populations: dict[str, Any] | None = None

    def status(value: str, **extra: Any) -> None:
        state.update(status=value, updated_at_utc=datetime.now(UTC).isoformat(), **extra)
        queue.write(root / "status.json", state)

    def verify_bindings() -> None:
        for key in (
            "supervisor_source",
            "original_supervisor_source",
            "pause_manifest",
            "base_plan",
        ):
            queue.check(plan[key])
        helper.verify_all_bindings(original, original_path, base["original_plan"]["sha256"])
        for job in plan["jobs"].values():
            queue.check(job["config"])

    def verify_job(name: str, adopted: bool = False) -> None:
        nonlocal populations
        status("VERIFYING_JOB", active_job=name)
        job = plan["jobs"][name]
        output = Path(job["output"])
        config = helper.load_campaign_config(queue.check(job["config"]))
        start, resume_sha = segment_identity(name, base, pause)
        verified = helper.verify_summary(
            output,
            config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=start,
            resume_sha256=resume_sha,
        )
        if adopted:
            previous = pause["completed_jobs"][name]
            if verified["summary_sha256"] != previous["summary_sha256"]:
                raise ValueError(f"Previously completed summary changed: {name}")
            for item in previous["full_evaluations"].values():
                queue.check(item)
        summary = verified["summary"]
        full = verify_full(helper, queue, output, summary, pause if name == "r2_seed7" else None)
        if populations is None:
            populations = full
        helper.verify_population_same(populations, full)
        records[name] = {
            "metric": summary["best_selection_metric"],
            "seed": config.seed,
            "summary_path": verified["summary_path"],
            "summary_sha256": verified["summary_sha256"],
            "selected_checkpoint": summary["selected_checkpoint"],
            "full_evaluations": {
                key: {k: v for k, v in item.items() if k != "ground_truth_sha256_by_token"}
                for key, item in full.items()
            },
        }
        gc.collect()
        status("VERIFIED_JOB", active_job=name)

    def start_job(name: str, resumed: bool = False) -> None:
        verify_bindings()
        job = plan["jobs"][name]
        config = helper.load_campaign_config(queue.check(job["config"]))
        deadline = time.monotonic() + base["resource_wait_hours"] * 3600
        while True:
            if (root / "STOP_AFTER_CURRENT").exists():
                raise InterruptedError("User requested stop; no next job launched")
            try:
                assert_no_other_campaign()
                xml = subprocess.run(
                    ["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True
                ).stdout
                compute = [
                    node.findtext("pid")
                    for node in ET.fromstring(xml).findall(".//process_info")
                    if node.findtext("type") == "C"
                ]
                if compute:
                    raise RuntimeError(f"Other GPU compute processes: {compute}")
                admission = helper.resource_preflight(config, original["gpu"])
                break
            except RuntimeError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Resource wait exceeded 24 hours") from error
                status("WAITING_FOR_RESOURCES", active_job=name, resource_reason=str(error))
                time.sleep(base["poll_seconds"])
        resume = None
        if resumed:
            resume = queue.check(pause["checkpoint"])
            queue.check(pause["selected_checkpoint"])
        status("RUNNING", active_job=name, resource_admission=admission)
        helper.run_child(
            name=name,
            python=helper._resolve(original["python"]["path"], "python"),
            producer=helper._resolve(original["producer"]["path"], "producer"),
            config_path=Path(job["config"]["path"]),
            config=config,
            output=Path(job["output"]),
            queue_output=root / "receipts",
            gpu=original["gpu"],
            resume=resume,
            stop_after_update=8192 if resumed else None,
        )
        verify_bindings()
        verify_job(name)

    awake = False
    try:
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        status("VERIFYING_COMPLETED_JOBS", prevents_idle_sleep=awake)
        verify_bindings()
        for name in ("r0_seed7", "r1_seed7"):
            verify_job(name, adopted=True)
        queue.check(pause["preserved_full_evaluation_4096"])
        start_job("r2_seed7", resumed=True)
        winner = queue.choose_screen(records)
        queue.write(
            root / "screening_decision.json",
            {
                "winner": winner,
                "metric": "best_selection_metric",
                "records": records.copy(),
            },
        )
        for name in queue.confirmation_jobs(winner):
            start_job(name)
        candidates = ["r0"] if winner == "r0" else ["r0", winner]
        means = {
            recipe: statistics.mean(
                records[f"{recipe}_seed{seed}"]["metric"] for seed in queue.SEEDS
            )
            for recipe in candidates
        }
        queue.write(
            root / "rgb_confirmation.json",
            {
                "status": "LOCAL_RGB_SCREEN_AND_CONFIRMATION_COMPLETE",
                "screening_winner": winner,
                "confirmed_recipe": max(candidates, key=lambda recipe: means[recipe]),
                "three_seed_monitor_means": means,
                "records": records,
                "claim": "Local RGB candidate only; not final SOTA or TTC generalization",
            },
        )
        status("COMPLETED_QUEUED_SCOPE", active_job=None, remaining=base["later_phases_not_queued"])
    except BaseException as error:
        status(
            "STOPPED_BY_REQUEST" if isinstance(error, InterruptedError) else "FAILED_NEEDS_REVIEW",
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main() -> None:
    """Prepare a pinned continuation or execute it after explicit user authorization."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--pause", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(load_queue().binding(prepare(args.pause, args.output))))
    else:
        run(args.plan.resolve(strict=True), args.plan_sha256)


if __name__ == "__main__":
    main()

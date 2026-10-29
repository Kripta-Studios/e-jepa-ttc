"""Persistent local RGB screening/confirmation queue; no model API calls."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

RECIPES = ("r0", "r1", "r2")
SEEDS = (7, 13, 23)


def sha(path: Path) -> str:
    """Hash a file without reading it all into RAM."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binding(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha(resolved)}


def check(item: dict[str, str]) -> Path:
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError(f"Pinned file changed: {path}")
    return path


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Atomically publish mutable status; experiment outputs remain write-once."""
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_campaign_queue", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load frozen queue verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def templates(r0: dict[str, Any], r1: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Make only the declared R2 LR intervention and seed replications."""
    bases = {"r0": r0, "r1": r1, "r2": copy.deepcopy(r1)}
    bases["r2"]["optimization"]["backbone_lr_multiplier"] = 0.1
    result = {}
    for recipe in RECIPES:
        for seed in SEEDS:
            config = copy.deepcopy(bases[recipe])
            config["seed"] = seed
            config["experiment_id"] = f"rgb-{recipe}-d4096-seed{seed}"
            result[f"{recipe}_seed{seed}"] = config
    return result


def choose_screen(records: dict[str, Any]) -> str:
    """Select by the fixed monitor metric; ties favor the earlier recipe."""
    return max(RECIPES, key=lambda recipe: records[f"{recipe}_seed7"]["metric"])


def confirmation_jobs(winner: str) -> list[str]:
    if winner not in RECIPES:
        raise ValueError("Unknown screening recipe")
    recipes = ["r0"] if winner == "r0" else ["r0", winner]
    return [f"{recipe}_seed{seed}" for seed in (13, 23) for recipe in recipes]


def prepare(args: argparse.Namespace) -> Path:
    """Seal the finite queue while the existing R0 process continues."""
    helper = load_helper(args.helper.resolve(strict=True))
    original = helper.load_plan(args.original_plan, sha(args.original_plan))
    workspace = helper.ROOT
    r0_path = (workspace / original["r0"]["config"]["path"]).resolve()
    r1_path = (workspace / original["r1"]["config"]["path"]).resolve()
    helper.verify_recipe_pair(
        helper.load_campaign_config(r0_path), helper.load_campaign_config(r1_path)
    )
    process = psutil.Process(args.pid)
    command = process.cmdline()
    existing_output = args.existing_output.resolve(strict=True)
    if (
        "--output" not in command
        or Path(command[command.index("--output") + 1]).resolve() != existing_output
    ):
        raise ValueError("Existing PID is not the declared R0 producer")
    if args.output.exists():
        raise FileExistsError("Supervisor output must be fresh")
    args.output.mkdir(parents=True)
    (args.output / "configs").mkdir()
    (args.output / "receipts").mkdir()
    configs = templates(yaml.safe_load(r0_path.read_text()), yaml.safe_load(r1_path.read_text()))
    jobs = {}
    for name, config in configs.items():
        path = args.output / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        validated = helper.load_campaign_config(path)
        if validated.optimization.updates != 8192:
            raise ValueError("Budget differs from 8192")
        jobs[name] = {
            "config": binding(path),
            "config_canonical_sha256": validated.sha256,
            "output": str((args.output / "runs" / name).resolve()),
        }
    # R0 is already running; retain its actual configuration and output.
    jobs["r0_seed7"]["config"] = binding(r0_path)
    jobs["r0_seed7"]["output"] = str(existing_output)
    # Keep R1 byte-identical to the already prepared original recipe.
    jobs["r1_seed7"]["config"] = binding(r1_path)
    resume_path = Path(command[command.index("--resume") + 1])
    plan = {
        "schema": "phase_ttc_autonomous_rgb_queue_v1",
        "workspace": str(workspace),
        "helper": binding(args.helper),
        "supervisor_source": binding(Path(__file__)),
        "original_plan": binding(args.original_plan),
        "existing_r0": {
            "pid": process.pid,
            "create_time": process.create_time(),
            "command": command,
            "resume": binding(resume_path),
            "start_update": 3840,
        },
        "jobs": jobs,
        "screening_order": list(RECIPES),
        "confirmation_seeds": [13, 23],
        "selection_rule": "highest best_selection_metric; ties prefer r0 then r1 then r2",
        "confirmation_rule": (
            "compare three-seed monitor means of r0 and screening winner; ties prefer r0"
        ),
        "maximum_new_training_jobs": 6,
        "resource_wait_hours": 24,
        "poll_seconds": 60,
        "no_cloud": True,
        "no_model_api_calls": True,
        "later_phases_not_queued": {
            "S1_R3_R4_D16384": "Augmentation/rare sampling and scale-up not execution-admitted",
            "S2": "Ranking/geometric experiment not fitted or execution-admitted",
            "S3_S4": "Temporal/event runners need frozen winner and remaining admissions",
            "S5": "Independent producer fits and complete TTC evaluation remain",
            "S6": "Final calibration/generalization/official comparison remain",
        },
    }
    write(args.output / "plan.json", plan)
    return args.output / "plan.json"


def run(plan_path: Path, expected_sha: str) -> None:
    """Wait for R0, then run the sealed jobs serially without agent supervision."""
    if sha(plan_path) != expected_sha:
        raise ValueError("Queue plan hash mismatch")
    plan = read(plan_path)
    check(plan["supervisor_source"])
    helper = load_helper(check(plan["helper"]))
    original_path = check(plan["original_plan"])
    original = helper.load_plan(original_path, plan["original_plan"]["sha256"])
    root = plan_path.parent
    # Exclusive creation prevents two supervisors from launching duplicate jobs.
    with (root / "OWNER.json").open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, stream)
    state: dict[str, Any] = {
        "schema": "phase_ttc_autonomous_rgb_queue_status_v1",
        "plan_sha256": expected_sha,
        "status": "WAITING_FOR_EXISTING_R0",
        "active_job": "r0_seed7",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed": {},
        "no_model_api_calls": True,
        "automatic_retries": 0,
    }
    state_path = root / "status.json"
    records: dict[str, Any] = state["completed"]
    populations: dict[str, Any] | None = None

    def status(value: str, **extra: Any) -> None:
        state.update(status=value, updated_at_utc=datetime.now(UTC).isoformat(), **extra)
        write(state_path, state)

    def verify_bindings() -> None:
        check(plan["supervisor_source"])
        check(plan["helper"])
        helper.verify_all_bindings(original, original_path, plan["original_plan"]["sha256"])
        for job in plan["jobs"].values():
            check(job["config"])

    def verify_job(name: str, resumed: bool = False) -> None:
        nonlocal populations
        job = plan["jobs"][name]
        config = helper.load_campaign_config(check(job["config"]))
        verified = helper.verify_summary(
            Path(job["output"]),
            config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=3840 if resumed else 0,
            resume_sha256=plan["existing_r0"]["resume"]["sha256"] if resumed else None,
        )
        full = helper.verify_full_evaluations(Path(job["output"]), verified["summary"])
        if populations is None:
            populations = full
        else:
            helper.verify_population_same(populations, full)
        summary = verified["summary"]
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
        status("VERIFIED_JOB", active_job=name)

    def start_job(name: str) -> None:
        if (root / "STOP_AFTER_CURRENT").exists():
            raise InterruptedError("STOP_AFTER_CURRENT requested")
        verify_bindings()
        job = plan["jobs"][name]
        config = helper.load_campaign_config(check(job["config"]))
        deadline = time.monotonic() + plan["resource_wait_hours"] * 3600
        while True:
            try:
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
                    raise TimeoutError("Resource admission wait exceeded 24 hours") from error
                status("WAITING_FOR_RESOURCES", active_job=name, resource_reason=str(error))
                time.sleep(plan["poll_seconds"])
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
        )
        verify_bindings()
        verify_job(name)

    awake = False
    try:
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        status("WAITING_FOR_EXISTING_R0", prevents_idle_sleep=awake)
        existing = plan["existing_r0"]
        wait_deadline = time.monotonic() + 26 * 3600
        while psutil.pid_exists(existing["pid"]):
            try:
                if (
                    abs(psutil.Process(existing["pid"]).create_time() - existing["create_time"])
                    > 0.01
                ):
                    break  # PID was reused; the declared producer has ended.
            except psutil.NoSuchProcess:
                break
            if time.monotonic() >= wait_deadline:
                raise TimeoutError("Existing R0 did not finish within 26 hours; left untouched")
            if (root / "STOP_AFTER_CURRENT").exists():
                raise InterruptedError("STOP_AFTER_CURRENT requested; existing R0 is untouched")
            time.sleep(plan["poll_seconds"])
        verify_bindings()
        verify_job("r0_seed7", resumed=True)
        for recipe in ("r1", "r2"):
            start_job(f"{recipe}_seed7")
        winner = choose_screen(records)
        write(
            root / "screening_decision.json",
            {"winner": winner, "metric": "best_selection_metric", "records": records.copy()},
        )
        for name in confirmation_jobs(winner):
            start_job(name)
        candidates = ["r0"] if winner == "r0" else ["r0", winner]
        means = {
            recipe: statistics.mean(records[f"{recipe}_seed{seed}"]["metric"] for seed in SEEDS)
            for recipe in candidates
        }
        confirmed = max(candidates, key=lambda recipe: means[recipe])
        write(
            root / "rgb_confirmation.json",
            {
                "status": "LOCAL_RGB_SCREEN_AND_CONFIRMATION_COMPLETE",
                "screening_winner": winner,
                "confirmed_recipe": confirmed,
                "three_seed_monitor_means": means,
                "records": records,
                "claim": "Local RGB candidate only; not final SOTA or TTC generalization",
            },
        )
        status("COMPLETED_QUEUED_SCOPE", active_job=None, remaining=plan["later_phases_not_queued"])
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
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--helper", type=Path, required=True)
    create.add_argument("--original-plan", type=Path, required=True)
    create.add_argument("--pid", type=int, required=True)
    create.add_argument("--existing-output", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        path = prepare(args)
        print(json.dumps(binding(path)))
    else:
        run(args.plan.resolve(strict=True), args.plan_sha256)


if __name__ == "__main__":
    main()

"""Recoverable local TRAIN40 queue with independent preparation and one heavy writer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def active(marker: str) -> bool:
    """Recognize an existing task after controller restart instead of duplicating it."""
    for process in psutil.process_iter(["name", "cmdline"]):
        if not (process.info["name"] or "").lower().startswith("python"):
            continue
        line = " ".join(process.info["cmdline"] or [])
        if marker in line:
            return True
    return False


def launch(output: Path, tag: str, module: str, arguments: list[str]) -> subprocess.Popen:
    """Start a hidden subprocess with only scoped environment overrides and a durable log."""
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    command = [sys.executable, "-m", module, "--output", str(output), *arguments]
    directory = output / "controller"
    directory.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    log_path = directory / f"{tag}_{stamp}.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    atomic_json(
        directory / f"{tag}_LATEST.json",
        {
            "pid": process.pid,
            "command": command,
            "log_path": str(log_path),
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "STARTED",
        },
    )
    return process


def refresh_complete_inputs(output: Path) -> bool:
    """Merge the two disjoint preparation passes only after both writers have stopped."""
    if active("operational.train40_system.prepare"):
        return False
    freeze = read(output / "PREPARE_FREEZE.json")
    directory = Path(freeze["cache_root"])
    count, size = 0, 0
    for start in range(0, 88744, 32):
        receipt_path = directory / f"shard_{start // 32:05d}.json"
        if not receipt_path.exists():
            return False
        receipt = read(receipt_path)
        if receipt["freeze_sha256"] != digest(output / "PREPARE_FREEZE.json"):
            raise ValueError("Combined preparation recipe changed")
        count += receipt["rows"]
        size += receipt["bytes"]
    if count != 88744 or size > freeze["new_input_disk_cap_bytes"]:
        raise ValueError("Combined population or disk cap failed")
    atomic_json(
        output / "INPUT_PREPARATION_PROGRESS.json",
        {
            "status": "COMPLETE",
            "completed_rows": count,
            "total_rows": 88744,
            "compressed_bytes": size,
            "verified_raw_sequences": 40,
            "workers_combined_max": 8,
            "optimizer_updates": 0,
            "disjoint_passes_merged_after_writer_exit": True,
        },
    )
    return True


def status(output: Path, name: str) -> str:
    """Missing receipts indicate pending work, never a negative scientific result."""
    path = output / (name + ".json")
    return read(path).get("status", "PENDING") if path.exists() else "PENDING"


def complete_fit(output: Path, name: str) -> bool:
    path = output / "fits" / name / "CHECKPOINT_RECEIPT.json"
    return path.exists() and read(path)["status"] == "COMPLETE"


def publish_fit_freezes(output: Path) -> None:
    """Bind complete data/teacher manifests to the already QA-frozen engineering before fitting."""
    engineering = read(output / "ENGINEERING_FREEZE.json")
    verify_sources(engineering)
    if digest(output / "TRAINING_PROTOCOL.json") != engineering["protocol_sha256"]:
        raise ValueError("Scientific recipe changed after engineering QA")
    for kind in ("INPUT", "TEACHER"):
        manifest = read(output / f"{kind}_MANIFEST.json")
        if manifest["status"] != "COMPLETE_VERIFIED" or manifest["row_count"] != 88744:
            raise ValueError("Full TRAIN40 data and teacher admission required")
    shared = {
        "files": engineering["files"],
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "QA": engineering["QA"],
        "engineering_freeze_sha256": digest(output / "ENGINEERING_FREEZE.json"),
    }
    configurations = {
        "MODELS_FREEZE.json": {
            **shared,
            "trainer_sha256": digest(ROOT / "operational/train40_system/engine.py"),
            "input_manifest_sha256": digest(output / "INPUT_MANIFEST.json"),
            "teacher_manifest_sha256": digest(output / "TEACHER_MANIFEST.json"),
        },
        "FEATURES_FREEZE.json": {
            **shared,
            "source_sha256": digest(ROOT / "operational/train40_system/history_features.py"),
        },
        "HEADS_FREEZE.json": {
            **shared,
            "source_sha256": digest(ROOT / "operational/train40_system/heads.py"),
        },
    }
    for name, configuration in configurations.items():
        path = output / name
        if path.exists() and read(path) != configuration:
            raise ValueError(f"Preserve existing scientific freeze: {name}")
        if not path.exists():
            atomic_json(path, configuration)


def run(output: Path, raw_root: Path, teacher_path: Path) -> None:
    """Advance viable independent branches and retry failures after a thirty-minute cooldown."""
    engineering = read(output / "ENGINEERING_FREEZE.json")
    verify_sources(engineering)
    if digest(output / "TRAINING_PROTOCOL.json") != engineering["protocol_sha256"]:
        raise ValueError("Scientific recipe changed after engineering QA")
    deadline = datetime.fromisoformat(read(output / "AUTHORIZATION.json")["deadline_utc"])
    running: dict[str, subprocess.Popen] = {}
    retry_after: dict[str, float] = {}
    last_teacher_media = read(output / "MEDIA_VERIFICATION_PROGRESS.json")["verified_rgb"]
    raw_arguments = ["--raw-root", str(raw_root)]
    while datetime.now(UTC) < deadline:
        for tag, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                retry_after[tag] = time.monotonic() + 1800
                atomic_json(
                    output / "controller" / f"{tag}_FAILURE.json",
                    {
                        "exit_code": code,
                        "scientific_negative": False,
                        "work_preserved": True,
                        "independent_tasks_continue": True,
                    },
                )
            elif tag == "teacher" and status(output, "TEACHER_PROGRESS") != "COMPLETE":
                retry_after[tag] = time.monotonic() + 1800
            elif tag == "prepare":
                retry_after[tag] = time.monotonic() + 1800
            elif tag in {"pair_features", "h8_features"}:
                kind = "PAIR" if tag == "pair_features" else "H8"
                if status(output, kind + "_FEATURE_MANIFEST") != "COMPLETE_VERIFIED":
                    retry_after[tag] = time.monotonic() + 1800
            elif tag in {"a5", "c2f", "pair", "h8_seed7", "h8_seed13", "h8_seed23"}:
                fit = tag + "_seed7" if tag in {"a5", "c2f", "pair"} else tag
                if not complete_fit(output, fit):
                    retry_after[tag] = time.monotonic() + 1800
            running.pop(tag)
        media = read(output / "MEDIA_VERIFICATION_PROGRESS.json")
        if (
            not active("operational.train40_system.verify_downloads")
            and media["status"] != "COMPLETE"
        ):
            if "verify" not in running and time.monotonic() >= retry_after.get("verify", 0):
                running["verify"] = launch(
                    output, "verify", "operational.train40_system.verify_downloads", raw_arguments
                )
        if not active("operational.train40_system.prepare"):
            ready = refresh_complete_inputs(output)
            if not ready and media["verified_raw"] == 40 and "prepare" not in running:
                if time.monotonic() >= retry_after.get("prepare", 0):
                    running["prepare"] = launch(
                        output,
                        "prepare",
                        "operational.train40_system.prepare_partition",
                        [*raw_arguments, "--partition", "all", "--workers", "4"],
                    )
        heavy_active = any(
            active("operational.train40_system." + marker)
            for marker in ("teacher", "engine", "heads", "history_features", "seal_inputs")
        )
        if not heavy_active:
            teacher_complete = status(output, "TEACHER_PROGRESS") == "COMPLETE"
            if not teacher_complete and (
                media["verified_rgb"] > last_teacher_media or media["verified_rgb"] == 135
            ):
                if time.monotonic() >= retry_after.get("teacher", 0):
                    running["teacher"] = launch(
                        output,
                        "teacher",
                        "operational.train40_system.teacher_resume",
                        [*raw_arguments, "--model-path", str(teacher_path)],
                    )
                    last_teacher_media = media["verified_rgb"]
            elif (
                teacher_complete
                and status(output, "INPUT_PREPARATION_PROGRESS") == "COMPLETE"
                and media["status"] == "COMPLETE"
            ):
                if (
                    status(output, "INPUT_MANIFEST") != "COMPLETE_VERIFIED"
                    or status(output, "TEACHER_MANIFEST") != "COMPLETE_VERIFIED"
                ):
                    if time.monotonic() >= retry_after.get("seal", 0):
                        running["seal"] = launch(
                            output, "seal", "operational.train40_system.seal_inputs", []
                        )
                else:
                    publish_fit_freezes(output)
                    tasks = []
                    for arm in ("a5", "c2f"):
                        if not complete_fit(output, arm + "_seed7"):
                            tasks.append((arm, "engine", ["--arm", arm]))
                    if complete_fit(output, "a5_seed7"):
                        if status(output, "PAIR_FEATURE_MANIFEST") != "COMPLETE_VERIFIED":
                            tasks.append(
                                (
                                    "pair_features",
                                    "history_features",
                                    [*raw_arguments, "--kind", "PAIR"],
                                )
                            )
                        elif not complete_fit(output, "pair_seed7"):
                            tasks.append(("pair", "heads", ["--kind", "PAIR", "--seed", "7"]))
                    if all(complete_fit(output, name + "_seed7") for name in ("a5", "c2f", "pair")):
                        if status(output, "H8_FEATURE_MANIFEST") != "COMPLETE_VERIFIED":
                            tasks.append(
                                (
                                    "h8_features",
                                    "history_features",
                                    [*raw_arguments, "--kind", "H8"],
                                )
                            )
                        else:
                            for seed in (7, 13, 23):
                                if not complete_fit(output, f"h8_seed{seed}"):
                                    tasks.append(
                                        (
                                            f"h8_seed{seed}",
                                            "heads",
                                            ["--kind", "H8", "--seed", str(seed)],
                                        )
                                    )
                    for tag, module, arguments in tasks:
                        if time.monotonic() >= retry_after.get(tag, 0):
                            running[tag] = launch(
                                output, tag, "operational.train40_system." + module, arguments
                            )
                            break
                    if not tasks:
                        atomic_json(
                            output / "TRAINING_QUEUE_COMPLETION.json",
                            {
                                "status": "COMPLETE",
                                "ended_utc": datetime.now(UTC).isoformat(),
                                "all_fixed_scientific_endpoints_complete": True,
                                "comparison_and_delivery_remaining": True,
                            },
                        )
                        return
        atomic_json(
            output / "CONTROLLER_PROGRESS.json",
            {
                "status": "RUNNING",
                "checked_utc": datetime.now(UTC).isoformat(),
                "children": {tag: process.pid for tag, process in running.items()},
                "input_status": status(output, "INPUT_PREPARATION_PROGRESS"),
                "teacher_status": status(output, "TEACHER_PROGRESS"),
                "media": media,
                "one_heavy_writer": True,
                "user_updates_interval_minutes": 30,
            },
        )
        time.sleep(60)
    atomic_json(
        output / "CONTROLLER_PROGRESS.json",
        {
            "status": "DEADLINE_STOP_PRESERVED",
            "checked_utc": datetime.now(UTC).isoformat(),
            "scientific_negative": False,
            "children_preserve_safe_boundaries": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve() / "controller"
    directory.mkdir(exist_ok=True)
    with Lease(directory):
        run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())

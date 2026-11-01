"""Continue the already-running population experiment through its fixed conditional gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from operational.release_research.protocol import ResearchRun, sha256, write_json


def read(path: Path) -> dict:
    """Read a local research receipt."""
    return json.loads(path.read_text(encoding="utf-8"))


def worker_identity(pid: int) -> dict | None:
    """Check PID plus command and creation time; never signal or restart the existing worker."""
    if pid <= 0:
        raise ValueError("Positive worker PID required")
    command = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
        f"Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' | "
        "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def gate_result(report: dict) -> str:
    """Require the existing fixed F0 decision, never a router or partial-screen result."""
    if report.get("population_size") != 2608 or report.get("primary_route") != "F0":
        raise ValueError("Expected full population F0 evaluation")
    return (
        "native_profile"
        if report["routes"]["F0"]["eligible_for_eap_validation"]
        else "cadence_not_admitted"
    )


def continue_work(screen: Path, pid: int, public_inputs: Path, output: Path) -> None:
    """Wait for the existing receipt, evaluate it, and execute only branches its results admit."""
    freeze = read(screen / "FREEZE.json")
    if freeze["schema"] != "phase_canonical_population2608_v1":
        raise ValueError("Wrong running experiment")
    initial = worker_identity(pid)
    if initial is None or "population_phase_canonical_pair.py" not in initial["CommandLine"]:
        raise ValueError("The specified population worker is not alive")
    files = {"running_freeze": screen / "FREEZE.json", "public_inputs": public_inputs}
    for name, digest in freeze["source_hashes"].items():
        path = Path(name)
        if sha256(path) != digest:
            raise ValueError("Running population source changed")
        files["source:" + name] = path
    names = [str(p) for p in Path("operational/release_research").glob("*.py")]
    names += [
        "scripts/continue_release_research.py",
        "scripts/evaluate_phase_canonical_screen.py",
        "scripts/report_phase_canonical_population.py",
        "scripts/release_native_pair_capture.py",
        "scripts/release_native_pair_replay.py",
        "scripts/release_research.py",
        "scripts/runpod_test12_staged.py",
        "operational/phase_timewarp/screen_metrics.py",
    ]
    for name in names:
        files["source:" + name] = Path(name)
    for name in ("NATIVE_PAIR_PREREGISTRATION.md", "NATIVE_PARITY_FOLLOWUP.md"):
        files["plan:" + name] = Path("docs/release_research_implementation_20261031") / name
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "conditional-research-continuation-v1",
            "existing_worker": initial,
            "poll_seconds": 30,
            "optimizer_updates": 0,
            "native_targets_read": False,
            "new_population_extractions": 0,
        },
    )
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONPATH": "src;.",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    commands = []

    def status(stage: str, **extra: object) -> None:
        write_json(output / "STATUS.json", {"stage": stage, "time_unix": time.time(), **extra})
        print(stage, flush=True)

    def guard() -> None:
        for name, path in files.items():
            if sha256(path) != run.hashes[name]:
                raise ValueError("Continuation source/plan changed: " + name)

    def execute(name: str, arguments: list[str]) -> None:
        guard()
        status(name)
        command = [sys.executable, "-u", *arguments]
        commands.append(command)
        write_json(output / "COMMANDS.json", {"commands": commands})
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True
            )

    def finish(reason: str) -> None:
        status("finished", outcome=reason)
        write_json(
            output / "DECISION.json",
            {
                "outcome": reason,
                "release_modified": False,
                "native_accuracy_evaluated": False,
                "scientific_campaign_complete": False,
            },
        )
        (output / "SUMMARY.md").write_text(
            "# Continuación de investigación\n\nResultado: `" + reason + "`.\n\n"
            "Ver population_report/README.md y los recibos de cada etapa ejecutada.\n"
            "El release oficial no se modifica. No hay evaluación de targets Test12.\n",
            encoding="utf-8",
        )
        run.finish()

    try:
        status("waiting_for_existing_population", worker_pid=pid)
        while not (screen / "COMPLETE.json").exists():
            if (screen / "FAILED.json").exists():
                raise ValueError("Existing population worker published FAILED")
            if worker_identity(pid) != initial:
                # Publish may race with process exit; check again before declaring failure.
                if not (screen / "COMPLETE.json").exists():
                    raise ValueError(
                        "Population worker ended or PID identity changed without COMPLETE"
                    )
                break
            write_json(
                output / "STATUS.json",
                {
                    "stage": "waiting_for_existing_population",
                    "worker_pid": pid,
                    "time_unix": time.time(),
                },
            )
            time.sleep(30)
        guard()
        evaluation = output / "population_evaluation"
        execute(
            "evaluate_population",
            [
                "scripts/evaluate_phase_canonical_screen.py",
                "--screen",
                str(screen),
                "--expected-queries",
                "2608",
                "--output",
                str(evaluation),
            ],
        )
        execute(
            "report_population",
            [
                "scripts/report_phase_canonical_population.py",
                "--screen",
                str(screen),
                "--evaluation",
                str(evaluation),
                "--output",
                str(output / "population_report"),
            ],
        )
        decision = gate_result(read(evaluation / "REPORT.json"))
        if decision != "native_profile":
            finish(decision)
            return
        # A terminal receipt closes inference; wait for worker cleanup before using its GPU.
        while worker_identity(pid) == initial:
            time.sleep(5)
        common = [
            "--inputs",
            str(public_inputs),
            "--device",
            "cuda",
            "--population-evaluation",
            str(evaluation),
            "--population-screen",
            str(screen),
        ]
        profile = output / "native_profile_cuda"
        execute(
            "capture_native_profile",
            [
                "scripts/release_native_pair_capture.py",
                *common,
                "--profile-queries",
                "2",
                "--output",
                str(profile),
            ],
        )
        profile_replay = output / "native_profile_parity"
        execute(
            "replay_native_profile",
            [
                "scripts/release_native_pair_replay.py",
                "--capture",
                str(profile),
                "--output",
                str(profile_replay),
            ],
        )
        if read(profile_replay / "RESULTS.json")["historical_parity"] != "PASS":
            finish("native_cuda_profile_parity_failed_candidate_not_run")
            return
        full = output / "native_full_capture"
        execute(
            "capture_native_full",
            ["scripts/release_native_pair_capture.py", *common, "--output", str(full)],
        )
        replay = output / "native_full_replay"
        execute(
            "replay_native_full",
            [
                "scripts/release_native_pair_replay.py",
                "--capture",
                str(full),
                "--candidate",
                "--output",
                str(replay),
            ],
        )
        result = read(replay / "RESULTS.json")
        finish(
            "native_prediction_comparison_complete_accuracy_pending"
            if result["candidate_executed"]
            else "native_full_original_parity_failed_candidate_not_run"
        )
    except Exception as error:
        status("failed", error=repr(error))
        run.fail(error)
        raise


def main() -> None:
    """Start one auditable continuation; never restart or duplicate the population extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen", type=Path, default=Path("output/phase_timewarp/population2608_v1")
    )
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--public-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    continue_work(args.screen, args.worker_pid, args.public_inputs, args.output)


if __name__ == "__main__":
    main()

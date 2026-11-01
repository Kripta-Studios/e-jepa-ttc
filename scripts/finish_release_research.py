"""Wait for the existing native diagnostic, then measure latency and publish verified results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.continue_release_research import worker_identity
from scripts.release_research import read, sources


def finish(native: Path, pid: int, output: Path, report: Path) -> None:
    """Never restart extraction or treat a partial parent as an admitted benchmark input."""
    initial = worker_identity(pid)
    if initial is None or "continue_native_platform_diagnostic.py" not in initial["CommandLine"]:
        raise ValueError("Expected the existing native diagnostic continuation")
    if report.exists():
        raise ValueError("Report destination must be fresh")
    files = {**sources(), "native_freeze": native / "FREEZE.json"}
    for name in (
        *DEPENDENCIES,
        "scripts/finish_release_research.py",
        "scripts/release_native_local_latency.py",
        "scripts/report_release_research_campaign.py",
        "scripts/continue_release_research.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_robust/bundle.py",
    ):
        files["source:" + name] = Path(name)
    files["latency_protocol"] = Path(
        "docs/release_research_implementation_20261031/NATIVE_LOCAL_LATENCY_PROTOCOL.md"
    )
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "research-final-reporting-continuation-v1",
            "native_worker_identity": initial,
            "report_output": str(report),
            "targets_read": False,
            "optimizer_updates": 0,
            "scope": "native local sensitivity and latency; no historical promotion",
        },
    )
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONPATH": "src;.",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    def status(stage: str) -> None:
        write_json(output / "STATUS.json", {"stage": stage, "time_unix": time.time()})

    def guard() -> None:
        for key, path in files.items():
            if sha256(path) != run.hashes[key]:
                raise ValueError("Reporting source/protocol changed: " + key)

    commands = []

    def execute(stage: str, arguments: list[str]) -> None:
        guard()
        status(stage)
        command = [sys.executable, "-u", *arguments]
        commands.append(command)
        write_json(output / "COMMANDS.json", commands)
        print(stage, flush=True)
        with (output / (stage + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)

    try:
        while not (native / "COMPLETE.json").exists():
            if (native / "FAILED.json").exists():
                raise ValueError("Native continuation failed; latency not admitted")
            if worker_identity(pid) != initial and not (native / "COMPLETE.json").exists():
                raise ValueError("Native worker ended without COMPLETE")
            status("waiting_for_native_complete")
            time.sleep(30)
        while worker_identity(pid) == initial:
            time.sleep(5)
        latency = output / "latency"
        execute(
            "latency",
            [
                "scripts/release_native_local_latency.py",
                "--parent",
                str(native),
                "--output",
                str(latency),
            ],
        )
        execute(
            "report",
            [
                "scripts/report_release_research_campaign.py",
                "--native",
                str(native),
                "--latency",
                str(latency),
                "--output",
                str(report),
            ],
        )
        write_json(
            output / "RESULTS.json",
            {
                "report": str(report),
                "report_complete_sha256": sha256(report / "COMPLETE.json"),
                "decision": read(report / "DECISION.json"),
            },
        )
        status("finished")
        run.finish()
    except Exception as error:
        status("failed")
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    finish(args.native, args.worker_pid, args.output, args.report)


if __name__ == "__main__":
    main()

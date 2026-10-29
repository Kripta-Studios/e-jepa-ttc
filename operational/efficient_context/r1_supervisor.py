"""V2 R1 supervisor: isolated attempt logs, finite recovery, exact process checks."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Lease, atomic_bytes, atomic_json, digest, read
from .r1_chain import DEADLINE, bundle, stamp


def heavy_entrypoint(name: str, command: list[str]) -> bool:
    """Inspect real Python entrypoints, never text embedded in shell commands."""
    if not name.lower().startswith("python"):
        return False
    if "-m" in command:
        position = command.index("-m") + 1
        if position >= len(command):
            return False
        entrypoint = command[position].lower()
    elif len(command) > 1 and command[1].endswith(".py"):
        entrypoint = Path(command[1]).stem.lower()
    else:
        return False
    if re.search(r"stage(?:70|71|72|73|74|75|76)(?:\D|$)", " ".join(command).lower()):
        return True
    markers = (
        "operational.efficient_context.garl_train",
        "operational.simplex_t_h16_replication.train",
        "operational.train40_system.engine",
        "operational.train40_system.history_resources",
        "operational.train40_system.garl_predictions",
        "operational.train40_system.feature_resources",
    )
    return entrypoint.startswith(markers) or entrypoint in ("train_baseline", "pretrain_jepa")


def external_heavy_pids() -> list[int]:
    """Return actual Python trainer/inference processes admitted elsewhere."""
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if heavy_entrypoint(process.info["name"] or "", process.info["cmdline"] or []):
                found.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return found


def failure_kind(text: str) -> str:
    """Classify only the terminal exception from one isolated attempt log."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    terminal = lines[-1] if lines else ""
    if terminal.startswith(("ValueError:", "AssertionError:")):
        return "INTEGRITY_OR_CONTRACT_FAILURE"
    if terminal.startswith(("InterruptedError:", "OSError:", "PermissionError:")):
        return "TRANSIENT"
    if terminal.startswith("FileNotFoundError:"):
        return "MISSING_DEPENDENCY"
    return "PERMANENT_FAILURE"


def phase_stop(output: Path, phase: str, message: bytes) -> None:
    """Request a preserved stop at the child phase's real guard location."""
    location = output / "window_parity" if phase == "window_parity" else output
    target = location / "STOP_REQUEST"
    if not target.exists():
        atomic_bytes(target, message)


def run(output: Path) -> None:
    """Complete the already-admitted engineering queue within its fixed deadline."""
    output.mkdir(parents=True, exist_ok=True)
    files = list((ROOT / "operational/efficient_context").glob("r1*.py"))
    files += [ROOT / "tests/unit/test_r1_reader.py", ROOT / "tests/unit/test_r1_supervisor.py"]
    freeze = dict(
        sources={str(p.relative_to(ROOT)): digest(p) for p in files},
        optimizer_updates=0,
        queries=64,
        blocks=3,
        arms=["H1", "H8", "H16", "WIDE"],
        deadline_utc=DEADLINE.isoformat(),
        retry_delay_seconds=1800,
        max_attempts=4,
        recovery="each attempt has a separate log; permanent/integrity errors never retried",
    )
    frozen = output / "SUPERVISOR_V2_FREEZE.json"
    if frozen.exists() and read(frozen) != freeze:
        raise ValueError("R1 supervisor v2 source freeze changed")
    if not frozen.exists():
        atomic_json(frozen, freeze)
    phases = (
        ("window_parity", "operational.efficient_context.r1_parity", output / "window_parity"),
        ("measurement", "operational.efficient_context.r1_measure", output),
    )
    with Lease(output):
        for phase, module, destination in phases:
            result = output / phase / "RESULT.json"
            if result.exists():
                continue
            attempts = 0
            retry_after = 0.0
            while not result.exists():
                if datetime.now(UTC) >= DEADLINE:
                    atomic_json(
                        output / "CHAIN_STATE.json",
                        dict(
                            status="DEADLINE_PRESERVED",
                            checked_utc=stamp(),
                            phase=phase,
                            optimizer_updates=0,
                        ),
                    )
                    return
                stop = (output / "STOP_REQUEST").exists() or (
                    phase == "window_parity" and (output / "window_parity/STOP_REQUEST").exists()
                )
                heavy = external_heavy_pids() if phase == "measurement" else []
                admitted = phase != "measurement" or (output / "GPU_SLOT_GRANTED.json").exists()
                if stop or heavy or not admitted or time.monotonic() < retry_after:
                    atomic_json(
                        output / "CHAIN_STATE.json",
                        dict(
                            status="WAITING",
                            phase=phase,
                            stop_requested=stop,
                            heavy_pids=heavy,
                            gpu_slot_admitted=admitted,
                            retry_remaining_s=max(0, retry_after - time.monotonic()),
                            checked_utc=stamp(),
                            optimizer_updates=0,
                        ),
                    )
                    time.sleep(30)
                    continue
                attempts += 1
                command = [sys.executable, "-m", module, "--output", str(destination)]
                log_path = output / "attempts" / f"{phase}_{time.time_ns()}.log"
                log_path.parent.mkdir(exist_ok=True)
                with log_path.open("xb", buffering=0) as log:
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    while child.poll() is None:
                        deadline = datetime.now(UTC) >= DEADLINE
                        heavy = external_heavy_pids() if phase == "measurement" else []
                        if deadline:
                            phase_stop(output, phase, b"R1 deadline; preserve all work\n")
                        elif heavy:
                            phase_stop(output, phase, b"R1 external heavy process; preserve work\n")
                        elif (output / "STOP_REQUEST").exists():
                            phase_stop(output, phase, b"R1 explicit stop requested\n")
                        atomic_json(
                            output / "CHAIN_STATE.json",
                            dict(
                                status="STOPPING_PRESERVED" if deadline or heavy else "RUNNING",
                                supervisor="v2",
                                phase=phase,
                                child_pid=child.pid,
                                command=command,
                                log=str(log_path),
                                attempt=attempts,
                                checked_utc=stamp(),
                                optimizer_updates=0,
                            ),
                        )
                        time.sleep(30)
                if child.returncode == 0 and result.exists():
                    break
                text = log_path.read_text(encoding="utf-8", errors="replace")
                kind = failure_kind(text)
                atomic_json(
                    log_path.with_suffix(".json"),
                    dict(
                        returncode=child.returncode,
                        failure_kind=kind,
                        phase=phase,
                        checked_utc=stamp(),
                        log_sha256=digest(log_path),
                        attempt=attempts,
                    ),
                )
                if kind != "TRANSIENT" or attempts >= 4:
                    atomic_json(
                        output / "CHAIN_STATE.json",
                        dict(
                            status="FAILED_PRESERVED",
                            phase=phase,
                            failure_kind=kind,
                            checked_utc=stamp(),
                            log=str(log_path),
                            optimizer_updates=0,
                        ),
                    )
                    return
                retry_after = time.monotonic() + 1800
        bundle(output)
        atomic_json(
            output / "CHAIN_STATE.json",
            dict(status="COMPLETE", supervisor="v2", checked_utc=stamp(), optimizer_updates=0),
        )


def main() -> int:
    """Start or resume R1 v2 without repeating optimizer work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/efficient_context_20261004/r1_20261008"
    )
    run(parser.parse_args().output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

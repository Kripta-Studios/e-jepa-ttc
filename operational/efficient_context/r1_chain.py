"""Durable zero-update R1 admission, profiling, and essential evidence delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Lease, atomic_bytes, atomic_json, digest, read

DEADLINE = datetime(2026, 10, 9, 18, 31, 35, tzinfo=UTC)


def external_heavy_pids() -> list[int]:
    """Detect other admitted trainer/inference entrypoints before taking GPU."""
    import psutil

    markers = (
        "operational.efficient_context.garl_train",
        "operational.simplex_t_h16_replication.train",
        "operational.train40_system.engine",
        "operational.train40_system.history_resources",
        "operational.train40_system.garl_predictions",
        "operational.train40_system.feature_resources",
    )
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if any(marker in command for marker in markers):
                found.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return found


def stamp() -> str:
    """Return an explicit UTC timestamp for durable operational evidence."""
    return datetime.now(UTC).isoformat()


def bundle(output: Path) -> None:
    """Generate engineering-only report and independently verify every ZIP member."""
    result = read(output / "measurement/RESULT.json")
    if not result["gpu_measured"] or result["pairs"] != 768:
        raise ValueError("three complete paired blocks required for R1 delivery")
    lines = [
        "# R1 chronological sensor replay",
        "",
        "Scientific optimizer updates: **0**.",
        "",
        "Fixed original 64 TRAIN queries, H1/H8/H16/WIDE, three paired blocks.",
        "Each route ingests all events between successive queries, including gaps.",
        "Both routes use a bounded sensor cache. Reference uses canonical per-window ROI",
        "projection; mapped uses the previously validated map-once union.",
        "",
        "| Block | Arm | Route | p50 ms | p95 ms | Ingest seconds | Fallback reads |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['block']} | {row['label']} | {row['route']} | "
            f"{row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
            f"{row['ingestion_seconds']:.2f} | {row['fallback_reads']} |"
        )
    lines += [
        "",
        "This measures sparse chronological replay, not an online system.",
        "Do not compare it directly with independent-request R0 or training speed.",
        "All preprocessed tensors are bit exact; outputs retain the original tolerances.",
        "The resident bound is 64 MiB per route plus source chunks and model tensors.",
        "Time starts at the first requested history in each recording.",
        "Model initialization is recorded separately and excluded from per-query latency.",
        "No protected holdouts were opened.",
    ]
    atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())
    files = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".csv", ".md")
        and p.name not in ("WRITER.lock", "BUNDLE_VERIFICATION.json", "CHAIN_STATE.json")
        and "cpu_parity" not in p.parts
    ]
    sources = list((ROOT / "operational/efficient_context").glob("r1*.py"))
    sources += [ROOT / "tests/unit/test_r1_reader.py"]
    entries = {str(p.relative_to(output)).replace("\\", "/"): p for p in files}
    entries.update({"source/" + str(p.relative_to(ROOT)).replace("\\", "/"): p for p in sources})
    manifest = {name: digest(path) for name, path in entries.items()}
    destination = output / "R1_ESSENTIAL.zip"
    temporary = output / "R1_ESSENTIAL.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr("SHA256_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"R1 bundle member mismatch: {name}")
    temporary.replace(destination)
    sha = digest(destination)
    atomic_bytes(output / "R1_ESSENTIAL.zip.sha256", f"{sha}  R1_ESSENTIAL.zip\n".encode())
    atomic_json(
        output / "BUNDLE_VERIFICATION.json",
        dict(status="PASSED", sha256=sha, members=len(manifest), checked_utc=stamp()),
    )


def run(output: Path) -> None:
    """Resume finite phases, preserving every completed pair on transient failure."""
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "CHAIN_FREEZE.json"
    files = list((ROOT / "operational/efficient_context").glob("r1*.py"))
    freeze = dict(
        sources={str(p.relative_to(ROOT)): digest(p) for p in files},
        optimizer_updates=0,
        queries=64,
        arms=["H1", "H8", "H16", "WIDE"],
        blocks=3,
        capacity_mib=64,
        deadline_utc=DEADLINE.isoformat(),
    )
    if freeze_path.exists() and read(freeze_path) != freeze:
        raise ValueError("R1 chain source freeze changed")
    if not freeze_path.exists():
        atomic_json(freeze_path, freeze)
    phases = [
        ("window_parity", "operational.efficient_context.r1_parity", output / "window_parity"),
        ("measurement", "operational.efficient_context.r1_measure", output),
    ]
    with Lease(output):
        for phase, module, destination in phases:
            result = output / phase / "RESULT.json"
            if result.exists():
                continue
            attempts = 0
            while not result.exists():
                if datetime.now(UTC) >= DEADLINE:
                    atomic_json(
                        output / "CHAIN_STATE.json",
                        dict(status="DEADLINE_PRESERVED", phase=phase, checked_utc=stamp()),
                    )
                    return
                while (
                    (output / "STOP_REQUEST").exists()
                    or (phase == "measurement" and not (output / "GPU_SLOT_GRANTED.json").exists())
                    or (phase == "measurement" and external_heavy_pids())
                ):
                    atomic_json(
                        output / "CHAIN_STATE.json",
                        dict(
                            status="WAITING",
                            phase=phase,
                            checked_utc=stamp(),
                            reason="stop marker or GPU slot not released",
                            optimizer_updates=0,
                        ),
                    )
                    if datetime.now(UTC) >= DEADLINE:
                        atomic_json(
                            output / "CHAIN_STATE.json",
                            dict(status="DEADLINE_PRESERVED", phase=phase, checked_utc=stamp()),
                        )
                        return
                    time.sleep(30)
                command = [sys.executable, "-m", module, "--output", str(destination)]
                log_path = output / (phase + "_chain.log")
                with log_path.open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    while process.poll() is None:
                        if datetime.now(UTC) >= DEADLINE:
                            atomic_bytes(output / "STOP_REQUEST", b"R1 campaign deadline\n")
                        if phase == "measurement" and external_heavy_pids():
                            atomic_bytes(output / "STOP_REQUEST", b"R1 external heavy process\n")
                        atomic_json(
                            output / "CHAIN_STATE.json",
                            dict(
                                status="RUNNING",
                                phase=phase,
                                child_pid=process.pid,
                                checked_utc=stamp(),
                                command=command,
                                attempt=attempts + 1,
                                optimizer_updates=0,
                            ),
                        )
                        time.sleep(30)
                if process.returncode == 0 and result.exists():
                    break
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                transient = any(
                    word in tail
                    for word in ("InterruptedError", "OSError", "WinError", "PermissionError")
                )
                attempts += 1
                atomic_json(
                    output / "CHAIN_STATE.json",
                    dict(
                        status="RETRY_WAIT" if transient and attempts < 4 else "FAILED_PRESERVED",
                        phase=phase,
                        returncode=process.returncode,
                        checked_utc=stamp(),
                        attempt=attempts,
                        log=str(log_path),
                        optimizer_updates=0,
                    ),
                )
                if not transient or attempts >= 4:
                    raise RuntimeError("R1 phase failed; see preserved log and chain state")
                time.sleep(60)
        bundle(output)
        atomic_json(
            output / "CHAIN_STATE.json",
            dict(status="COMPLETE", checked_utc=stamp(), optimizer_updates=0),
        )


def main() -> int:
    """Start or safely resume the finite engineering queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/efficient_context_20261004/r1_20261008"
    )
    run(parser.parse_args().output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

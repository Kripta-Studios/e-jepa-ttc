"""Local atomic evidence, identity, ownership and resource guards."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from operational.rgb_port.accounting import atomic_write_bytes, read_bytes_shared

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "docs/range_transfer_handoff_20261010"
PARENTS = {"E_A5_MATCHED": 30330, "E_C2F_MATCHED": 30330, "PAIR_E_MATCHED": 6840}
SCREEN = [(m, loss, 7) for loss in range(2) for m in range(3)]


def read(path: Path) -> dict[str, Any]:
    """Read JSON without modifying the producer's files."""
    return json.loads(read_bytes_shared(path).decode("utf-8-sig"))


def sha(path: Path) -> str:
    """Stream a file hash only at admission/durable boundaries."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    """Hash canonical JSON."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path: Path, value: object) -> None:
    """Publish complete JSON with fsync and atomic replacement."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_bytes(path, payload.encode("utf-8"))


def freeze(path: Path, value: object) -> None:
    """Accept reentry only for an identical immutable document."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen identity changed: {path}")
    else:
        write(path, value)


def utc() -> str:
    """Return an explicit UTC timestamp."""
    return datetime.now(UTC).isoformat()


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write diagnostic rows without importing any models or training code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def git(repo: Path, *args: str) -> str:
    """Read local Git identity without network operations."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def code_binding() -> dict[str, str]:
    """Bind the runner and native scientific/checkpoint implementations."""
    paths = list((ROOT / "operational/range_transfer").glob("*.py"))
    paths += list((ROOT / "src/e_jepa_ttc/range_transfer").glob("*.py"))
    paths += [
        ROOT / x
        for x in (
            "src/e_jepa_ttc/simplex_t/model.py",
            "src/e_jepa_ttc/simplex_t/phase.py",
            "operational/rgb_port/train_heads.py",
            "operational/rgb_port/train_producers.py",
            "operational/rgb_port/accounting.py",
            "operational/rgb_port/evaluate.py",
            "scripts/range_transfer_v12_trace.py",
            "scripts/Resume-RangeTransfer.ps1",
        )
    ]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(paths)}


def windows_commit_headroom() -> int | None:
    """Read the native Windows commit counter without importing training code."""
    if os.name != "nt":
        return None

    class Performance(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ],
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    counter = Performance()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(counter), counter.cb):
        raise OSError("Windows commit counters unavailable")
    return int((counter.CommitLimit - counter.CommitTotal) * counter.PageSize)


def resources(run: Path, *, starting: bool = False) -> dict[str, Any]:
    """Observe shared resources without changing any process or global setting."""
    project, processes = 0, []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            command = process.info["cmdline"] or []
            text = " ".join(command).lower()
            if "python" in (process.info["name"] or "").lower() and any(
                x in text for x in ("e-jepa-ttc", "e_jepa_ttc", "operational.")
            ):
                rss = process.memory_info().rss
                project += rss
                processes.append({**process.info, "rss_bytes": rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    own = psutil.Process()
    # Include the waiting V14 orchestrator when this is its historical V12 worker.
    parent = own.parent()
    while parent is not None:
        try:
            command = " ".join(parent.cmdline()).lower()
            if "python" not in parent.name().lower() or not any(
                marker in command
                for marker in ("operational.range_transfer.run", "range_transfer_v12_trace.py")
            ):
                break
            own = parent
            parent = own.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    own_rss = own.memory_info().rss
    for child in own.children(recursive=True):
        try:
            own_rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    free = psutil.virtual_memory().available
    disk = shutil.disk_usage(run if run.exists() else run.parent).free
    commit = windows_commit_headroom()
    reserve = 1024**3  # two checkpoints per head plus predictions/results
    minimum = (3 if starting else 2) * 1024**3
    allowed = (
        free >= minimum
        and project <= 23_000_000_000
        and own_rss <= 4 * 1024**3
        and disk - reserve >= 10_000_000_000
        and (commit is None or commit >= minimum)
    )
    return {
        "utc": utc(),
        "allowed": allowed,
        "available_bytes": free,
        "commit_headroom_bytes": commit,
        "project_rss_bytes": project,
        "v14_tree_rss_bytes": own_rss,
        "disk_free_bytes": disk,
        "disk_reservation_bytes": reserve,
        "processes": processes,
        "numeric_threads": 1,
        "gpu_used_by_v14": False,
    }


@contextmanager
def ownership(run: Path) -> Iterator[None]:
    """Hold an OS file lock for this V14 run only; crashes release it."""
    run.mkdir(parents=True, exist_ok=True)
    path = run / "V14.lock"
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def fit_id(mass: int, loss: int, seed: int) -> str:
    """Canonical, closed experiment identity."""
    if mass not in (0, 1, 2) or loss not in (0, 1) or seed not in (7, 13, 23):
        raise ValueError("Recipe outside the closed protocol")
    return f"RT-M{mass}-L{loss}-S{seed}"

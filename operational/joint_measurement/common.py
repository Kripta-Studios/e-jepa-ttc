"""V18-only roots, admission, budgets and resource observations."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch

from operational.range_transfer.common import (
    digest,
    freeze,
    read,
    sha,
    utc,
    windows_commit_headroom,
    write,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/joint_measurement_20261012"
CONFIG = ROOT / "configs/joint_measurement/primary.json"


def config() -> dict:
    """Read the prospective campaign config."""
    return read(CONFIG)


def owner() -> Path:
    """Resolve the read-only parent campaign from the declared relative root."""
    return (ROOT / config()["owner_relative"]).resolve()


def resource_state(*, starting: bool = False, disk_reserve: int = 0) -> dict:
    """Observe aggregated project memory and commit, without altering other jobs."""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            command = " ".join(proc.info["cmdline"] or [])
            if "python" in (proc.info["name"] or "").lower() and any(
                x in command.lower() for x in ("e-jepa-ttc", "e_jepa_ttc", "operational.")
            ):
                processes.append({**proc.info, "rss": proc.memory_info().rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    minimum = (3 if starting else 2) * 1024**3
    available = psutil.virtual_memory().available
    commit = windows_commit_headroom()
    cache = Path(config()["cache_root"])
    disk = shutil.disk_usage(cache if cache.exists() else cache.anchor).free
    rss = sum(p["rss"] for p in processes)
    allowed = (
        available >= minimum
        and (commit is None or commit >= minimum)
        and rss <= 23_000_000_000
        and disk - disk_reserve >= 10_000_000_000
        and shutil.disk_usage(ROOT).free >= 11_000_000_000
    )
    return {
        "utc": utc(),
        "allowed": allowed,
        "available_bytes": available,
        "commit_headroom_bytes": commit,
        "project_rss_bytes": rss,
        "cache_disk_free_bytes": disk,
        "disk_reserve": disk_reserve,
        "processes": processes,
    }


def require_resources(*, starting: bool = False, disk_reserve: int = 0) -> None:
    """Fail durably before consuming resources below the guard."""
    state = resource_state(starting=starting, disk_reserve=disk_reserve)
    write(RUN / "RESOURCE_LATEST.json", state)
    if not state["allowed"]:
        raise RuntimeError("RESOURCE_WAIT: see RESOURCE_LATEST.json")


def accounting() -> dict:
    """Count all own durable, technical, pending and recovery updates."""
    result = {"scientific": 0, "technical": 0, "recovery_upper": 0, "pending_upper": 0}
    for path in RUN.glob("**/UPDATE_JOURNAL.json"):
        row = read(path)
        key = "technical" if "qa" in path.relative_to(RUN).parts else "scientific"
        result[key] += row["completed_updates"]
        result["recovery_upper"] += row["recovery_upper"]
        result["pending_upper"] += row["pending_update_upper"]
    result["physical_upper"] = sum(result.values())
    caps = config()["caps"]
    for name, limit in (
        ("scientific", "scientific"),
        ("technical", "technical"),
        ("recovery_upper", "recovery"),
        ("physical_upper", "physical"),
    ):
        if result[name] > caps[limit]:
            raise RuntimeError(f"V18 {name} cap exceeded")
    write(RUN / "ACCOUNTING.json", result)
    return result


@contextmanager
def gpu_charge(scope: str, reserve: float = 900) -> Iterator[float]:
    """Reserve wall time before own GPU work and retain crashed reservations."""
    with gpu_ownership():
        with _gpu_charge(scope, reserve) as deadline:
            yield deadline


@contextmanager
def gpu_ownership() -> Iterator[None]:
    """Serialize V18 writers and defer to foreign Python GPU owners."""
    import msvcrt

    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "V18_GPU.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                timeout=15,
            ).decode()
            foreign = []
            for value in raw.splitlines():
                if not value.strip().isdigit() or int(value) == os.getpid():
                    continue
                try:
                    proc = psutil.Process(int(value))
                    if "python" in proc.name().lower():
                        foreign.append({"pid": proc.pid, "command": proc.cmdline()})
                except psutil.NoSuchProcess:
                    pass
            write(RUN / "GPU_OWNERSHIP.json", {"utc": utc(), "foreign_python": foreign})
            if foreign:
                raise RuntimeError("GPU_OWNER_WAIT: foreign Python GPU owner remains alive")
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _gpu_charge(scope: str, reserve: float) -> Iterator[float]:
    """Durable accounting while the exclusive V18 GPU lock is held."""
    path = RUN / "GPU_BUDGET.json"
    book = read(path) if path.exists() else {"cap_s": 64800, "entries": []}
    remaining = book["cap_s"] - sum(r["charged_s"] for r in book["entries"])
    reserve = min(reserve, remaining)
    if reserve < 60:
        raise RuntimeError("GPU_BUDGET_EXHAUSTED")
    row = {
        "scope": scope,
        "utc": utc(),
        "pid": os.getpid(),
        "charged_s": reserve,
        "status": "RESERVED",
    }
    book["entries"].append(row)
    write(path, book)
    started = time.perf_counter()
    try:
        yield started + reserve - 30
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row.update(charged_s=elapsed, status="SETTLED", end_utc=utc())
        book["charged_total_s"] = sum(r["charged_s"] for r in book["entries"])
        write(path, book)
        if elapsed > reserve:
            raise RuntimeError("GPU reservation overrun; no further work admitted")


def admit() -> dict:
    """Bind completed P-only A5, exact roles and the historical source recipe."""
    parent = owner()
    old = read((ROOT / config()["v15_admission_relative"]).resolve())
    files = [CONFIG, ROOT / "configs/rgb_port/producers.json"]
    files += [parent / f"{role}_MANIFEST.json" for role in ("P", "H", "V")]
    files += [
        parent / "fits/E_A5_MATCHED" / name
        for name in ("checkpoint_last.pt", "CHECKPOINT_RECEIPT.json", "RUN_PROVENANCE.json")
    ]
    receipt = read(parent / "fits/E_A5_MATCHED/CHECKPOINT_RECEIPT.json")
    checkpoint_sha = sha(parent / "fits/E_A5_MATCHED/checkpoint_last.pt")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != 30330
        or checkpoint_sha != old["parents"]["E_A5_MATCHED"]
        or checkpoint_sha != receipt["checkpoint_sha256"]
    ):
        raise ValueError("A5 P-only final parent identity mismatch")
    groups = []
    for role, count in (("P", 53890), ("H", 18422), ("V", 16432)):
        manifest = read(parent / f"{role}_MANIFEST.json")
        rows = Path(manifest["rows_path"])
        if manifest["role"] != role or manifest["population_size"] != count:
            raise ValueError("Role population mismatch")
        if sha(rows) != manifest["rows_sha256"]:
            raise ValueError("Role rows changed")
        groups.append(set(manifest["group_ids"]))
        files.append(rows)
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Role group leakage")
    result = {
        "status": "ADMITTED",
        "parent_sha256": checkpoint_sha,
        "files": {str(p): sha(p) for p in files},
        "config_hash": digest(config()),
        "source_commit": "396abe922537d9dac19d6db3451e59d283636e20",
    }
    freeze(RUN / "ADMISSION.json", result)
    accounting()
    return result

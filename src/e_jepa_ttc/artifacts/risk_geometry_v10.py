"""Durable receipts, allowlisted reads, phase ordering and resource supervision."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import psutil


def digest(path: Path) -> str:
    """Hash a physical file with bounded memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish durable JSON only after the complete payload is synced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def binding(path: Path) -> dict[str, Any]:
    p = path.resolve(strict=True)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}


def verify(record: dict[str, Any]) -> Path:
    p = Path(record["path"]).resolve(strict=True)
    if p.stat().st_size != record["bytes"] or digest(p) != record["sha256"]:
        raise ValueError(f"physical identity changed: {p}")
    return p


class AllowlistedSources:
    """Read permission is exact canonical identity, never a directory substring."""

    def __init__(self, records: list[dict[str, Any]], journal: Path) -> None:
        self.records = {str(Path(r["path"]).resolve()): r for r in records}
        self.journal = journal
        self.verified: dict[str, tuple[int, int, int]] = {}

    def access(self, path: Path) -> Path:
        p = path.resolve(strict=True)
        key = str(p)
        if key not in self.records:
            raise PermissionError(f"source not explicitly allowlisted: {p}")
        stat = p.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if self.verified.get(key) != stamp:
            verify(self.records[key])
            self.verified[key] = stamp
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"path": key, "sha256": self.records[key]["sha256"], "time_ns": time.time_ns()}
                )
                + "\n"
            )
        return p


ORDER = (
    "NEW",
    "INPUTS_VERIFIED",
    "BASELINES_REPLAYED",
    "IMPLEMENTATION_TESTED",
    "SCIENTIFIC_LOCKED",
    "FITTING",
    "ENDPOINTS_FROZEN",
    "OUTER_EVALUATED",
    "DIAGNOSTICS_MATERIALIZED",
    "DECIDED",
)


class PhaseLedger:
    """Transitions bind actual physical receipts and every previous event."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events = json.loads(path.read_text()) if path.exists() else []
        previous = "0" * 64
        for event in self.events:
            payload = {k: v for k, v in event.items() if k != "sha256"}
            if event["previous"] != previous or object_digest(payload) != event["sha256"]:
                raise ValueError("corrupt phase ledger")
            for record in event["receipts"]:
                verify(record)
            previous = event["sha256"]

    @property
    def state(self) -> str:
        return self.events[-1]["state"] if self.events else "NEW"

    def require(self, state: str) -> None:
        if ORDER.index(self.state) < ORDER.index(state):
            raise ValueError(f"phase prerequisite {state} missing; current={self.state}")
        for event in self.events:
            for record in event["receipts"]:
                verify(record)

    def advance(self, state: str, paths: list[Path]) -> None:
        if ORDER.index(state) != ORDER.index(self.state) + 1 or not paths:
            raise ValueError(f"illegal transition {self.state} -> {state}")
        self.require(self.state)
        event = {
            "state": state,
            "time_ns": time.time_ns(),
            "previous": self.events[-1]["sha256"] if self.events else "0" * 64,
            "receipts": [binding(p) for p in paths],
        }
        event["sha256"] = object_digest(event)
        self.events.append(event)
        atomic_json(self.path, self.events)


class CampaignOwner:
    """One foreground writer, with process identity and five-second telemetry."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.stop = threading.Event()
        self.failure: str | None = None
        # Conservative compact-output estimate, including retained full resume states.
        self.estimated_remaining_bytes = 5 * 1024**3

    def check(self) -> None:
        if self.failure:
            raise RuntimeError(self.failure)
        if psutil.virtual_memory().available < 6 * 1024**3:
            raise RuntimeError("RESOURCE_BLOCKED: available RAM below 6 GiB")
        if psutil.Process().memory_info().rss > 24 * 1024**3:
            raise RuntimeError("RESOURCE_BLOCKED: RSS above 24 GiB")
        if shutil.disk_usage(self.root).free < max(
            10 * 1024**3, 2 * self.estimated_remaining_bytes
        ):
            raise RuntimeError("RESOURCE_BLOCKED: disk below max(10 GiB, twice remaining estimate)")

    def _monitor(self) -> None:
        while not self.stop.is_set():
            try:
                self.check()
                with (self.root / "telemetry.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "time_ns": time.time_ns(),
                                "rss": psutil.Process().memory_info().rss,
                                "available_ram": psutil.virtual_memory().available,
                                "disk_free": shutil.disk_usage(self.root).free,
                            }
                        )
                        + "\n"
                    )
            except Exception as error:
                self.failure = str(error)
            self.stop.wait(5)

    def __enter__(self) -> CampaignOwner:
        self.check()
        self.lock = self.root / "ACTIVE_OWNER.json"
        if self.lock.exists():
            old = json.loads(self.lock.read_text())
            if (
                psutil.pid_exists(old["pid"])
                and psutil.Process(old["pid"]).create_time() == old["created"]
            ):
                raise RuntimeError("campaign already owned by live process")
            self.lock.rename(self.root / f"STALE_OWNER_{time.time_ns()}.json")
        with self.lock.open("x") as stream:
            json.dump(
                {
                    "pid": os.getpid(),
                    "created": psutil.Process().create_time(),
                    "host": platform.node(),
                },
                stream,
            )
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()
        self.lock.rename(self.root / f"CLOSED_OWNER_{time.time_ns()}.json")

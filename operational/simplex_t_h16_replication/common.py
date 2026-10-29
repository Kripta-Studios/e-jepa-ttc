"""Independent H16 campaign: resource, identity and durable publication helpers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources
    from e_jepa_ttc.simplex_t.registry import FitSpec

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/simplex_t/h16_replication_20261003"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "operational/simplex_t_closure"))
from runtime import atomic_bytes, atomic_json, digest, publish_json  # noqa: E402

HISTORICAL_FREEZE = "2eb3fff9ea24145c4af16c0c8645eb65893144580e41880790e169239fa8f3a7"
CAMPAIGN = "H16_REPLICATION_20261003"
ARM = "TPR-D1-H16-C160"
RESERVATION = 1024**3
COMMIT_FLOOR = 1024**3


def record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def durable_stream(target: Path, stream: IO[bytes], expected_sha256: str) -> None:
    """Publish a verified stream atomically; a partial temporary file is recoverable."""
    if target.exists():
        if digest(target) != expected_sha256:
            raise ValueError(f"immutable delivery member changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending")
    h = hashlib.sha256()
    with pending.open("wb") as output:
        while block := stream.read(1024 * 1024):
            h.update(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if h.hexdigest() != expected_sha256:
        raise ValueError(f"delivery source changed: {target}")
    pending.replace(target)


def durable_copy(source: Path, target: Path) -> None:
    with source.open("rb") as stream:
        durable_stream(target, stream, digest(source))


def ids() -> list[dict[str, Any]]:
    return [
        dict(key=f"{CAMPAIGN}/{ARM}/fold{fold}/seed{seed}", fold=fold, seed=seed, updates=2500)
        for seed in (13, 23)
        for fold in range(3)
    ]


def memory() -> dict[str, int]:
    class Performance(ctypes.Structure):
        _fields_ = (
            [("cb", ctypes.c_ulong)]
            + [
                (k, ctypes.c_size_t)
                for k in (
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
            ]
            + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")]
        )

    counter = Performance()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(counter), counter.cb):
        raise OSError("Windows commit counters unavailable")
    process = psutil.Process()
    rss = process.memory_info().rss + sum(
        p.memory_info().rss for p in process.children(recursive=True)
    )
    return dict(
        available=psutil.virtual_memory().available,
        tree_rss=rss,
        commit_headroom=(counter.CommitLimit - counter.CommitTotal) * counter.PageSize,
        free_disk=psutil.disk_usage(str(ROOT)).free,
    )


class Resources:
    """Check the user limits each update; persist measurements at bounded intervals."""

    def __init__(self, *, training: bool = False) -> None:
        self.last = 0.0
        self.training = training
        self.peak = 0
        self.minimum_available = 2**63 - 1
        self.minimum_commit = 2**63 - 1

    def __call__(self) -> bool:
        m = memory()
        self.peak = max(self.peak, m["tree_rss"])
        self.minimum_available = min(self.minimum_available, m["available"])
        self.minimum_commit = min(self.minimum_commit, m["commit_headroom"])
        reasons = []
        window = ROOT / "artifacts/simplex_t/nocturnal_20261003/WINDOW_AUTHORIZATION.json"
        if self.training and window.exists():
            deadline = datetime.fromisoformat(record(window)["deadline_utc"])
            if datetime.now(UTC) >= deadline:
                reasons.append("NOCTURNAL_TRAINING_DEADLINE_REACHED")
            journals = [OUT / "PHYSICAL_WORK.json", window.parent / "execution/PHYSICAL_WORK.json"]
            saved = lost = 0
            for path in journals:
                if path.exists():
                    a = record(path)["accounting"]
                    saved += a["scientific_saved_updates"]
                    lost += a["scientific_uncertain_lost_upper"]
            technical = window.parent / "TECHNICAL_WORK.json"
            technical_upper = (
                record(technical).get("reserved_updates", 0) if technical.exists() else 0
            )
            if saved > 60000 or lost > 6000 or technical_upper > 200:
                reasons.append("UNION_UPDATE_BUDGET_EXCEEDED")
            if 60000 + lost + technical_upper > 66200:
                reasons.append("UNION_PHYSICAL_UPPER_CAP_EXCEEDED")
        if m["available"] < 2 * 1024**3:
            reasons.append("AVAILABLE_RAM_BELOW_2_GIB")
        if m["tree_rss"] > 4 * 1024**3:
            reasons.append("TREE_RSS_EXCEEDS_4_GIB")
        # Conservative outstanding reservation stays charged throughout execution.
        if m["free_disk"] - RESERVATION < 10_000_000_000:
            reasons.append("DISK_AFTER_1_GIB_RESERVATION_BELOW_10_GB")
        if m["commit_headroom"] < COMMIT_FLOOR:
            reasons.append("WINDOWS_COMMIT_HEADROOM_BELOW_1_GIB")
        if reasons or time.monotonic() - self.last > 5:
            atomic_json(
                OUT / "RESOURCES.json",
                dict(
                    snapshot=m,
                    reasons=reasons,
                    peak_tree_rss=self.peak,
                    minimum_available=self.minimum_available,
                    minimum_commit_headroom=self.minimum_commit,
                    reservation_bytes=RESERVATION,
                ),
            )
            self.last = time.monotonic()
        return not reasons

    def check(self) -> None:
        if not self():
            raise InterruptedError("PAUSED_RESOURCE: see RESOURCES.json")


def protocol() -> tuple[dict, str]:
    path = OUT / "PROTOCOL.json"
    p = record(path)
    pin = (OUT / "PROTOCOL.sha256").read_text().split()[0]
    if digest(path) != pin:
        raise ValueError("independent preregistered protocol changed")
    return p, pin


def validate_pins(p: dict, *, full: bool) -> None:
    for row in p["input_inventory"]:
        path = Path(row["path"])
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"preregistered input changed: {path}")
        if full and digest(path) != row["sha256"]:
            raise ValueError(f"preregistered input hash changed: {path}")


def sources(p: dict) -> CampaignSources:
    from e_jepa_ttc.simplex_t.configuration_preflight import open_acknowledged_source_configuration

    return open_acknowledged_source_configuration(
        Path(p["launch"]["local_paths"]),
        Path(p["launch"]["source_configuration"]),
        p["launch"]["source_configuration_sha256"],
    )[0]


def historical_spec(s: CampaignSources, fold: int, history: int = 16) -> FitSpec:
    from e_jepa_ttc.simplex_t.registry import FitSpec

    stage = "T3" if history == 16 else "T2"
    return next(v for v in s.graph if v == FitSpec(stage, f"TPR-D1-H{history}-C160", fold, 7))


def npz(path: Path, arrays: dict) -> None:
    import io

    import numpy as np

    if path.exists():
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != set(arrays) or any(
                not np.array_equal(a[k], v) for k, v in arrays.items()
            ):
                raise ValueError(f"conflicting immutable export: {path}")
        return
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False, float_format="%.17g").encode("utf-8"))


def inventory(paths: list[Path]) -> list[dict]:
    return [
        dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=digest(p))
        for p in sorted(set(paths))
    ]


class Lease:
    """Unique campaign writer; preserve a proven dead lease before recovery."""

    def __enter__(self) -> Lease:
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / "WRITER.lock"
        if self.path.exists():
            prior = record(self.path)
            try:
                owner = psutil.Process(prior["pid"])
                if owner.create_time() == prior["create_time"]:
                    raise RuntimeError(f"live campaign owner: {prior}")
            except psutil.NoSuchProcess:
                pass
            publish_json(OUT / ("STALE_OWNER_" + digest(self.path)[:12] + ".json"), prior)
            self.path.unlink()
        with self.path.open("x", encoding="utf-8") as f:
            json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time()), f)
            f.flush()
            os.fsync(f.fileno())
        return self

    def __exit__(self, *exc: object) -> None:
        if record(self.path)["pid"] != os.getpid():
            raise RuntimeError("campaign writer ownership changed")
        self.path.unlink()

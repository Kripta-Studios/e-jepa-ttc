"""Persisted, conservative wall deadlines shared across campaign invocations."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.time_cap_amendment import AMENDMENT_NAME, validate_time_amendment


class CampaignBudget:
    """An immutable deadline: restarts and replica changes cannot renew a cap.

    Downtime counts conservatively. A cap produces a partial endpoint, never a
    shortened scientific final checkpoint. The campaign owns the writer lock.
    """

    def __init__(self, path: Path, *, hours: float, clock: Callable[[], float] = time.time) -> None:
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("campaign hours must be finite and positive")
        self.path = path
        self.clock = clock
        self.amendment_path = path.parent.parent / AMENDMENT_NAME
        amendment = validate_time_amendment(path.parent.parent, Path(__file__).resolve().parents[3])
        self.wall_time_unlimited = amendment is not None
        self.amendment_sha256 = (
            compute_file_hash(str(self.amendment_path)) if amendment is not None else None
        )
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["hours"] != hours or value["clock_policy"] != "conservative_wall_v1":
                raise ValueError("persisted campaign budget identity mismatch")
            self.started = float(value["started_epoch"])
            self.deadline = float(value["deadline_epoch"])
            if (
                not math.isfinite(self.started)
                or not math.isfinite(self.deadline)
                or self.deadline != self.started + hours * 3600
            ):
                raise ValueError("persisted campaign deadline is inconsistent")
        else:
            self.started = clock()
            self.deadline = self.started + hours * 3600
            value = {
                "hours": hours,
                "clock_policy": "conservative_wall_v1",
                "started_epoch": self.started,
                "deadline_epoch": self.deadline,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())

    def check(self) -> None:
        """Reject clock rollback and raise at the original persisted deadline."""
        now = self.clock()
        if self.wall_time_unlimited and (
            not self.amendment_path.is_file()
            or compute_file_hash(str(self.amendment_path)) != self.amendment_sha256
        ):
            raise ValueError("time amendment changed during execution")
        if now < self.started:
            raise ValueError("campaign wall clock moved before budget creation")
        if not self.wall_time_unlimited and now >= self.deadline:
            raise TimeoutError(f"campaign resource cap reached: {self.path.name}")


def check_resource_margins(root: Path, device: str = "cpu") -> None:
    """Check the preregistered live disk, RAM and free VRAM margins."""
    memory = psutil.virtual_memory()
    if psutil.disk_usage(str(root)).free < 40 * 1024**3:
        raise TimeoutError("campaign disk margin below 40 GiB")
    if memory.available / memory.total < 0.2:
        raise TimeoutError("campaign RAM available fraction below 0.2")
    if device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        if free < 2 * 1024**3:
            raise TimeoutError("campaign free VRAM margin below 2 GiB")

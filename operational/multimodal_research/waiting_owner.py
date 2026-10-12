"""Admit frozen inference alongside a verified idle V18 supervisor, never its GPU work."""

import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import psutil
import torch

from operational.multimodal_validation.common import ROOT, read, sha, write

from .common import OUT

V18 = ROOT.parent / "e-jepa-ttc-v18-joint-measurement"
V18_RUN = V18 / "artifacts/joint_measurement_20261012"


def validate_idle(state: dict, budget: dict, now: datetime) -> None:
    age = (now - datetime.fromisoformat(state["utc"])).total_seconds()
    assert 0 <= age <= 120, "V18 supervisor state is stale"
    allowed = state["status"] == "WAIT_P_SENSOR" or (
        state["status"] == "RESOURCE_WAIT"
        and str(state.get("reason", "")).startswith("GPU_OWNER_WAIT")
    )
    assert allowed, "V18 supervisor is not waiting"
    assert all(r["status"] == "SETTLED" for r in budget["entries"]), "V18 has GPU reservations"


def active_gpu_check() -> list[int]:
    contract = read(OUT / "WAITING_OWNER_CONTRACT.json")
    for name, expected in contract["v18_sources"].items():
        assert sha(V18 / name) == expected, "V18 ownership source changed"
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    idle = []
    for value in raw.splitlines():
        if not value.strip().isdigit() or int(value) in own:
            continue
        try:
            proc = psutil.Process(int(value))
            if not proc.name().lower().startswith("python"):
                continue
            command = proc.cmdline()
            assert "-m" in command and command[command.index("-m") + 1] == (
                "operational.joint_measurement.run"
            ), "Foreign Python GPU worker is active"
            validate_idle(
                read(V18_RUN / "RUNNER_STATE.json"),
                read(V18_RUN / "GPU_BUDGET.json"),
                datetime.now(UTC),
            )
            idle.append(proc.pid)
        except psutil.NoSuchProcess:
            continue
    assert len(idle) <= 1, "Multiple foreign idle GPU supervisors"
    return idle


def wrap(original: Callable[..., None]) -> Callable[..., None]:
    def guarded(*, prospective_updates: int = 0) -> None:
        assert prospective_updates == 0, "Idle-owner admission is frozen inference only"
        try:
            original(prospective_updates=0)
        except RuntimeError as error:
            if not str(error).startswith("Live research GPU owner must finish:"):
                raise
        idle = active_gpu_check()
        free, _ = torch.cuda.mem_get_info()
        observed = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        assert min(free, int(observed.splitlines()[0]) * 1024**2) >= 6 * 1024**3
        assert torch.cuda.memory_reserved() <= 1024**3
        write(
            OUT / "WAITING_OWNER_ADMISSION_LATEST.json",
            dict(
                utc=datetime.now(UTC).isoformat(),
                own_pid=os.getpid(),
                verified_idle_foreign_pids=idle,
                source_sha256=sha(Path(__file__)),
                free_cuda_bytes=free,
                own_reserved_bytes=torch.cuda.memory_reserved(),
                optimizer_updates=0,
            ),
        )

    return guarded

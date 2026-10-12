"""Record an observed Git/environment snapshot before a new optimizer trajectory."""

from __future__ import annotations

import platform
import socket
import subprocess
from pathlib import Path

import torch

from operational.range_transfer.common import freeze, sha, utc

from .run import ROOT, RUN


def record_start(directory: Path, identity: dict, role: str) -> None:
    path = directory / "RUN_PROVENANCE.json"
    if path.exists():
        return
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    checks = {}
    for relative, expected in identity["sources"].items():
        local = ROOT / relative
        if sha(local) != expected:
            raise ValueError("Scientific source changed before first checkpoint")
        blob = subprocess.check_output(
            ["git", "show", commit + ":" + relative.replace("\\", "/")], cwd=ROOT
        )
        checks[relative] = blob.replace(b"\r\n", b"\n") == local.read_bytes().replace(
            b"\r\n", b"\n"
        )
    if not all(checks.values()):
        raise ValueError("Commit the exact scientific sources before optimizer updates")
    freeze(
        path,
        {
            "experiment_id": "V17/" + identity["fit_id"],
            "git_commit": commit,
            "config_hash": identity["identity_sha256"],
            "seed": identity["seed"],
            "cache_identity_sha256": identity["cache"],
            "role": role,
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "start_time": utc(),
            "scientific_sources_match_commit_after_CRLF_LF_normalization": checks,
            "checkpoint": str(directory / "checkpoint_last.pt"),
            "end_status_and_updates": str(directory / "CHECKPOINT_RECEIPT.json"),
            "cost_and_end_times": str(RUN / "GPU_BUDGET.json"),
        },
    )

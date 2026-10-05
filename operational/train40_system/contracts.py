"""Content contracts for the new TRAIN40 system, separate from historical roots."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch

from operational.efficient_context.common import ROOT, digest


def read(path: Path) -> dict:
    """Read a UTF-8 JSON contract without accepting executable configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources(freeze: dict) -> None:
    """Reject a changed source before opening scientific training state."""
    for item in freeze["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path) != item["sha256"]:
            raise ValueError(f"Frozen source changed: {item['path']}")


def environment() -> dict:
    """Capture the host and installed execution environment without changing it."""
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def verified_endpoint(output: Path, fit: str, expected_updates: int) -> tuple[Path, dict]:
    """Require a complete fixed endpoint and verify its full checkpoint bytes."""
    directory = output / "fits" / fit
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    checkpoint = directory / "checkpoint_last.pt"
    if (
        receipt["status"] != "COMPLETE"
        or receipt["committed_updates"] != expected_updates
        or digest(checkpoint) != receipt["sha256"]
    ):
        raise ValueError(f"Complete verified endpoint required: {fit}")
    return checkpoint, receipt

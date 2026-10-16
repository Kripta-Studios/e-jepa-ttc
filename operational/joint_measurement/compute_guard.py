"""Distinguish the known MCP control-plane service from project compute owners."""

from __future__ import annotations

import os
from pathlib import Path

import psutil

from .common import ROOT, require_resources


def is_mlflow_mcp(command: list[str]) -> bool:
    """Allow only the exact MLflow MCP server entrypoint, never MLflow training."""
    return (
        len(command) == 4
        and Path(command[1]).name.lower() == "mlflow.exe"
        and command[2:] == ["mcp", "run"]
    )


def guard_idle() -> None:
    """Reject project Python compute; GPU ownership is checked separately by its lock."""
    own = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in own or "python" not in (process.info["name"] or "").lower():
                continue
            command = process.info["cmdline"] or []
            if "operational.joint_measurement.hourly_observe" in command or is_mlflow_mcp(command):
                continue
            if Path(process.cwd()).is_relative_to(ROOT.parent):
                raise RuntimeError(f"Project compute still live: PID {process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    require_resources(starting=True)

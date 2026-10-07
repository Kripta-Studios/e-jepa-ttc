"""Identity-bound Windows process QoS inspection and reversible live trial."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path

import psutil

from operational.efficient_context.common import ROOT, read
from operational.train40_system.durable_io import atomic_json


class PowerState(ctypes.Structure):
    """Windows PROCESS_POWER_THROTTLING_STATE, version one."""

    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


def state_for(pid: int, change: dict | None = None) -> dict:
    """Read QoS, optionally setting only an explicitly supplied process state."""
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    library.OpenProcess.restype = wintypes.HANDLE
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    for name in ("GetProcessInformation", "SetProcessInformation"):
        function = getattr(library, name)
        function.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        function.restype = wintypes.BOOL
    handle = library.OpenProcess(0x1000 | (0x0200 if change else 0), False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if change is not None:
            desired = PowerState(1, change["ControlMask"], change["StateMask"])
            if not library.SetProcessInformation(
                handle, 4, ctypes.byref(desired), ctypes.sizeof(desired)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        current = PowerState(1, 0, 0)
        if not library.GetProcessInformation(
            handle, 4, ctypes.byref(current), ctypes.sizeof(current)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"ControlMask": int(current.ControlMask), "StateMask": int(current.StateMask)}
    finally:
        library.CloseHandle(handle)


def run(output: Path, action: str) -> dict:
    """Restrict mutations to the exact current H8 writer and preserve prior QoS."""
    lease = read(output / "WRITER.lock")
    process = psutil.Process(lease["pid"])
    if abs(process.create_time() - lease["create_time"]) > 0.05:
        raise ValueError("H8 writer identity changed")
    if "operational.train40_system.history_resources8_fast" not in process.cmdline():
        raise ValueError("Only the admitted H8 extractor can be adjusted")
    directory = output / "h8_bottleneck_20261007"
    baseline = directory / "QOS_BASELINE.json"
    before = state_for(process.pid)
    report = {
        "pid": process.pid,
        "create_time": process.create_time(),
        "action": action,
        "before": before,
        "priority": int(process.nice()),
        "affinity": process.cpu_affinity(),
        "checked_utc": datetime.now(UTC).isoformat(),
    }
    if action == "enable":
        if baseline.exists():
            original = read(baseline)
            if (original["pid"], original["create_time"], original["before"]) != (
                process.pid,
                process.create_time(),
                before,
            ):
                raise ValueError("Existing QoS trial must be explicitly restored first")
        else:
            atomic_json(baseline, report)
        desired = {"ControlMask": before["ControlMask"] | 1, "StateMask": before["StateMask"] & ~1}
        report["after"] = state_for(process.pid, desired)
    elif action == "restore":
        original = read(baseline)
        if (original["pid"], original["create_time"]) != (process.pid, process.create_time()):
            raise ValueError("QoS restore owner changed")
        report["after"] = state_for(process.pid, original["before"])
    else:
        report["children"] = [
            {"pid": child.pid, "state": state_for(child.pid)} for child in process.children()
        ]
    atomic_json(directory / f"QOS_{action.upper()}.json", report)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    atomic_json(directory / "qos_actions" / f"{stamp}_{action}.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "enable", "restore"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    arguments = parser.parse_args()
    print(run(arguments.output.resolve(), arguments.action))

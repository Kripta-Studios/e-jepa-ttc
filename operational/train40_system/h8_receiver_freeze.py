"""Seal the additive continuous receiver around frozen H8 transport execution."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files


def _qa(output: Path) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for name in ("H8_RECEIVER_PYTEST.txt", "H8_RECEIVER_RUFF.txt", "H8_RECEIVER_PYRIGHT.txt"):
        path = output / name
        text = path.read_text(encoding="utf-8")
        if name.endswith("PYTEST.txt"):
            passed = (
                "[100%]" in text
                and re.search(r"\b\d+ passed\b", text) is not None
                and re.search(
                    r"\b\d+ (?:failed|errors?)\b|\b(?:ERRORS|FAILURES)\b",
                    text,
                    re.IGNORECASE,
                )
                is None
            )
        elif name.endswith("RUFF.txt"):
            passed = "All checks passed" in text
        else:
            passed = "0 errors, 0 warnings" in text
        if not passed:
            raise ValueError("H8 receiver QA failed: " + name)
        receipts[name] = digest(path)
    return receipts


def run(output: Path) -> None:
    """Publish only after predecessor closure and receiver QA both verify."""

    transport_path = output / "H8_TRANSPORT_FREEZE.json"
    transport = read(transport_path)
    verify_sources(transport)
    directory = ROOT / "operational/train40_system"
    admission_path = output / "h8_transport_admission/H8_RECEIVER_LIVE_ADMISSION.json"
    admission = read(admission_path)
    if admission["status"] != "PASSED" or admission["optimizer_updates"] != 0:
        raise ValueError("Live continuous receiver admission is required")
    if admission["source_sha256"] != digest(directory / "resource_monitor_receiver.py"):
        raise ValueError("Admitted receiver source changed")
    if admission["admission_source_sha256"] != digest(directory / "h8_receiver_admission.py"):
        raise ValueError("Receiver admission source changed")
    snapshot = admission["snapshot"]
    closed = admission["closed_snapshot"]
    if (
        int(snapshot["snapshots_received"]) < 27
        or snapshot["continuous_receiver"]["error"] is not None
        or not snapshot["continuous_receiver"]["thread_alive"]
        or snapshot["latest"]["torch_imported"] is not False
        or float(admission["guard"]["monitor_age_seconds"]) > 12.0
        or closed["continuous_receiver"]["thread_alive"]
    ):
        raise ValueError("Continuous receiver lifecycle admission failed")
    receipts = _qa(output)
    seeds = [
        directory / name
        for name in (
            "resource_monitor_receiver.py",
            "h8_receiver_admission.py",
            "history_resources8_transport_receiver.py",
            "controller_transport_receiver.py",
            "hourly_supervisor_transport_receiver.py",
            "h8_receiver_freeze.py",
        )
    ]
    seeds.append(ROOT / "tests/unit/test_train40_h8_receiver.py")
    seeds.append(ROOT / "tests/unit/test_train40_resource_monitor_receiver.py")
    freeze = {
        "schema": "train40_h8_continuous_monitor_receiver_v1",
        "files": dependency_files(seeds),
        "entrypoint_sha256": digest(directory / "history_resources8_transport_receiver.py"),
        "controller_sha256": digest(directory / "controller_transport_receiver.py"),
        "supervisor_sha256": digest(directory / "hourly_supervisor_transport_receiver.py"),
        "receiver_source_sha256": digest(directory / "resource_monitor_receiver.py"),
        "receiver_live_admission_sha256": digest(admission_path),
        "transport_freeze_sha256": digest(transport_path),
        "transport_entrypoint_sha256": transport["entrypoint_sha256"],
        "transport_controller_sha256": transport["controller_sha256"],
        "receiver_poll_interval_seconds": 0.5,
        "monitor_thresholds_unchanged": True,
        "scientific_sources_unchanged": True,
        "optimizer_updates": 0,
        "new_receipts_bind_receiver_execution_freeze": True,
        "QA": receipts,
    }
    destination = output / "H8_RECEIVER_FREEZE.json"
    if destination.exists() and read(destination) != freeze:
        raise ValueError("Preserve existing H8 receiver freeze")
    atomic_json(destination, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    args = parser.parse_args()
    run(args.output.resolve())

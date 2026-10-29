"""Supervise EVENT V21, including exact automatic pre-head resource pauses."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V3_SHA256 = "0055f19a690c1fcbcf1ad357e64dcbeae0deb8314d566156d3f3b700ab4cdb14"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v4() -> Any:
    source = Path(__file__).resolve()
    v3_path = source.with_name("runpod_event_resource_watchdog_v3.py")
    if digest(v3_path) != V3_SHA256:
        raise ValueError("Frozen EVENT watchdog V3 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v3_v4", v3_path)
    if spec is None or spec.loader is None:
        raise ImportError(v3_path)
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v3()
    original_validate = module.validate_config
    original_downstream = module.downstream_terminal

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_overlay_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V4 overlay source binding differs")
        return value

    def downstream_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
        terminal, reason = original_downstream(task, managed)
        if terminal != "BLOCKED" or reason not in {
            "EVENT_PAUSE_NOT_AUTOMATIC",
            "EVENT_PIPELINE_RECEIPT_NOT_FRESH",
            "NO_FRESH_PAUSED_RESOURCE_HEAD_BOUNDARY",
        }:
            return terminal, reason
        run, rotation = Path(task["run"]), int(task["rotation"])
        started, pid = float(managed["started_epoch"]), int(managed["pid"])
        pause_path = run / f"EVENT_DOWNSTREAM_PAUSE_R{rotation}.json"
        if not module.fresh(pause_path, started):
            return terminal, reason
        pause = module.read(pause_path)
        if not (
            pause.get("schema") == "ejepa_event_downstream_pause_v1"
            and pause.get("status") == "PAUSED_RECOVERABLE"
            and pause.get("rotation") == rotation
            and pause.get("reason") == "resource reserve unavailable"
            and pause.get("manual") is False
            and pause.get("pause_classification") == "AUTOMATIC_RESOURCE_PAUSE"
            and pause.get("owner_pid") == pid
            and pause.get("owner_start_ticks") == managed.get("start_ticks")
            and isinstance(pause.get("resources"), dict)
        ):
            return terminal, reason
        for fit in ("PAIR_E", "E_CTX"):
            receipt = run / f"rotations/{rotation}/fits/{fit}/CHECKPOINT_RECEIPT.json"
            if receipt.is_file() and module.fresh(receipt, started):
                return terminal, reason
        return "RESTARTABLE", "FRESH_AUTOMATIC_PRE_HEAD_RESOURCE_PAUSE"

    module.validate_config = validate_config
    module.downstream_terminal = downstream_terminal
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v4().main())


if __name__ == "__main__":
    raise SystemExit(main())

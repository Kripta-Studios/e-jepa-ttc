"""Restrict waiter-to-queue exec adoption to exact downstream identities."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V5_SHA256 = "01ffcf5f16f6e99278f1cceed9660566547f8e446483733f1a152df8a098746b"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v6() -> Any:
    source = Path(__file__).resolve()
    v5_path = source.with_name("runpod_event_resource_watchdog_v5.py")
    if digest(v5_path) != V5_SHA256:
        raise ValueError("Frozen EVENT watchdog V5 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v5_v6", v5_path)
    if spec is None or spec.loader is None:
        raise ImportError(v5_path)
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v5()
    original_validate = module.validate_config
    original_adopt = module.Watchdog._adopt

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_exec_strict_overlay_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V6 strict exec overlay binding differs")
        return value

    def adopt(self: Any, task: dict[str, Any], current: dict[str, Any]) -> Any:
        previous = dict(current)
        process = original_adopt(self, task, current)
        if current.get("reason") != "ADOPTED_AFTER_ADMITTED_EXEC_TRANSITION":
            return process
        adopted = task.get("adopted_argv", [])
        current_argv = process.get("argv") if isinstance(process, dict) else None
        prior = [
            argv
            for argv in adopted
            if isinstance(argv, list)
            and module.command_sha256(argv) == previous.get("cmdline_sha256")
        ]
        valid = (
            task.get("fit") == module.HEAD
            and isinstance(process, dict)
            and process.get("pid") == previous.get("pid")
            and process.get("start_ticks") == previous.get("start_ticks")
            and isinstance(current_argv, list)
            and current_argv in adopted
            and all(current_argv != argv for argv in prior)
            and len(prior) == 1
            and process.get("cmdline_sha256") == module.command_sha256(current_argv)
        )
        if not valid:
            current.update(
                cmdline_sha256=previous.get("cmdline_sha256"),
                state="BLOCKED",
                reason="PROCESS_IDENTITY_MISMATCH",
            )
        return process

    module.validate_config = validate_config
    module.Watchdog._adopt = adopt
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v6().main())


if __name__ == "__main__":
    raise SystemExit(main())

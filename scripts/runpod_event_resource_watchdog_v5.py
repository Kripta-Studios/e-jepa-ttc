"""Adopt an admitted waiter-to-queue exec transition with the same process identity."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V4_SHA256 = "b6863d701524b998edafb3da92a9c79353a7711124e6a299b1e3c67cd596a855"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v5() -> Any:
    source = Path(__file__).resolve()
    v4_path = source.with_name("runpod_event_resource_watchdog_v4.py")
    if digest(v4_path) != V4_SHA256:
        raise ValueError("Frozen EVENT watchdog V4 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v4_v5", v4_path)
    if spec is None or spec.loader is None:
        raise ImportError(v4_path)
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v4()
    original_validate = module.validate_config
    original_adopt = module.Watchdog._adopt

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_exec_overlay_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V5 exec overlay source binding differs")
        return value

    def adopt(self: Any, task: dict[str, Any], current: dict[str, Any]) -> Any:
        process = original_adopt(self, task, current)
        if (
            process is not None
            and current.get("state") == "BLOCKED"
            and current.get("reason") == "PROCESS_IDENTITY_MISMATCH"
            and process.get("state") != "Z"
            and process.get("start_ticks") == current.get("start_ticks")
            and process.get("argv") in task.get("adopted_argv", [])
            and process.get("cmdline_sha256") == module.command_sha256(process["argv"])
        ):
            current.update(
                cmdline_sha256=process["cmdline_sha256"],
                state="RUNNING",
                reason="ADOPTED_AFTER_ADMITTED_EXEC_TRANSITION",
            )
        return process

    module.validate_config = validate_config
    module.Watchdog._adopt = adopt
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v5().main())


if __name__ == "__main__":
    raise SystemExit(main())

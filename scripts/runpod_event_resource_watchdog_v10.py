"""Preserve an admitted exec identity when the process exits between polls."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V9_SHA256 = "5e04f4b50e633b1d8df995307a7d168e932ceaaab956fc8a122e60458d03f53d"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v10() -> Any:
    source = Path(__file__).resolve()
    v9 = source.with_name("runpod_event_resource_watchdog_v9.py")
    if digest(v9) != V9_SHA256:
        raise ValueError("Frozen EVENT watchdog V9 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v9_v10", v9)
    if spec is None or spec.loader is None:
        raise ImportError(v9)
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module = wrapper.install_v9()
    original_validate = module.validate_config
    original_adopt = module.Watchdog._adopt

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_exec_exit_overlay_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V10 source binding differs")
        return value

    def adopt(self: Any, task: dict[str, Any], current: dict[str, Any]) -> Any:
        previous = dict(current)
        process = original_adopt(self, task, current)
        if (
            process is None
            and previous.get("reason") == "ADOPTED_AFTER_ADMITTED_EXEC_TRANSITION"
            and current.get("reason") == "PROCESS_IDENTITY_MISMATCH"
        ):
            current.clear()
            current.update(previous)
        return process

    module.validate_config = validate_config
    module.Watchdog._adopt = adopt
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v10().main())


if __name__ == "__main__":
    raise SystemExit(main())

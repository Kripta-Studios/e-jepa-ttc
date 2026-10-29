"""Use psutil process identity around the frozen durable RGB phase worker V1."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import psutil

V1_SHA256 = "cc8648fb050cd84cf880b98def48e2aac66694612cfb13b14f86caf9486d51f4"


def process_matches(pid: int, argv: list[str]) -> bool:
    """Match a live non-zombie PID to its exact argv without shell decoding."""
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and process.cmdline() == argv
        )
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, OSError) as error:
        raise RuntimeError(f"Cannot prove process identity for PID {pid}") from error


def load_v1() -> Any:
    source = Path(__file__).resolve().with_name("runpod_rgb_durable_phase_worker_v1.py")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != V1_SHA256:
            raise ValueError("Frozen durable RGB phase worker V1 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_rgb_durable_phase_worker_v1", source)
    if spec is None or spec.loader is None:
        raise ImportError(str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mutable: Any = module
    mutable.process_matches = process_matches
    return mutable


def main() -> int:
    return int(load_v1().main())


if __name__ == "__main__":
    raise SystemExit(main())

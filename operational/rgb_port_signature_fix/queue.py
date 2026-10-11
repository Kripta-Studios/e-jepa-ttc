"""Route inference through the repair; preserve producer and head runtimes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_io_recovery import queue as io_queue
from operational.rgb_port_rgb_overlap import queue as parent

from .contracts import NAME, validate


def route(command: list[str]) -> list[str]:
    """Only replace the inference entrypoint; retain all arguments and other commands."""
    result = list(command)
    if "-m" in result:
        index = result.index("-m") + 1
        if result[index] in {
            "operational.rgb_port.infer_experts",
            "operational.rgb_port_streaming.infer",
        }:
            result[index] = "operational.rgb_port_signature_fix.infer"
    return result


def main() -> int:
    """Install before inherited resolvers so final routing keeps the repaired entrypoint."""
    if "--help" in sys.argv:
        return parent.main()
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    validate(run)
    original_resolve = io_queue.frozen_queue._resolve_command
    original_load = io_queue.frozen_queue._load_config

    def resolve(command: list[Any], repository: Path) -> list[str]:
        return route(original_resolve(command, repository))

    def load(path: Path) -> dict[str, Any]:
        config = original_load(path)
        for member in (
            NAME,
            "repo-tree:operational/rgb_port_signature_fix",
            "repo:tests/test_rgb_port_signature_fix.py",
        ):
            if member not in config["package"]["members"]:
                config["package"]["members"].append(member)
        for field in ("heavy_command_markers", "project_command_markers"):
            marker = "operational.rgb_port_signature_fix.infer"
            if marker not in config["resources"][field]:
                config["resources"][field].append(marker)
        return config

    with (
        patch.object(io_queue.frozen_queue, "_resolve_command", resolve),
        patch.object(io_queue.frozen_queue, "_load_config", load),
    ):
        return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())

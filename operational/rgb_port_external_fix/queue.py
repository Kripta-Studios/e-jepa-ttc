"""Continue the admitted queue with native masks in external inference routes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_fusion_fix import queue as parent
from operational.rgb_port_io_recovery import queue as io_queue

from .contracts import NAME, validate

TARGETS = {"operational.rgb_port.transfer", "operational.rgb_port.cost_routes"}


def route(command: list[str], run: Path) -> list[str]:
    """Wrap fusion consumers only, preserving original command arguments."""
    if "-m" not in command:
        return command
    index = command.index("-m") + 1
    target = command[index]
    if target not in TARGETS:
        return command
    return [
        *command[:index],
        "operational.rgb_port_external_fix.runtime",
        str(run),
        target,
        *command[index + 1 :],
    ]


def prioritize_costs(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure costs before long CPU transfer; preserve task definitions/dependencies."""
    result = list(tasks)
    positions = {task["id"]: index for index, task in enumerate(result)}
    if {"DEV32_PREDICT", "PROFILE_ROUTES"}.issubset(positions):
        first, second = positions["DEV32_PREDICT"], positions["PROFILE_ROUTES"]
        if first < second:
            result[first], result[second] = result[second], result[first]
    return result


def main() -> int:
    """Retain every upstream admission and inference repair."""
    if "--help" in sys.argv:
        return parent.main()
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    validate(run)
    original_resolve = io_queue.frozen_queue._resolve_command
    original_load = io_queue.frozen_queue._load_config

    def resolve(command: list[Any], repository: Path) -> list[str]:
        return route(original_resolve(command, repository), run)

    def load(path: Path) -> dict[str, Any]:
        config = original_load(path)
        config["tasks"] = prioritize_costs(config["tasks"])
        for member in (
            NAME,
            "repo-tree:operational/rgb_port_external_fix",
            "repo:tests/test_rgb_port_external_fix.py",
            "fits/F_TRUE/FUSION_RUNTIME.json",
            "fits/F_ZERO/FUSION_RUNTIME.json",
        ):
            if member not in config["package"]["members"]:
                config["package"]["members"].append(member)
        for field in ("heavy_command_markers", "project_command_markers"):
            marker = "operational.rgb_port_external_fix.runtime"
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

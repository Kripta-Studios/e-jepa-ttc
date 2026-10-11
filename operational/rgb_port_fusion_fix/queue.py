"""Continue the existing queue with deterministic fusion training/evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_io_recovery import queue as io_queue
from operational.rgb_port_signature_fix import queue as parent

from .contracts import NAME, validate
from .runtime import TARGETS


def route(command: list[str], run: Path) -> list[str]:
    """Wrap fusion consumers only, preserving original command arguments."""
    if "-m" not in command:
        return command
    index = command.index("-m") + 1
    target = command[index]
    if target not in TARGETS:
        return command
    if target.endswith("train_heads") and (
        "--fit-id" not in command
        or command[command.index("--fit-id") + 1] not in {"F_TRUE", "F_ZERO"}
    ):
        return command
    return [
        *command[:index],
        "operational.rgb_port_fusion_fix.runtime",
        str(run),
        target,
        *command[index + 1 :],
    ]


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
        for member in (
            NAME,
            "repo-tree:operational/rgb_port_fusion_fix",
            "repo:tests/test_rgb_port_fusion_fix.py",
            "fits/F_TRUE/FUSION_RUNTIME.json",
            "fits/F_ZERO/FUSION_RUNTIME.json",
        ):
            if member not in config["package"]["members"]:
                config["package"]["members"].append(member)
        for field in ("heavy_command_markers", "project_command_markers"):
            marker = "operational.rgb_port_fusion_fix.runtime"
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

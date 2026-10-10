"""Resume the original bounded campaign through guarded producer processes."""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_io_recovery import queue as io_queue

from .contracts import validate
from .git_admission import validate_git


def main() -> int:
    """Retain I/O ownership, resource guards and concurrent routing unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["resume"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    validate(run)
    admitted = validate_git(run)
    original_load = io_queue.frozen_queue._load_config
    original_freeze = io_queue.frozen_queue._freeze_inputs
    original_route = io_queue.route

    def load(path: Path) -> dict[str, Any]:
        config = original_load(path)
        if config["base_commit"] != admitted["original_base"]:
            raise ValueError("Original scientific config base changed")
        config["base_commit"] = admitted["admitted_head"]
        for member in [
            "CONTINUITY_FREEZE.json",
            "CONTINUITY_GIT_ADMISSION.json",
            "repo-tree:operational/rgb_port_continuity",
            "repo-tree:operational/rgb_port_streaming",
            "repo:tests/test_rgb_port_continuity.py",
            "repo:tests/test_rgb_port_streaming.py",
            "continuity_20261010/QA_RECEIPT.json",
        ]:
            if member not in config["package"]["members"]:
                config["package"]["members"].append(member)
        return config

    def freeze(path: Path, config: dict[str, Any], root: Path) -> dict[str, Any]:
        validate(root)
        if config["base_commit"] != admitted["admitted_head"]:
            raise ValueError("Runtime Git admission differs")
        scientific = copy.deepcopy(config)
        scientific["base_commit"] = admitted["original_base"]
        return original_freeze(path, scientific, root)

    def route(command: list[str], freeze: Path) -> list[str]:
        if (
            "-m" in command
            and command[command.index("-m") + 1] == "operational.rgb_port.infer_experts"
        ):
            command = list(command)
            command[command.index("-m") + 1] = "operational.rgb_port_streaming.infer"
            return command
        return original_route(command, freeze)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(io_queue, "PRODUCER_MODULE", "operational.rgb_port_continuity.producer")
        )
        stack.enter_context(patch.object(io_queue, "route", route))
        stack.enter_context(patch.object(io_queue.frozen_queue, "_load_config", load))
        stack.enter_context(patch.object(io_queue.frozen_queue, "_freeze_inputs", freeze))
        return io_queue.main(["resume", "--run", str(run)])


if __name__ == "__main__":
    raise SystemExit(main())

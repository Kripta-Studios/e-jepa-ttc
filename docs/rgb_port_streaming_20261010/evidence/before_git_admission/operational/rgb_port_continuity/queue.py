"""Resume the original bounded campaign through guarded producer processes."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from operational.rgb_port_io_recovery import queue as io_queue

from .contracts import validate


def main() -> int:
    """Retain I/O ownership, resource guards and concurrent routing unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["resume"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    validate(run)
    original_route = io_queue.route

    def route(command: list[str], freeze: Path) -> list[str]:
        if (
            "-m" in command
            and command[command.index("-m") + 1] == "operational.rgb_port.infer_experts"
        ):
            command = list(command)
            command[command.index("-m") + 1] = "operational.rgb_port_streaming.infer"
            return command
        return original_route(command, freeze)

    with patch.object(io_queue, "PRODUCER_MODULE", "operational.rgb_port_continuity.producer"):
        with patch.object(io_queue, "route", route):
            return io_queue.main(["resume", "--run", str(run)])


if __name__ == "__main__":
    raise SystemExit(main())

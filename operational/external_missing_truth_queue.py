"""Continue the admitted V13 queue with explicit missing-GT scoring support."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_external_fix import queue as parent
from operational.rgb_port_io_recovery import queue as io_queue


def route(command: list[str]) -> list[str]:
    """Change only the Dev32 scoring command; leave inference and fits untouched."""
    if "-m" not in command:
        return command
    index = command.index("-m") + 1
    if command[index : index + 2] != ["operational.rgb_port.transfer", "score-dev32"]:
        return command
    arguments = [
        "--comparator" if arg == "--comparator-csv" else arg for arg in command[index + 2 :]
    ]
    return [*command[:index], "operational.dev32_missing_truth_score", *arguments]


def main() -> int:
    """Keep the previous source admissions and checkpoint continuity validators."""
    resolve_original = io_queue.frozen_queue._resolve_command

    def resolve(command: list[Any], repository: Path) -> list[str]:
        return route(resolve_original(command, repository))

    with patch.object(io_queue.frozen_queue, "_resolve_command", resolve):
        return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())

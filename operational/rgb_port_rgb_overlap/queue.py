"""Keep the durable supervisor while routing RGB through the overlap runtime."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_io_recovery import queue as io_queue
from operational.rgb_port_rgb_loader import queue as parent

from .contracts import NAME, ROOT, validate


def main() -> int:
    """Install the final module after the inherited supervisor patches."""
    values = sys.argv[1:]
    if "--help" in values:
        return parent.main()
    run = Path(values[values.index("--run") + 1]).resolve(strict=True)
    validate(run)
    original_main = io_queue.main
    original_load = io_queue.frozen_queue._load_config

    def load(path: Path) -> dict[str, Any]:
        config = original_load(path)
        members = [NAME, "repo-tree:operational/rgb_port_rgb_overlap"]
        members.extend(
            "repo:" + file.relative_to(ROOT).as_posix()
            for file in sorted((ROOT / "tests").glob("test_rgb_port_rgb_overlap*.py"))
        )
        for member in members:
            if member not in config["package"]["members"]:
                config["package"]["members"].append(member)
        return config

    def launch(argv: Sequence[str] | None = None) -> int:
        with patch.object(io_queue, "PRODUCER_MODULE", "operational.rgb_port_rgb_overlap.producer"):
            return original_main(argv)

    with (
        patch.object(io_queue, "main", launch),
        patch.object(io_queue.frozen_queue, "_load_config", load),
    ):
        return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())

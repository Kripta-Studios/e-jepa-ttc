"""Keep the existing queue/ownership policy and admit the bounded loader."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_continuity import queue as continuity
from operational.rgb_port_io_recovery import queue as io_queue
from operational.rgb_port_loader_v3.contracts import NAME, validate


def main() -> int:
    """Install the producer route after the continuity queue's outer patches."""
    values = sys.argv[1:]
    if "--help" in values:
        return continuity.main()
    run = Path(values[values.index("--run") + 1]).resolve(strict=True)
    validate(run)
    original_main = io_queue.main
    original_load = io_queue.frozen_queue._load_config

    def load(path: Path) -> dict[str, Any]:
        config = original_load(path)
        config["package"]["members"].extend(
            [
                NAME,
                "repo-tree:operational/rgb_port_loader_v3",
                "repo:tests/test_rgb_port_loader_v3.py",
            ]
        )
        return config

    def launch(argv: Sequence[str] | None = None) -> int:
        with patch.object(io_queue, "PRODUCER_MODULE", "operational.rgb_port_loader_v3.producer"):
            return original_main(argv)

    with (
        patch.object(io_queue, "main", launch),
        patch.object(io_queue.frozen_queue, "_load_config", load),
    ):
        return continuity.main()


if __name__ == "__main__":
    raise SystemExit(main())

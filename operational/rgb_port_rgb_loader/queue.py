"""Route RGB through the exact input overlay, retaining queue ownership and retries."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port_io_recovery import queue as io_queue
from operational.rgb_port_loader_v3 import queue as parent

from .contracts import NAME, validate


def main() -> int:
    """Install the final producer route after all inherited outer patches."""
    values = sys.argv[1:]
    if "--help" in values:
        return parent.main()
    run = Path(values[values.index("--run") + 1]).resolve(strict=True)
    validate(run)
    original_main = io_queue.main
    original_load = io_queue.frozen_queue._load_config

    def load(path: Path) -> dict[str, Any]:
        config = original_load(path)
        for member in [
            NAME,
            "repo-tree:operational/rgb_port_rgb_loader",
            "repo:tests/test_rgb_port_rgb_loader_cache.py",
            "repo:tests/test_rgb_port_rgb_loader_collate.py",
            "repo:tests/test_rgb_port_rgb_loader_runtime.py",
            "repo:tests/test_rgb_port_rgb_loader_guards.py",
            "repo:tests/test_rgb_port_rgb_loader_benchmark.py",
        ]:
            if member not in config["package"]["members"]:
                config["package"]["members"].append(member)
        return config

    def launch(argv: Sequence[str] | None = None) -> int:
        with patch.object(io_queue, "PRODUCER_MODULE", "operational.rgb_port_rgb_loader.producer"):
            return original_main(argv)

    with (
        patch.object(io_queue, "main", launch),
        patch.object(io_queue.frozen_queue, "_load_config", load),
    ):
        return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())

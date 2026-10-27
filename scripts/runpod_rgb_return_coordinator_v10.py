"""Run the global RGB coordinator with exact V10 Windows admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration.rgb_return_admission_v10 import admit_local_rgb_return
from operational.runpod_migration.rgb_return_coordinator import Coordinator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Coordinator config must be a JSON object")
    return Coordinator(
        config, repository, admit=admit_local_rgb_return
    ).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())

"""Monitor, download and locally verify one complete RGB cloud return."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration.rgb_return_coordinator import (
    load_coordinator,
    validate_coordinator_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.validate_config:
        print(
            json.dumps(
                validate_coordinator_config(args.config, repository), indent=2, sort_keys=True
            )
        )
        return 0
    return load_coordinator(args.config, repository).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())

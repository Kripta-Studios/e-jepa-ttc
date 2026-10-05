"""Resume frozen teacher math with retryable Windows progress publication."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import Lease
from operational.train40_system import teacher
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    # Only metadata publication changes; all frozen tensor operations and receipt fields remain.
    teacher.atomic_json = atomic_json
    with Lease(args.output.resolve()):
        teacher.run(args.output.resolve(), args.raw_root.resolve(), args.model_path.resolve())

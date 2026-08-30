#!/usr/bin/env python
"""Train a signed V8 temporal frontend fold on the closed train-only cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.training.scientific_recovery_v8_trainer import (  # noqa: E402
    run_v8_temporal_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fixture-smoke", action="store_true", help="one-epoch CPU fixture proof; never aggregate"
    )
    args = parser.parse_args()
    try:
        result = run_v8_temporal_training(
            config_path=args.config.resolve(),
            output_dir=args.output_dir.resolve(),
            device_name=args.device,
            resume=args.resume,
            fixture_smoke=args.fixture_smoke,
        )
    except (OSError, ValueError, RuntimeError, ArithmeticError) as error:
        parser.exit(2, f"V8 temporal training refused: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": result["status"], "run_name": result["run_name"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

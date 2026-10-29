"""Run the deterministic robustness evaluator on a synthetic fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.smokes import run_robustness_smoke  # noqa: E402


def run_smoke(*, output: Path, samples: int, seed: int) -> dict[str, object]:
    """Compatibility wrapper for the package-level robustness smoke."""

    return run_robustness_smoke(output=output, samples=samples, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/robustness_synthetic_smoke_current_v1.json"),
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    payload = run_smoke(output=args.output, samples=args.samples, seed=args.seed)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "corruptions_tested": payload["corruptions_tested"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

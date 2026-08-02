"""Exercise the deterministic ONNX and bounded streaming smoke paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.smokes import run_runtime_smoke  # noqa: E402


def run_smoke(*, output_dir: Path, seed: int) -> dict[str, object]:
    """Compatibility wrapper for the package-level runtime smoke."""

    return run_runtime_smoke(output_dir=output_dir, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/demos/runtime_smoke_current_v1"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    payload = run_smoke(output_dir=args.output_dir, seed=args.seed)
    print(
        json.dumps(
            {
                "output": (args.output_dir / "runtime_smoke_metrics.json").as_posix(),
                "status": payload["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

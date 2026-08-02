"""Run the bounded semantic-shortcut JEPA falsifier and write signed JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.evaluation.semantic_shortcuts import (  # noqa: E402
    SemanticShortcutConfig,
    run_semantic_shortcut_benchmark,
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/jepa_semantic_shortcut_benchmark_v1.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 23])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-sequences", type=int, default=48)
    parser.add_argument("--test-sequences", type=int, default=24)
    parser.add_argument(
        "--shortcut-mode",
        choices=("sequence", "frame"),
        default="sequence",
        help="Keep the planted shortcut fixed per sequence or refresh it per frame.",
    )
    args = parser.parse_args()

    config = SemanticShortcutConfig(
        epochs=args.epochs,
        train_sequences=args.train_sequences,
        test_sequences=args.test_sequences,
        shortcut_mode=args.shortcut_mode,
    )
    payload = run_semantic_shortcut_benchmark(
        config=config,
        seeds=tuple(args.seeds),
        device_name=args.device,
    )
    payload["git_commit"] = _git_commit()
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["artifact_sha256"] = _canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

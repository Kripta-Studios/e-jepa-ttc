"""Benchmark a V8 delivery factory without accessing sealed evaluation data."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.scientific_recovery_v8_delivery import (
    assert_v8_delivery_paths_safe,
    benchmark_v8_delivery,
)


def _factory(value: str) -> dict[str, Any]:
    module_name, separator, attribute = value.partition(":")
    if not separator:
        raise ValueError("--factory must have the form package.module:callable")
    result = getattr(importlib.import_module(module_name), attribute)()
    if not isinstance(result, dict):
        raise TypeError("V8 benchmark factory must return a mapping")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    assert_v8_delivery_paths_safe((args.output,))
    state = _factory(args.factory)
    required = ("model", "read_sample", "tensorize")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"V8 benchmark factory lacks keys: {missing}")
    result = benchmark_v8_delivery(
        state["model"],
        state["read_sample"],
        state["tensorize"],
        device=args.device,
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
    )
    payload = {
        "artifact_type": "scientific_recovery_v8_efficiency_v1",
        "scope": "outer_development_or_synthetic_only",
        "factory": args.factory,
        **result,
    }
    sign_artifact(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the V8 robustness matrix through an explicit outer-development factory.

The factory import path must resolve to a zero-argument callable returning a
mapping with ``model``, ``samples`` and ``representation``.  This deliberately
keeps raw-data access outside the delivery evaluator and prevents this script from
discovering a public validation or test split by convention.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.scientific_recovery_v8_delivery import (
    assert_v8_delivery_paths_safe,
    evaluate_v8_robustness,
)


def _factory(value: str) -> Any:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--factory must have the form package.module:callable")
    candidate = getattr(importlib.import_module(module_name), attribute)
    if not callable(candidate):
        raise TypeError("--factory target must be callable")
    result = candidate()
    if not isinstance(result, dict):
        raise TypeError("V8 robustness factory must return a mapping")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    assert_v8_delivery_paths_safe((args.output,))
    state = _factory(args.factory)
    required = ("model", "samples", "representation")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"V8 robustness factory lacks keys: {missing}")
    result = evaluate_v8_robustness(
        state["model"],
        state["samples"],
        state["representation"],
        device=args.device,
        seed=args.seed,
        temporal_history_provider=state.get("temporal_history_provider"),
    )
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_robustness_v1",
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

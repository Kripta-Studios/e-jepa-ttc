"""Calibrate V8 uncertainty/risk only on train or nested inner-OOF arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.scientific_recovery_v8_delivery import (
    assert_v8_delivery_paths_safe,
    evaluate_v8_calibration,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--fit-scope", choices=("train", "inner_oof"), required=True)
    parser.add_argument("--risk-threshold-s", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_v8_delivery_paths_safe((args.fit, args.evaluation, args.output))
    result = evaluate_v8_calibration(
        _load(args.fit),
        _load(args.evaluation),
        fit_scope=args.fit_scope,
        risk_threshold_s=args.risk_threshold_s,
    )
    payload = {
        "artifact_type": "scientific_recovery_v8_calibration_v1",
        "scope": "outer_development_evaluation_only",
        **result,
    }
    sign_artifact(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

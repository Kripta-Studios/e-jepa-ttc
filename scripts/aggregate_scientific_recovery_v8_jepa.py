#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Aggregate signed D0--D4 OOF artifacts and apply the phase-D causal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8 import hierarchical_sequence_bootstrap  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8_jepa_attribution import (  # noqa: E402
    JEPACausalGateConfig,
    classify_jepa_causal_gate,
)


def _write(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    sign_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return value


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _oof_contract_hashes(rows: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: str(row["token_id"]))
    return {
        "row_identity_sha256": _hash(
            [(row["token_id"], row["sequence_id"], row["track_id"]) for row in ordered]
        ),
        "target_sha256": _hash([(row["token_id"], row["target_ttc"]) for row in ordered]),
        "fold_sha256": _hash([(row["token_id"], row["outer_fold"]) for row in ordered]),
        "sample_weight_sha256": _hash([(row["token_id"], row["sample_weight"]) for row in ordered]),
    }


def _load(root: Path, arm: str) -> dict[float, pd.DataFrame]:
    values: dict[float, list[dict[str, Any]]] = {}
    paths = sorted(root.glob(f"{arm.lower()}/fold*/seed7/oof_predictions.json"))
    if len(paths) != 3:
        raise ValueError(f"{arm} requires exactly three signed outer-fold OOF artifacts")
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not verify_artifact_hash(value)
            or value.get("arm") != arm
        ):
            raise ValueError(f"invalid signed {arm} OOF artifact: {path}")
        fractions = value.get("fractions")
        declared_hashes = value.get("oof_contract_hashes")
        if not isinstance(fractions, dict) or not isinstance(declared_hashes, dict):
            raise ValueError(f"{arm} OOF artifact lacks fractions")
        for label, rows in fractions.items():
            if not isinstance(rows, list):
                raise ValueError(f"{arm} fraction {label} is invalid")
            expected = _oof_contract_hashes(rows)
            if declared_hashes.get(label) != expected:
                raise ValueError(
                    f"{arm} fraction {label} lacks exact row/target/fold/weight hashes"
                )
            values.setdefault(float(label), []).extend(rows)
    return {fraction: pd.DataFrame(rows) for fraction, rows in values.items()}


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    key = ["token_id", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    for frame in (left, right):
        if frame.duplicated(key).any() or frame.empty:
            raise ValueError("JEPA aggregate has duplicate or empty OOF identities")
    joined = left.merge(right, on=key, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("JEPA aggregate OOF identities are not exactly aligned")
    return joined


def _mid(frame: pd.DataFrame, column: str) -> float:
    return float(
        sequence_macro_signed_metrics(
            frame["target_ttc"].to_numpy(),
            frame[column].to_numpy(),
            frame["sequence_id"].astype(str),
        )["sequence_macro_paper_MiD_overall"]
    )


def _candidate(
    root: Path, arm: str, controls: dict[str, dict[float, pd.DataFrame]]
) -> dict[str, Any]:
    candidate = _load(root, arm)
    required = (0.01, 0.05, 0.10, 0.25, 1.0)
    if set(candidate) != set(required) or any(
        set(data) != set(required) for data in controls.values()
    ):
        raise ValueError("D0--D4 must all report the five frozen low-label fractions")
    low = required[:-1]
    mid_values = {fraction: _mid(candidate[fraction], "prediction_ttc") for fraction in required}
    baseline_mid = {
        fraction: _mid(controls["D0"][fraction], "prediction_ttc") for fraction in required
    }
    d1_mid = {fraction: _mid(controls["D1"][fraction], "prediction_ttc") for fraction in required}
    d4_mid = {fraction: _mid(controls["D4"][fraction], "prediction_ttc") for fraction in required}
    joined = _aligned(candidate[0.25], controls["D0"][0.25])
    joined = joined.rename(
        columns={
            "prediction_ttc_candidate": "prediction_ttc",
            "prediction_ttc_reference": "scratch_prediction",
        }
    )
    bootstrap = hierarchical_sequence_bootstrap(
        joined,
        candidate_prediction_column="prediction_ttc",
        reference_prediction_column="scratch_prediction",
        resamples=500,
        seed=20260814,
    )
    all_predictions = pd.concat(list(candidate.values()), ignore_index=True)
    finite_mask = all_predictions["finite"].astype(bool).to_numpy() & np.isfinite(
        all_predictions["prediction_ttc"].to_numpy(dtype=np.float64)
    )
    gate = classify_jepa_causal_gate(
        {
            "low_label_auc_mid": float(
                np.trapezoid([mid_values[value] for value in low], x=np.asarray(low))
            ),
            "scratch_low_label_auc_mid": float(
                np.trapezoid([baseline_mid[value] for value in low], x=np.asarray(low))
            ),
            "random_frozen_low_label_auc_mid": float(
                np.trapezoid([d1_mid[value] for value in low], x=np.asarray(low))
            ),
            "shuffled_future_low_label_auc_mid": float(
                np.trapezoid([d4_mid[value] for value in low], x=np.asarray(low))
            ),
            "paired_ci95_high_vs_scratch": float(
                bootstrap["delta_candidate_minus_reference"]["upper_95"]
            ),
            "full_label_delta_mid_vs_scratch": mid_values[1.0] - baseline_mid[1.0],
            "all_finite": bool(finite_mask.all()) and all(np.isfinite(list(mid_values.values()))),
            "failure_rate": float(1.0 - finite_mask.mean()),
        },
        config=JEPACausalGateConfig(),
    )
    return {
        "arm": arm,
        "mid_by_fraction": mid_values,
        "bootstrap": bootstrap,
        "gate_decision": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/jepa"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/jepa/aggregate.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "arms": ["D0", "D1", "D2", "D3", "D4"],
                    "bootstrap": "sequence_then_track",
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        controls = {arm: _load(args.results_root, arm) for arm in ("D0", "D1", "D4")}
        d2, d3 = (_candidate(args.results_root, arm, controls) for arm in ("D2", "D3"))
        positive = [item for item in (d2, d3) if item["gate_decision"]["causally_positive"]]
        result = _write(
            args.output,
            {
                "artifact_type": "scientific_recovery_v8_jepa_aggregate_v1",
                "status": "completed",
                "arms": {"D2": d2, "D3": d3},
                "jepa_causally_positive": bool(positive),
                "best_positive_arm": positive[0]["arm"] if len(positive) == 1 else None,
                "interpretation": "no causal JEPA claim permitted"
                if not positive
                else "JEPA multiseed controls required",
            },
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"V8 JEPA aggregation failed closed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {"status": "completed", "artifact_sha256": result["artifact_sha256"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

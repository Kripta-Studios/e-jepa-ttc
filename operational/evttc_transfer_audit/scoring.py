"""Signed diagnostic metrics with explicit populations and visible missingness."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any

from .common import canonical_sha


def _metrics(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    errors = [abs(p - t) for p, t in pairs]
    relative = [100 * abs(p - t) / abs(t) for p, t in pairs if t != 0]
    return {
        "scored": len(pairs),
        "relative_scored": len(relative),
        "mae_seconds": mean(errors) if errors else None,
        "relative_absolute_error_pct": mean(relative) if relative else None,
    }


def score_rows(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Score a fixed target population; missing model outputs retain their rows."""
    scope = policy.get("scope", "diagnostic_partial")
    if scope != "diagnostic_partial":
        raise ValueError(
            "Only diagnostic_partial is implemented; official equivalence is unverified"
        )
    target_ids = [r["query_id"] for r in targets]
    prediction_ids = [r["query_id"] for r in predictions]
    if len(set(target_ids)) != len(target_ids) or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("Duplicate population IDs")
    if not set(prediction_ids).issubset(target_ids):
        raise ValueError("Predictions outside the declared population")
    by_id = {r["query_id"]: r["prediction"] for r in predictions}
    groups: dict[str, list[tuple[float, float]]] = {}
    missing: Counter[str] = Counter()
    finite = compatible = 0
    ledger = []
    for target in targets:
        prediction = by_id.get(target["query_id"])
        valid_prediction = prediction is not None and math.isfinite(float(prediction))
        valid_target = target["target_status"] == "valid"
        finite += int(valid_prediction)
        compatible += int(valid_target)
        if valid_target and (
            target["target_ttc"] is None or not math.isfinite(target["target_ttc"])
        ):
            raise ValueError("Valid target must be finite")
        if not valid_target:
            missing[target["reason"]] += 1
        if not valid_prediction:
            missing["missing_prediction" if prediction is None else "nonfinite_prediction"] += 1
        group = target.get("group", target["sequence_id"])
        groups.setdefault(group, [])
        if valid_prediction and valid_target:
            assert prediction is not None
            groups[group].append((float(prediction), float(target["target_ttc"])))
        ledger.append(
            dict(
                target,
                prediction=float(prediction)
                if valid_prediction and prediction is not None
                else None,
                prediction_status="finite" if valid_prediction else "missing",
                scored=bool(valid_prediction and valid_target),
            )
        )
    per_group = {key: _metrics(pairs) for key, pairs in sorted(groups.items())}
    micro = _metrics([pair for pairs in groups.values() for pair in pairs])
    macro = {}
    for metric in ("mae_seconds", "relative_absolute_error_pct"):
        values = [row[metric] for row in per_group.values() if row[metric] is not None]
        macro[metric] = mean(values) if values else None
    return {
        "schema": "evttc_transfer_score_v1",
        "status": "diagnostic_partial",
        "scope": scope,
        "protocol_sha256": canonical_sha(policy),
        "population_sha256": canonical_sha(targets),
        "prediction_sha256": canonical_sha(
            [
                {
                    "query_id": r["query_id"],
                    "prediction": r["prediction"],
                    "prediction_status": r["prediction_status"],
                }
                for r in ledger
            ]
        ),
        "target_contract_sha256": sorted({r["target_contract_sha256"] for r in targets}),
        "counts": {
            "requested": len(targets),
            "finite_prediction": finite,
            "compatible_target": compatible,
            "scored": micro["scored"],
            "relative_scored": micro["relative_scored"],
        },
        "missing_reasons": dict(missing),
        "metrics": {"per_group": per_group, "macro": macro, "micro": micro},
        "official_equivalence": "unverified",
        "rows": ledger,
        "limitations": [
            "Finite shared subsets may be easier; missing outputs are not imputed.",
            "Zero targets contribute to MAE, never relative error.",
        ],
    }

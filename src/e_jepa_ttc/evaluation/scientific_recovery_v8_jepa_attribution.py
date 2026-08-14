"""Closed analysis primitives for Scientific Recovery V8 phase D.

The module keeps the causal claim deliberately narrow: a JEPA arm is positive
only when its preregistered low-label curve beats scratch, a random frozen
encoder, and an equal-compute shuffled-future control on the same OOF rows.
It does not select a downstream candidate and it never reads sealed splits.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class JEPACausalGateConfig:
    """Frozen phase-D thresholds from ``CODEX_HANDOFF.md`` section 14."""

    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 1.0)
    max_full_label_regression_mid: float = 3.0
    required_paired_ci95_high: float = 0.0
    required_failure_rate: float = 0.0

    def manifest(self) -> dict[str, Any]:
        return {"contract": "scientific_recovery_v8_jepa_causal_gate_v1", **asdict(self)}


def nested_low_label_tokens(
    rows: Sequence[Mapping[str, object]], *, fractions: Sequence[float], seed: int
) -> dict[float, tuple[str, ...]]:
    """Create deterministic nested subsets, retaining every sequence/track stratum.

    Each non-empty ``(sequence_id, track_id)`` stratum contributes at least one
    token to every positive fraction.  This guards against a tiny label slice
    silently dropping a complete track while preserving strict nesting.
    """

    normalized = tuple(sorted({float(value) for value in fractions}))
    if (
        not normalized
        or normalized[-1] != 1.0
        or any(not 0.0 < value <= 1.0 for value in normalized)
    ):
        raise ValueError("fractions must be positive, unique and include 1.0")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        token = str(row.get("sample_token", ""))
        sequence = str(row.get("sequence_id", ""))
        track = str(row.get("track_id", ""))
        if not token or not sequence or not track:
            raise ValueError(
                "low-label rows require non-empty sample_token, sequence_id and track_id"
            )
        strata[(sequence, track)].append(token)
    if not strata:
        raise ValueError("low-label selection requires at least one row")
    orders: dict[tuple[str, str], list[str]] = {}
    for stratum, tokens in strata.items():
        orders[stratum] = sorted(
            tokens,
            key=lambda token: _canonical_sha256(
                {"seed": int(seed), "stratum": stratum, "token": token}
            ),
        )
    result: dict[float, tuple[str, ...]] = {}
    for fraction in normalized:
        selected: list[str] = []
        for tokens in orders.values():
            count = len(tokens) if fraction == 1.0 else max(1, math.ceil(len(tokens) * fraction))
            selected.extend(tokens[:count])
        result[fraction] = tuple(sorted(selected))
    return result


_EQUAL_COMPUTE_KEYS: tuple[str, ...] = (
    "seed",
    "total_updates",
    "batch_schedule_sha256",
    "model_initialization_sha256",
    "trainer_config_sha256",
    "compute_manifest_sha256",
)


def validate_equal_compute(d2: Mapping[str, object], d4: Mapping[str, object]) -> dict[str, Any]:
    """Fail closed unless D2/D4 differ only by the future-pairing control."""

    for name, value in (("D2", d2), ("D4", d4)):
        if value.get("shuffled_future") is not (name == "D4"):
            raise ValueError(f"{name} has an invalid shuffled_future declaration")
    checked: dict[str, object] = {}
    for key in _EQUAL_COMPUTE_KEYS:
        if key not in d2 or key not in d4:
            raise ValueError(f"equal-compute manifests lack {key}")
        if d2[key] != d4[key]:
            raise ValueError(f"D2/D4 equal-compute mismatch for {key}")
        checked[key] = d2[key]
    return {
        "contract": "scientific_recovery_v8_d2_d4_equal_compute_v1",
        "passed": True,
        "only_difference": "future_pairing",
        "checked": checked,
        "checked_sha256": _canonical_sha256(checked),
    }


def classify_jepa_causal_gate(
    values: Mapping[str, object], *, config: JEPACausalGateConfig | None = None
) -> dict[str, Any]:
    """Apply the preregistered phase-D claim rule without inventing a result."""

    rules = config or JEPACausalGateConfig()
    required = (
        "low_label_auc_mid",
        "scratch_low_label_auc_mid",
        "random_frozen_low_label_auc_mid",
        "shuffled_future_low_label_auc_mid",
        "paired_ci95_high_vs_scratch",
        "full_label_delta_mid_vs_scratch",
        "all_finite",
        "failure_rate",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"JEPA causal gate lacks required values: {missing}")
    numeric = {key: float(values[key]) for key in required[:6]}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("JEPA causal gate cannot classify non-finite metrics")
    candidate = numeric["low_label_auc_mid"]
    gates = {
        "better_than_scratch": candidate < numeric["scratch_low_label_auc_mid"],
        "better_than_d1": candidate < numeric["random_frozen_low_label_auc_mid"],
        "better_than_d4": candidate < numeric["shuffled_future_low_label_auc_mid"],
        "paired_ci95_below_zero_vs_scratch": numeric["paired_ci95_high_vs_scratch"]
        < rules.required_paired_ci95_high,
        "full_label_regression_within_3_mid": numeric["full_label_delta_mid_vs_scratch"]
        <= rules.max_full_label_regression_mid,
        "all_finite": values["all_finite"] is True,
        "zero_failure_rate": float(values["failure_rate"]) <= rules.required_failure_rate,
    }
    return {
        "contract": "scientific_recovery_v8_jepa_causal_gate_v1",
        "gate_config": rules.manifest(),
        "gates": gates,
        "causally_positive": all(gates.values()),
        "interpretation": (
            "JEPA causal signal supported"
            if all(gates.values())
            else "no causal JEPA claim permitted"
        ),
    }


__all__ = [
    "JEPACausalGateConfig",
    "classify_jepa_causal_gate",
    "nested_low_label_tokens",
    "validate_equal_compute",
]

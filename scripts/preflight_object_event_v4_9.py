#!/usr/bin/env python3
"""Fail-closed preflight for the v4.9 fixed event-only fusion screen."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_BASE = "488b433857090e525c447bf2974ac72639f25194"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _finite_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def assess_v42_fusion_baseline(
    payload: dict[str, Any],
    *,
    allow_marginal_negative_accuracy_only: bool,
) -> dict[str, Any]:
    """Decide whether v4.2 is usable by the fixed-fusion replication.

    Standalone v4.9 remains strict.  The v4.10 runner may opt into the same
    narrow replication exception already audited by its outer preflight: a
    v4.2 screen whose sole failed gate is negative accuracy, while every
    non-degeneracy check remains satisfied.  The original screen is never
    relabelled as passed.
    """

    screen_passed = bool(payload.get("screen_passed"))
    selection_gates_raw = payload.get("selection_gates", {})
    selection_gates = (
        {str(name): bool(value) for name, value in selection_gates_raw.items()}
        if isinstance(selection_gates_raw, dict)
        else {}
    )
    failed_gates = sorted(name for name, passed in selection_gates.items() if not passed)

    validation = payload.get("validation_metrics", {})
    validation = validation if isinstance(validation, dict) else {}
    event = validation.get("event", {})
    event = event if isinstance(event, dict) else {}
    per_sequence = validation.get("per_sequence", {})
    per_sequence = per_sequence if isinstance(per_sequence, dict) else {}
    dependence = validation.get("event_dependence", {})
    dependence = dependence if isinstance(dependence, dict) else {}

    minimum_checks = {
        "pearson": _finite_float(event.get("pearson"), default=-math.inf) >= 0.50,
        "balanced_sign": _finite_float(
            event.get("balanced_sign_accuracy"), default=-math.inf
        )
        >= 0.68,
        "negative_accuracy_above_chance": _finite_float(
            event.get("negative_accuracy"), default=-math.inf
        )
        >= 0.50,
        "expansion_mae": _finite_float(
            event.get("expansion_mae"), default=math.inf
        )
        <= 0.025,
        "saturation": _finite_float(
            event.get("ttc_saturation_rate"), default=math.inf
        )
        <= 0.10,
        "all_sequences_positive": _finite_float(
            per_sequence.get("minimum_pearson"), default=-math.inf
        )
        > 0.0,
        "zero_event_dependence": _finite_float(
            dependence.get("zero_event_pearson_drop"), default=-math.inf
        )
        >= 0.15,
        "shuffled_event_dependence": _finite_float(
            dependence.get("shuffled_event_pearson_drop"), default=-math.inf
        )
        >= 0.10,
    }
    marginal_exception = (
        allow_marginal_negative_accuracy_only
        and not screen_passed
        and failed_gates == ["negative_accuracy"]
        and all(minimum_checks.values())
    )
    accepted = screen_passed or marginal_exception
    return {
        "accepted_for_fusion": accepted,
        "original_screen_passed": screen_passed,
        "acceptance_reason": (
            "screen_passed"
            if screen_passed
            else "marginal_negative_accuracy_only"
            if marginal_exception
            else "rejected"
        ),
        "failed_gates": failed_gates,
        "minimum_checks": minimum_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v42-summary", type=Path, required=True)
    parser.add_argument("--v42-train-predictions", type=Path, required=True)
    parser.add_argument("--v42-validation-predictions", type=Path, required=True)
    parser.add_argument("--v48-summary", type=Path, required=True)
    parser.add_argument("--v48-train-predictions", type=Path, required=True)
    parser.add_argument("--v48-validation-predictions", type=Path, required=True)
    parser.add_argument(
        "--allow-marginal-v42-negative-accuracy-only",
        action="store_true",
        help=(
            "Allow only the audited v4.10 replication exception where the "
            "sole failed v4.2 gate is negative accuracy and all minimum "
            "non-degeneracy checks pass."
        ),
    )
    args = parser.parse_args()

    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_BASE, head], check=False
    ).returncode == 0
    if not ancestor:
        raise RuntimeError(f"HEAD {head} does not descend from {REQUIRED_BASE}")
    for path in (
        args.v42_summary,
        args.v42_train_predictions,
        args.v42_validation_predictions,
        args.v48_summary,
        args.v48_train_predictions,
        args.v48_validation_predictions,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    v42 = json.loads(args.v42_summary.read_text(encoding="utf-8"))
    v48 = json.loads(args.v48_summary.read_text(encoding="utf-8"))
    if v42.get("artifact_type") != "object_event_v4_2_full_event_only_screen":
        raise RuntimeError("Unexpected v4.2 summary")
    v42_assessment = assess_v42_fusion_baseline(
        v42,
        allow_marginal_negative_accuracy_only=(
            args.allow_marginal_v42_negative_accuracy_only
        ),
    )
    if not bool(v42_assessment["accepted_for_fusion"]):
        raise RuntimeError(
            "v4.2 is not usable for this fusion run: "
            f"{json.dumps(v42_assessment, sort_keys=True)}"
        )
    if v48.get("artifact_type") != "object_event_v4_8_dense_foreground_motion":
        raise RuntimeError("Unexpected v4.8 summary")
    if v48.get("mode") != "screen" or not bool(v48.get("passed")):
        raise RuntimeError("v4.8 screen did not pass")
    print(
        json.dumps(
            {
                "status": "passed",
                "head": head,
                "required_base_ancestor": REQUIRED_BASE,
                "v4_2_summary": args.v42_summary.resolve().as_posix(),
                "v4_2_assessment": v42_assessment,
                "v4_8_summary": args.v48_summary.resolve().as_posix(),
                "scientific_contract": {
                    "fixed_fusion_only": True,
                    "alpha_informed_by_prior_development_results": True,
                    "v4_9_does_not_fit_alpha": True,
                    "standalone_v4_9_remains_strict_by_default": True,
                    "marginal_v42_exception_requires_explicit_flag": True,
                    "failed_v42_screen_is_not_relabelled": True,
                    "official_eap_test_not_opened": True,
                    "evttc_not_opened": True,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

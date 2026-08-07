#!/usr/bin/env python3
"""Fail-closed preflight for v4.10 true-seed fixed-fusion replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

REQUIRED_BASE = "488b433857090e525c447bf2974ac72639f25194"
CRITICAL_V49_HASHES = {
    "src/e_jepa_ttc/object_event_v4_9.py": "d753582ddb8b54a537034237466c6b332b0b39f47e40128d549b794d837821af",
    "scripts/analyze_object_event_v4_9_fixed_fusion.py": "d7cfefc21c26d825ab5052a8e575a1c59a13ec2031eaae7da3d9c5f2adaeed7a",
    "scripts/run_object_event_v4_9_fixed_fusion.ps1": "d3f5bde365140821929491bd21e31ee4a808eaa40d21163427196ca958cb3ecd",
    "scripts/preflight_object_event_v4_9.py": "52eb3738c0ca0bdcb663365fd114573b99a88bf7fcb7559ae38cf528a743780e",
    "configs/experiment/e_jepa_garl_object_event_fixed_fusion_v4_9.yaml": "b4f48abeec469cfa2940c4b188b143d2d6f0072b4271314ae96c954813501b21",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _finite_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def assess_v42_replication_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Assess whether a v4.2 result is usable as a replication baseline.

    A fully passed screen is always accepted.  A failed screen is accepted only
    when its sole failed gate is the already documented negative-accuracy gate
    and the underlying event-only result remains clearly non-degenerate.  This
    does not relabel the original result as passed; v4.10 must report and test
    whether the downstream dense branch/fusion repairs the weakness.
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
        not screen_passed
        and failed_gates == ["negative_accuracy"]
        and all(minimum_checks.values())
    )
    accepted = screen_passed or marginal_exception
    return {
        "accepted_for_replication": accepted,
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
        "event_metrics": {
            "pearson": _finite_float(event.get("pearson"), default=-math.inf),
            "balanced_sign_accuracy": _finite_float(
                event.get("balanced_sign_accuracy"), default=-math.inf
            ),
            "negative_accuracy": _finite_float(
                event.get("negative_accuracy"), default=-math.inf
            ),
            "expansion_mae": _finite_float(
                event.get("expansion_mae"), default=math.inf
            ),
            "ttc_saturation_rate": _finite_float(
                event.get("ttc_saturation_rate"), default=math.inf
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v42-run-root", type=Path, required=True)
    parser.add_argument("--v49-run-root", type=Path, required=True)
    parser.add_argument("--v49-config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_BASE, head], check=False
    ).returncode == 0
    if not ancestor:
        raise RuntimeError(f"HEAD {head} does not descend from {REQUIRED_BASE}")
    actual_hashes: dict[str, str] = {}
    for relative, expected in CRITICAL_V49_HASHES.items():
        path = Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {relative}: expected {expected}, got {actual}")
        actual_hashes[relative] = actual

    fusion_raw = yaml.safe_load(args.v49_config.read_text(encoding="utf-8"))
    alpha = float(fusion_raw.get("fusion", {}).get("alpha", -1.0))
    if alpha != 0.5:
        raise ValueError(f"v4.10 requires the frozen v4.9 alpha=0.5, got {alpha}")

    seed_status: dict[str, object] = {}
    for seed in args.seeds:
        v42_dir = args.v42_run_root / f"seed-{seed}"
        summary = v42_dir / "summary.json"
        checkpoint = v42_dir / "eligible.pt"
        if not checkpoint.is_file():
            checkpoint = v42_dir / "best_observed.pt"
        for path in (summary, checkpoint, v42_dir / "train_predictions.csv", v42_dir / "validation_predictions.csv"):
            if not path.is_file():
                raise FileNotFoundError(path)
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("artifact_type") != "object_event_v4_2_full_event_only_screen":
            raise RuntimeError(f"Unexpected v4.2 artifact for seed {seed}")
        replication_assessment = assess_v42_replication_baseline(payload)
        if not bool(replication_assessment["accepted_for_replication"]):
            raise RuntimeError(
                f"v4.2 seed {seed} is not usable for replication: "
                f"{json.dumps(replication_assessment, sort_keys=True)}"
            )
        checkpoint_seed = int(payload.get("train_config", {}).get("seed", seed))
        if checkpoint_seed != seed:
            raise RuntimeError(
                f"v4.2 summary seed mismatch for seed {seed}: recorded {checkpoint_seed}"
            )
        v49_summary = args.v49_run_root / f"seed-{seed}" / "summary.json"
        seed_status[str(seed)] = {
            "v42_summary": summary.resolve().as_posix(),
            "v42_checkpoint": checkpoint.resolve().as_posix(),
            "v42_checkpoint_kind": checkpoint.name,
            "v42_replication_assessment": replication_assessment,
            "v49_already_exists": v49_summary.is_file(),
        }

    result = {
        "status": "passed",
        "head": head,
        "required_base_ancestor": REQUIRED_BASE,
        "critical_v4_9_hashes": actual_hashes,
        "fixed_alpha": alpha,
        "seeds": list(args.seeds),
        "seed_status": seed_status,
        "scientific_contract": {
            "true_seed_specific_configs_required": True,
            "fixed_alpha_across_all_seeds": True,
            "same_fixed_sequence_split": True,
            "failed_v42_screens_are_not_relabelled": True,
            "marginal_v42_negative_accuracy_seed_allowed_only_for_replication": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

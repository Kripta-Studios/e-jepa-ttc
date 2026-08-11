#!/usr/bin/env python
"""Pre-registered branch classifier for A5-S1, A6-S1 and A7-S1.

Decisions are data, not process exit failures: the script exits zero after a valid
classification even when the scientific branch is stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.artifacts.hashing import sign_artifact


def _read(path: Path) -> dict[str, Any]:
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict):
        raise ValueError(path)
    if x.get("official_test_opened") is not False and x.get("sealed_sources", {}).get("private_test_opened") is not False:
        raise ValueError(f"summary does not prove sealed test stayed closed: {path}")
    return x


def _pearson(node: Any, level: str) -> float:
    if not isinstance(node, dict) or not isinstance(node.get(level), dict):
        return float("nan")
    return float(node[level].get("pearson", float("nan")))


def _m(summary: dict[str, Any]) -> dict[str, float]:
    vm = summary["validation_metrics"]
    gd = vm.get("geometry_diagnostics") or {}
    return {
        "mid": float(vm["sequence_macro"]["sequence_macro_paper_MiD_overall"]),
        "failure": float(vm["signed"]["failure_rate_pct"]),
        "pearson": float(vm["log_ratio_pearson"]),
        "delta_h_global": _pearson(gd.get("delta_log_height_vs_physical"), "global"),
        "delta_h_macro": _pearson(gd.get("delta_log_height_vs_physical"), "macro_by_sequence"),
        "abs_h_macro": _pearson(gd.get("absolute_log_height"), "macro_by_sequence"),
    }


def _geometry_preserved(base: dict[str, float], cand: dict[str, float], fraction: float = 0.90) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {}
    ok = True
    for key in ("delta_h_global", "delta_h_macro", "abs_h_macro"):
        b, c = base[key], cand[key]
        threshold = fraction * b if b >= 0 else b / fraction
        passed = math.isfinite(b) and math.isfinite(c) and c >= threshold
        checks[key] = {"base": b, "candidate": c, "threshold": threshold, "pass": passed}
        ok &= passed
    return ok, checks


def _fraction_recovered(base: float, target: float, candidate: float, lower_is_better: bool) -> float:
    gain = (base - target) if lower_is_better else (target - base)
    recovered = (base - candidate) if lower_is_better else (candidate - base)
    if gain <= 0:
        return float("nan")
    return recovered / gain


def classify(stage: str, base: dict[str, Any], candidate: dict[str, Any], a5: dict[str, Any] | None) -> dict[str, Any]:
    b, c = _m(base), _m(candidate)
    geom_ok, geom_checks = _geometry_preserved(b, c)
    if stage == "a5":
        checks = {
            "mid_improvement_at_least_5pct": {"value": (b["mid"] - c["mid"]) / b["mid"], "threshold": 0.05},
            "pearson_improvement_at_least_0_05": {"value": c["pearson"] - b["pearson"], "threshold": 0.05},
            "failure_not_worse_by_more_than_1pp": {"value": c["failure"] - b["failure"], "threshold_max": 1.0},
        }
        ttc_signal = (
            checks["mid_improvement_at_least_5pct"]["value"] >= 0.05
            and checks["pearson_improvement_at_least_0_05"]["value"] >= 0.05
            and checks["failure_not_worse_by_more_than_1pp"]["value"] <= 1.0
        )
        decision = "REPLICATE_A5" if (ttc_signal and geom_ok) else "RUN_A6" if ttc_signal else "STOP_TRANSPORT_BRANCH"
        reason = "A5 TTC signal and geometry pass" if decision == "REPLICATE_A5" else "A5 TTC signal passes but geometry conflicts" if decision == "RUN_A6" else "A5-S1 does not preserve a meaningful TTC advantage"
        recovery = None
    else:
        if a5 is None:
            raise ValueError(f"{stage} classification requires --a5-summary")
        a = _m(a5)
        mid_rec = _fraction_recovered(b["mid"], a["mid"], c["mid"], True)
        pearson_rec = _fraction_recovered(b["pearson"], a["pearson"], c["pearson"], False)
        recovery = {"mid_fraction": mid_rec, "pearson_fraction": pearson_rec}
        threshold = 0.50 if stage == "a6" else 0.75
        recovery_ok = math.isfinite(mid_rec) and math.isfinite(pearson_rec) and mid_rec >= threshold and pearson_rec >= threshold
        positive_recovery = math.isfinite(mid_rec) and math.isfinite(pearson_rec) and mid_rec > 0 and pearson_rec > 0
        if stage == "a6":
            if geom_ok and recovery_ok:
                decision, reason = "REPLICATE_A6", "A6 preserves geometry and recovers >=50% of A5 TTC gains"
            elif geom_ok and positive_recovery:
                decision, reason = "RUN_A7", "A6 preserves geometry but adapter capacity recovers <50%; dual-stream is justified"
            elif not geom_ok:
                decision, reason = "STOP_AND_DEBUG_FREEZE", "A6 violated frozen-geometry non-inferiority; do not escalate capacity"
            else:
                decision, reason = "STOP_TRANSPORT_BRANCH", "A6 has no meaningful positive recovery"
        else:
            if geom_ok and recovery_ok:
                decision, reason = "REPLICATE_A7", "A7 preserves geometry and recovers >=75% of A5 TTC gains"
            else:
                decision, reason = "STOP_TRANSPORT_BRANCH", "A7 does not justify its added complexity under preregistered gate"
        checks = {"required_recovery_fraction": threshold, "positive_recovery": positive_recovery, "recovery_pass": recovery_ok}
    return {
        "artifact_type": f"scientific_recovery_{stage}_gate_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "base_metrics": b,
        "candidate_metrics": c,
        "ttc_checks": checks,
        "geometry_noninferiority": {"fraction": 0.90, "pass": geom_ok, "checks": geom_checks},
        "recovery": recovery,
        "contract": {
            "thresholds_fixed_before_A5_S1_validation": True,
            "public_validation_only": True,
            "public_validation_does_not_authorize_sota": True,
            "private_test_opened": False,
            "gate_failure_is_a_scientific_decision_not_an_infrastructure_error": True,
        },
        "sota_claim_authorized": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("a5", "a6", "a7"), required=True)
    p.add_argument("--base-summary", type=Path, required=True)
    p.add_argument("--candidate-summary", type=Path, required=True)
    p.add_argument("--a5-summary", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        result = classify(args.stage, _read(args.base_summary), _read(args.candidate_summary), _read(args.a5_summary) if args.a5_summary else None)
        sign_artifact(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"gate classification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

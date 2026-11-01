"""Measure the fixed event-expansion prototype on synthetic fixtures only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from operational.phase_event_expansion.model import (
    ExpansionResult,
    SolverConfig,
    estimate_expansion,
)
from operational.phase_event_expansion.synthetic import SyntheticFixture, make_fixture

COHORTS = {"development": (7,), "exposed": (7, 13, 21), "confirmation": (31, 47)}
KINDS = ("expansion", "receding", "translation", "stationary", "noise", "edge")


def score_fixture(fixture: SyntheticFixture, result: ExpansionResult) -> dict[str, object]:
    """Count abstention as failed motion recovery and finite output as failed stress rejection."""
    expected_motion = fixture.kind in ("expansion", "receding")
    q_error = abs(result.q - fixture.q) if result.q is not None else None
    translation_error = (
        float(np.max(np.abs(np.asarray(result.translation) - fixture.translation)))
        if result.translation is not None
        else None
    )
    if expected_motion:
        passed = (
            result.valid
            and q_error is not None
            and translation_error is not None
            and q_error <= max(0.25, 0.1 * abs(fixture.q))
            and translation_error <= 0.15
        )
    else:
        passed = not result.valid and result.ttc_seconds is None
    return {
        "passed": bool(passed),
        "expected_motion": expected_motion,
        "absolute_q_error": q_error,
        "maximum_translation_error": translation_error,
        "relative_ttc_error": abs(result.ttc_seconds * fixture.q - 1)
        if expected_motion and result.ttc_seconds is not None
        else None,
    }


def main() -> None:
    """Freeze source/config before generating independent evaluation fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="development")
    parser.add_argument("--objective", choices=("energy", "temporal_ncc"), default="energy")
    parser.add_argument("--reference", choices=("endpoint", "midpoint"), default="endpoint")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use fresh output directory; synthetic evidence is immutable")
    config = SolverConfig(objective=args.objective, reference=args.reference)
    seeds = COHORTS[args.cohort]
    sources = [
        Path(__file__),
        *Path("operational/phase_event_expansion").glob("*.py"),
        Path("tests/unit/test_phase_event_expansion.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-causal-event-expansion.md"),
    ]
    receipt = {
        "schema": "phase_event_expansion_synthetic_v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "solver": asdict(config),
        "seeds": seeds,
        "cohort": args.cohort,
        "kinds": KINDS,
        "events_per_fixture": 4096,
        "grid": "q=linspace(-10,10,9); bx/by={-2.5,0,2.5}; best3 distinct q starts",
        "bounds": {"q": [-10, 10], "translation": [-5, 5]},
        "optimizer": "Powell; maxfev240/start; xtol1e-4; ftol1e-5",
        "loss": "energy: negative fixed-canvas blurred IWE energy/raw energy; "
        "temporal_ncc: negative zero-mean early/late correlation; both+100*lost_fraction",
        "temporal_ncc_gates": "absolute correlation>=.8; absolute gain>=.02; "
        "fixed-b q+/-0.5 objective gap>=.002; all original other gates retained",
        "reference_conversion": "all search/output parameters remain endpoint-based; "
        "internal midpoint parameters=endpoint/(1+q_endpoint*offset)",
        "acceptance": "motion: valid, abs(q error)<=max(.25,.1*abs(q)), max b error<=.15; "
        "translation/stationary/noise/edge: abstain with no finite TTC",
        "scope": "Synthetic contour process only; no photometric simulation, real data or targets",
        "development_seed": 7,
        "historically_exposed_seeds": [7, 13, 21],
        "independent_evaluation_seeds": [31, 47],
        "protocol": "Recipe fixed before scoring; no threshold or solver tuning after outcomes",
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
            "processor": platform.processor(),
        },
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    args.output.mkdir(parents=True)
    (args.output / "CONFIG.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for path in sources:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    rows = []
    for seed in seeds:
        for kind in KINDS:
            fixture = make_fixture(seed, kind)
            result = estimate_expansion(fixture.events, fixture.roi, fixture.query_us, config)
            score = score_fixture(fixture, result)
            row = {
                "seed": seed,
                "kind": kind,
                "split": "evaluation" if args.cohort == "confirmation" else "development",
                "truth": {"q": fixture.q, "translation": fixture.translation.tolist()},
                "result": asdict(result),
                "assessment": score,
                "events_sha256": hashlib.sha256(fixture.events.tobytes()).hexdigest(),
            }
            rows.append(row)
            # Each completed fixture is durable even if a later computation fails.
            (args.output / f"{seed}_{kind}.json").write_text(
                json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                f"seed={seed} {kind}: pass={score['passed']} valid={result.valid} "
                f"q_true={fixture.q:.4f} q_est={result.q} reason={result.reason} "
                f"seconds={result.elapsed_seconds:.4f}",
                flush=True,
            )
    timings = np.array([row["result"]["elapsed_seconds"] for row in rows])
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    summary = {
        "rows": rows,
        "all_synthetic_gates_pass": all(r["assessment"]["passed"] for r in rows),
        "evaluation_passed": sum(r["assessment"]["passed"] for r in evaluation),
        "evaluation_total": len(evaluation),
        "timing_seconds": {
            "total": float(timings.sum()),
            "median": float(np.median(timings)),
            "p95": float(np.quantile(timings, 0.95)),
            "max": float(timings.max()),
        },
        "timing_scope": "full estimate_expansion including validation, ROI/sample selection, "
        "grid, optimization, diagnostic probes; excludes fixture generation",
        "real_data_authorized_by_this_result": False,
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"COMPLETE synthetic_gates_pass={summary['all_synthetic_gates_pass']}", flush=True)


if __name__ == "__main__":
    main()

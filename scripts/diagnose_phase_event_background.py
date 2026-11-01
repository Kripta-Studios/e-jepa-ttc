"""Freeze and execute eight synthetic background sensitivity cases; no real targets."""

import argparse
import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_event_expansion.background import background_fixture
from operational.phase_event_expansion.model import SolverConfig, estimate_expansion
from scripts.benchmark_phase_event_expansion_synthetic import score_fixture


def main() -> None:
    """Preserve fixed recipe, all failures, and source before seeing any outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    config = SolverConfig(objective="temporal_ncc", reference="endpoint")
    sources = [
        Path(__file__),
        Path("operational/phase_event_expansion/background.py"),
        Path("operational/phase_event_expansion/model.py"),
        Path("operational/phase_event_expansion/synthetic.py"),
        Path("scripts/benchmark_phase_event_expansion_synthetic.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-event-background-stress.md"),
    ]
    atomic_receipt(
        args.output / "CONFIG.json",
        {
            "schema": "phase_background_sensitivity_v1",
            "seeds": [61, 73],
            "fractions": [0, 0.25, 0.5, 0.75],
            "solver": asdict(config),
            "source_sha256": {str(p): file_sha(p) for p in sources},
            "scope": "Idealized synthetic contour mixtures; no sensor/label reads or tuning",
            "expected": "foreground recovery; abstention counts as failed recovery",
        },
    )
    for path in sources:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    rows = []
    for seed in (61, 73):
        for fraction in (0, 0.25, 0.5, 0.75):
            fixture, _audit_mask = background_fixture(seed, fraction)
            result = estimate_expansion(fixture.events, fixture.roi, fixture.query_us, config)
            row = {
                "seed": seed,
                "background_fraction": fraction,
                "events": len(fixture.events),
                "events_sha256": hashlib.sha256(fixture.events.tobytes()).hexdigest(),
                "truth": {"q": fixture.q, "translation": fixture.translation.tolist()},
                "result": asdict(result),
                "assessment": score_fixture(fixture, result),
            }
            atomic_receipt(args.output / "cases" / f"{seed}_{fraction}.json", row)
            rows.append(row)
            print(
                f"seed={seed} background={fraction} valid={result.valid} "
                f"q={result.q} truth={fixture.q:.5f} reason={result.reason}",
                flush=True,
            )
    atomic_receipt(
        args.output / "SUMMARY.json",
        {
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "cases": rows,
            "passed": sum(bool(row["assessment"]["passed"]) for row in rows),
            "total": len(rows),
            "promoted": False,
            "real_data_cause_established": False,
        },
    )


if __name__ == "__main__":
    main()

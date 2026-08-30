#!/usr/bin/env python
"""Aggregate only signed, frozen, train-only V8 seed-7 OOF evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from e_jepa_ttc.evaluation.scientific_recovery_v8_aggregate import (
        V8AggregateIntegrityError,
        aggregate_seed7,
    )
    from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import verify_frozen_inputs
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    p.add_argument(
        "--results-root", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/results"
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=20260814)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    try:
        frozen = verify_frozen_inputs(a.protocol, a.manifest)
        if a.dry_run:
            print(
                json.dumps(
                    {
                        "status": "validated_frozen_inputs",
                        "results_root": str(a.results_root),
                        "sealed_evaluation": "closed",
                    }
                )
            )
            return 0
        out = a.output or a.results_root / "aggregate_seed7.json"
        report = aggregate_seed7(
            protocol=frozen.protocol,
            manifest=frozen.manifest,
            results_root=a.results_root.resolve(),
            repository_root=ROOT,
            resamples=a.resamples,
            bootstrap_seed=a.bootstrap_seed,
            existing_output=out if out.is_file() else None,
        )
        previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        reused = bool(previous.get("artifact_sha256") == report.get("artifact_sha256"))
        if not reused:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "output": str(out),
                    "candidate": report["candidate_id"],
                    "multiseed_replication_candidate": report["multiseed_replication_candidate"],
                    "reused": reused,
                }
            )
        )
        return 0
    except (OSError, ValueError, V8AggregateIntegrityError) as e:
        p.exit(2, f"V8 aggregate failed closed: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())

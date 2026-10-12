"""Posthoc whole-sequence contrasts after freezing the first phase-only test12 candidate."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from operational.multimodal_validation.common import ROOT, read, sha, write

from .common import OUT


def main() -> None:
    freeze = OUT / "test12_phase_only/MODEL_INPUT_FREEZE.json"
    assert read(freeze)["candidate"] == "PHASE_ONLY_EVENT_S7"
    sys.path.insert(0, str(ROOT / "docs/multimodal_next_handoff_20261012/scripts"))
    from metrics_core import phase

    records, sources = [], {}
    for seed in (7, 13, 23):
        study = "PHASE_ONLY" if seed == 7 else "PHASE_ONLY_REPLICATION"
        for cohort, route in (("FCWD", "native_gpu"), ("DEV32", "cpu")):
            path = (
                OUT
                / f"studies/{study}/S{seed}/external_evaluation_v2"
                / route
                / cohort
                / "SCORED_PREDICTIONS.csv"
            )
            sources[str(path.relative_to(OUT))] = sha(path)
            frame = pd.read_csv(path)
            full = (
                "GARL_FULL_REGISTERED_FALLBACK_EVENT"
                if cohort == "FCWD"
                else "public_Garl_rgb_event_full"
            )
            for comparator in ("public_Garl_event_lhr", full, "F_TRUE_SAME_ROUTE"):
                for metric, band in (("mean_MiD", "all"), ("MAE", "all"), ("MiDc", "crucial")):
                    mask = frame.truth_ttc_seconds.notna()
                    if band == "crucial":
                        mask &= (frame.truth_ttc_seconds > 0) & (frame.truth_ttc_seconds <= 3)
                    sub = frame.loc[mask]
                    truth, a, b = (
                        sub[name].to_numpy()
                        for name in ("truth_ttc_seconds", "R_ANCHOR_EVENT", comparator)
                    )
                    if metric == "MAE":
                        difference = np.abs(a - truth) - np.abs(b - truth)
                    else:
                        t, aa, bb = [
                            np.asarray([phase(float(v)) for v in x]) for x in (truth, a, b)
                        ]
                        difference = 1e4 * (np.abs(aa - t) - np.abs(bb - t))
                    assert np.isfinite(difference).all()
                    groups = [
                        np.flatnonzero(sub.sequence_id.to_numpy() == sequence)
                        for sequence in sorted(sub.sequence_id.unique())
                    ]
                    rng = np.random.default_rng(20261012)
                    draws = [
                        float(
                            np.concatenate(
                                [
                                    difference[groups[i]]
                                    for i in rng.integers(0, len(groups), len(groups))
                                ]
                            ).mean()
                        )
                        for _ in range(4000)
                    ]
                    lo, hi = np.quantile(draws, [0.025, 0.975]).tolist()
                    records.append(
                        dict(
                            seed=seed,
                            cohort=cohort,
                            comparator=comparator,
                            metric=metric,
                            queries=len(sub),
                            sequences=len(groups),
                            delta=float(difference.mean()),
                            ci95_low=lo,
                            ci95_high=hi,
                        )
                    )
    destination = OUT / "PHASE_ONLY_GARL_POSTHOC.csv"
    pd.DataFrame(records).to_csv(destination, index=False)
    write(
        OUT / "PHASE_ONLY_GARL_POSTHOC.json",
        dict(
            utc=datetime.now(UTC).isoformat(),
            source_sha256=sha(__file__),
            sources=sources,
            fixed_candidate_sha256=sha(freeze),
            fixed_candidate_utc=read(freeze)["utc"],
            csv_sha256=sha(destination),
            draws=4000,
            bootstrap_unit="whole sequence",
            role="Descriptive posthoc; external data cannot replace the frozen seed 7 candidate",
            limitations=[
                "FCWD has only 3 sequences; intervals do not establish broad population coverage",
                "No correction for adaptive recipe search or multiple contrasts",
                "Event-only Garl uses fewer modalities than the candidate",
                "Head seeds share frozen producers and are not independent system replicas",
            ],
            optimizer_updates=0,
            official_SOTA=False,
        ),
    )
    print(pd.DataFrame(records).query("seed == 7").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

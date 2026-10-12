"""Score Q1 only after validating both complete prediction seals."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common import OUT, ROOT, csvwrite, read, sha, write


def sealed(kind: str) -> pd.DataFrame:
    directory = OUT / f"garl_{kind}"
    seal = read(directory / "PREDICTIONS_SEALED.json")
    assert sha(directory / "BINDING.json") == seal["binding_sha256"]
    records = []
    for name, digest in seal["query_sha256"].items():
        p = directory / "queries" / name
        assert sha(p) == digest
        value = read(p)
        assert value["binding_sha256"] == seal["binding_sha256"]
        records.append(value)
    assert len(records) == seal["population"] == 630
    return pd.DataFrame(records).set_index("query_id")


def main() -> None:
    # No label-bearing source is read before these two validations.
    full, visual = sealed("full"), sealed("visual")
    sys.path.insert(0, str(ROOT / "docs/multimodal_next_handoff_20261012/scripts"))
    from metrics_core import phase, summarize

    old = ROOT / "docs/fcwd_rgb_registration_20261012/SCORED_PREDICTIONS.csv"
    frame = pd.read_csv(old).set_index("query_id")
    assert frame.index.is_unique and set(frame.index) == set(full.index) == set(visual.index)
    for source in (full, visual):
        source = source.loc[frame.index]
        assert np.array_equal(source.anchor_us, frame.anchor_us)
        assert np.array_equal(source.sequence_id, frame.sequence_id)
    frame["GARL_FULL_REGISTERED"] = full.ttc
    frame["GARL_VISUAL_REGISTERED"] = visual.ttc
    # Availability, not prediction quality, determines fallback.
    frame["GARL_FULL_REGISTERED_FALLBACK_EVENT"] = np.where(
        full.loc[frame.index].available, frame.GARL_FULL_REGISTERED, frame.public_Garl_event_lhr
    )
    frame["garl_full_available"] = full.available
    frame["garl_visual_available"] = visual.available
    frame["garl_full_reason"] = full.reason
    frame["garl_visual_reason"] = visual.reason
    frame.reset_index().to_csv(OUT / "FCWD_MULTIMODAL_PREDICTIONS.csv", index=False)
    common = frame.garl_full_available & frame.rgb_available & frame.garl_visual_available
    metrics = []
    for scope, mask in [("all_queries", np.ones(len(frame), bool)), ("common_support", common)]:
        subset = frame.loc[mask]
        for method in [
            "E_CTX_MATCHED",
            "R_CTX",
            "F_TRUE",
            "public_Garl_event_lhr",
            "GARL_VISUAL_REGISTERED",
            "GARL_FULL_REGISTERED",
            "GARL_FULL_REGISTERED_FALLBACK_EVENT",
        ]:
            for cap in ("native", "cap60_sensitivity"):
                pred = subset[method] if cap == "native" else subset[method].clip(-60, 60)
                groups = [("ALL", np.ones(len(subset), bool))] + [
                    (s, subset.sequence_id == s) for s in sorted(subset.sequence_id.unique())
                ]
                for seq, g in groups:
                    bands = [("all", g)] + [
                        (n, g & (subset.truth_ttc_seconds > lo) & (subset.truth_ttc_seconds <= hi))
                        for n, lo, hi in [
                            ("c", 0, 3),
                            ("s", 3, 6),
                            ("l", 6, 10),
                            ("gt10", 10, np.inf),
                        ]
                    ]
                    for band, selected in bands:
                        result = summarize(
                            zip(
                                subset.loc[selected, "truth_ttc_seconds"],
                                pred[selected],
                                strict=True,
                            )
                        )
                        finite = (
                            (pred[selected] - subset.loc[selected, "truth_ttc_seconds"])
                            .abs()
                            .dropna()
                        )
                        result.update(
                            p95_AE=float(finite.quantile(0.95)) if len(finite) else None,
                            p99_AE=float(finite.quantile(0.99)) if len(finite) else None,
                        )
                        metrics.append(
                            dict(
                                scope=scope,
                                method=method,
                                precision=cap,
                                sequence=seq,
                                band=band,
                                queries=int(np.sum(selected)),
                                **result,
                            )
                        )
    csvwrite(OUT / "FCWD_METRICS.csv", metrics)
    coverage = []
    for name, source in [("full", full), ("visual", visual)]:
        for sequence in ["ALL", *sorted(source.sequence_id.unique())]:
            sub = source if sequence == "ALL" else source[source.sequence_id == sequence]
            coverage.append(
                dict(
                    model=name,
                    sequence=sequence,
                    queries=len(sub),
                    available=int(sub.available.sum()),
                    unavailable=int((~sub.available).sum()),
                )
            )
    csvwrite(OUT / "COVERAGE.csv", coverage)
    boot = []
    for scope, mask in [("common_support", common), ("all_system", np.ones(len(frame), bool))]:
        sub = frame[mask & frame.truth_ttc_seconds.notna()]
        comparator = (
            "GARL_FULL_REGISTERED"
            if scope == "common_support"
            else "GARL_FULL_REGISTERED_FALLBACK_EVENT"
        )
        for metric in ("MAE", "MiD"):
            a = sub.F_TRUE.to_numpy()
            b = np.asarray(sub[comparator])
            y = sub.truth_ttc_seconds.to_numpy()
            if metric == "MAE":
                diff = np.abs(a - y) - np.abs(b - y)
            else:
                diff = 1e4 * (
                    np.abs(np.array([phase(v) for v in a]) - np.array([phase(v) for v in y]))
                    - np.abs(np.array([phase(v) for v in b]) - np.array([phase(v) for v in y]))
                )
            assert np.isfinite(diff).all()
            groups = [
                np.flatnonzero(sub.sequence_id.to_numpy() == s)
                for s in sorted(sub.sequence_id.unique())
            ]
            rng = np.random.default_rng(20261012)
            draws = [
                float(
                    np.concatenate(
                        [diff[groups[i]] for i in rng.integers(0, len(groups), len(groups))]
                    ).mean()
                )
                for _ in range(4000)
            ]
            boot.append(
                dict(
                    scope=scope,
                    contrast="F_TRUE-" + comparator,
                    metric=metric,
                    n=len(sub),
                    groups=len(groups),
                    delta=float(diff.mean()),
                    ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                    role="posthoc_three_sequences_not_confirmation",
                )
            )
    write(OUT / "GARL_PAIRED_REVIEW.json", boot)
    write(
        OUT / "Q1_SCORE_BINDING.json",
        dict(
            labels_source_sha256=sha(old),
            full_seal_sha256=sha(OUT / "garl_full/PREDICTIONS_SEALED.json"),
            visual_seal_sha256=sha(OUT / "garl_visual/PREDICTIONS_SEALED.json"),
            prediction_csv_sha256=sha(OUT / "FCWD_MULTIMODAL_PREDICTIONS.csv"),
            optimizer_updates=0,
        ),
    )
    for item in metrics:
        if item["sequence"] == "ALL" and item["band"] == "all" and item["precision"] == "native":
            print(
                item["scope"],
                item["method"],
                item["finite_pairs"],
                item["MAE_finite_s"],
                item["strict_mean_MiD"],
                flush=True,
            )


if __name__ == "__main__":
    main()

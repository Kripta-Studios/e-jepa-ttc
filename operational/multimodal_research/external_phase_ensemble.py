"""Descriptive transfer of the fixed three-seed emitted-phase ensemble."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.multimodal_validation.common import ROOT, read, sha, write

from .common import OUT
from .phase_ensemble import DEST, PARENTS, combine, register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("FCWD", "DEV32"))
    parser.add_argument("--route", choices=("cpu", "native_gpu"), default="cpu")
    parser.set_defaults(study="PHASE_ENSEMBLE", seed="7+13+23")
    args = parser.parse_args()
    torch.set_num_threads(1)
    register()
    study = DEST
    assert (study / "V_RESULTS.json").exists()
    binding = dict(source_sha256=sha(__file__), protocol_sha256=sha(DEST / "PROTOCOL.json"))
    binding["prior_failed_schema_binding"] = sha(DEST / "EXTERNAL_ANALYSIS_BINDING.json")
    binding_path = DEST / "EXTERNAL_ANALYSIS_BINDING_V2.json"
    if binding_path.exists():
        assert read(binding_path) == binding
    else:
        write(binding_path, binding)
    frames, sources = [], {}
    for i, parent in enumerate(PARENTS):
        directory = parent / "external_evaluation_v2" / args.route / args.cohort
        if i == 0 and args.route == "cpu":
            directory = parent / "external_evaluation" / args.cohort
        path = directory / "PREDICTIONS.csv"
        assert sha(path) == read(directory / "PREDICTIONS_SEALED.json")["sha256"]
        frame = pd.read_csv(path)
        # The first evaluator explicitly labelled its parent comparator columns CPU.
        frame = frame.rename(
            columns={name + "_CPU": name + "_SAME_ROUTE" for name in ("E_CTX", "R_CTX", "F_TRUE")}
        )
        assert not frame.duplicated(["sequence_id", "anchor_us"]).any()
        frames.append(frame.set_index(["sequence_id", "anchor_us"]))
        sources[str(path.relative_to(OUT))] = sha(path)
    frame = frames[0].copy()
    aligned = [f.loc[frame.index] for f in frames]
    for other in aligned:
        assert len(other) == len(frame) and set(other.index) == set(frame.index)
        for column in ("query_id", "rgb_available", "E_CTX_SAME_ROUTE", "F_TRUE_SAME_ROUTE"):
            assert np.array_equal(frame[column], other[column]), column
    missing = ~frame.rgb_available.to_numpy()
    for mode in ("EVENT", "ZERO"):
        name = "R_ANCHOR_" + mode
        values = np.stack([f[name].to_numpy() for f in aligned])
        prediction = combine(values)
        for value in values:
            assert np.array_equal(value[missing], frame.E_CTX_SAME_ROUTE.to_numpy()[missing])
        prediction[missing] = frame.E_CTX_SAME_ROUTE.to_numpy()[missing]
        frame[name] = prediction
    frame = frame.reset_index()
    destination = study / "external_evaluation_v2" / args.route / args.cohort
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "PREDICTIONS.csv"
    frame.to_csv(path, index=False)
    write(
        destination / "PREDICTIONS_SEALED.json",
        dict(
            sha256=sha(path),
            population=len(frame),
            sources=sources,
            source_sha256=sha(__file__),
            protocol_sha256=sha(DEST / "PROTOCOL.json"),
            targets_read_for_combination=False,
            prior_external_exposure=True,
            fallback="Exact same-route E_CTX; all queries retained",
            optimizer_updates=0,
        ),
    )
    # External TTC is opened only after the complete new prediction seal.
    historical_path = (
        PREVIOUS / "FCWD_MULTIMODAL_PREDICTIONS.csv"
        if args.cohort == "FCWD"
        else ROOT / "docs/dev32_transfer_20261012/SCORED_PREDICTIONS.csv"
    )
    historical = pd.read_csv(historical_path)
    assert not historical.duplicated(["sequence_id", "anchor_us"]).any()
    merged = frame.merge(
        historical,
        on=["sequence_id", "anchor_us"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_historical"),
        indicator=True,
    )
    assert (merged._merge == "both").all() and len(merged) == len(frame)
    assert (merged.query_id == merged.query_id_historical).all()
    assert (merged.rgb_available == merged.rgb_available_historical).all()
    merged.to_csv(destination / "SCORED_PREDICTIONS.csv", index=False)
    sys.path.insert(0, str(ROOT / "docs/multimodal_next_handoff_20261012/scripts"))
    from metrics_core import phase, summarize

    common = merged.rgb_available.copy()
    comparators = [
        "R_ANCHOR_EVENT",
        "R_ANCHOR_ZERO",
        "R_CTX_SAME_ROUTE",
        "F_TRUE_SAME_ROUTE",
        "E_CTX_SAME_ROUTE",
        "R_CTX",
        "F_TRUE",
        "E_CTX_MATCHED",
        "public_Garl_event_lhr",
    ]
    if args.cohort == "FCWD":
        common &= merged.garl_full_available
        comparators += ["GARL_FULL_REGISTERED", "GARL_FULL_REGISTERED_FALLBACK_EVENT"]
    else:
        comparators += ["public_Garl_rgb_event_full"]
        common &= merged.public_Garl_rgb_event_full.notna()
    metrics = []
    for scope, mask in (("all_queries", np.ones(len(merged), bool)), ("common_support", common)):
        subset = merged.loc[mask]
        for name in comparators:
            for band, band_mask in (
                ("all", np.ones(len(subset), bool)),
                ("crucial", (subset.truth_ttc_seconds > 0) & (subset.truth_ttc_seconds <= 3)),
                ("small", (subset.truth_ttc_seconds > 3) & (subset.truth_ttc_seconds <= 6)),
                ("large", (subset.truth_ttc_seconds > 6) & (subset.truth_ttc_seconds <= 10)),
                ("gt10", subset.truth_ttc_seconds > 10),
            ):
                group = subset.loc[band_mask]
                result = summarize(zip(group.truth_ttc_seconds, group[name], strict=True))
                metrics.append(
                    dict(scope=scope, method=name, band=band, queries=len(group), **result)
                )
    pd.DataFrame(metrics).to_csv(destination / "METRICS.csv", index=False)
    contrasts = []
    for comparator, mask in (
        ("R_CTX_SAME_ROUTE", common),
        ("F_TRUE_SAME_ROUTE", np.ones(len(merged), bool)),
        ("R_ANCHOR_ZERO", np.ones(len(merged), bool)),
    ):
        sub = merged.loc[mask & merged.truth_ttc_seconds.notna()]
        truth = sub.truth_ttc_seconds.to_numpy()
        a, b = sub.R_ANCHOR_EVENT.to_numpy(), sub[comparator].to_numpy()
        for metric in ("MAE", "mean_MiD"):
            if metric == "MAE":
                difference = np.abs(a - truth) - np.abs(b - truth)
            else:
                t, aa, bb = [np.array([phase(float(v)) for v in array]) for array in (truth, a, b)]
                difference = 1e4 * (np.abs(aa - t) - np.abs(bb - t))
            assert np.isfinite(difference).all()
            groups = [
                np.flatnonzero(sub.sequence_id.to_numpy() == s)
                for s in sorted(sub.sequence_id.unique())
            ]
            rng = np.random.default_rng(20261012)
            draws = [
                float(
                    np.concatenate(
                        [difference[groups[i]] for i in rng.integers(0, len(groups), len(groups))]
                    ).mean()
                )
                for _ in range(4000)
            ]
            contrasts.append(
                dict(
                    comparator=comparator,
                    metric=metric,
                    queries=len(sub),
                    sequences=len(groups),
                    delta=float(difference.mean()),
                    ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                    role="posthoc descriptive; no external candidate selection",
                )
            )
    write(destination / "PAIRED_POSTHOC.json", contrasts)
    write(
        destination / "EVALUATION.json",
        dict(
            historical_source_sha256=sha(historical_path),
            source_sha256=sha(__file__),
            queries=len(merged),
            labelled=int(merged.truth_ttc_seconds.notna().sum()),
            rgb_available=int(merged.rgb_available.sum()),
            official_overall_MiD_available=False,
            external_selection=False,
            optimizer_updates=0,
            study=args.study,
            seed=args.seed,
            parent_route=args.route,
            validation_decision=read(study / "V_RESULTS.json")["candidate"],
        ),
    )
    print(pd.DataFrame(metrics).query("band == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

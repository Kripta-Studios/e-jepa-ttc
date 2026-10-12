"""Seal complete frozen external predictions before descriptive label joins."""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.multimodal_validation.common import ROOT, read, sha, write

from .common import OUT
from .model import RGBAnchor
from .phase_units import STUDY, PhaseUnitAnchor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("FCWD", "DEV32"))
    parser.add_argument("--study", choices=("baseline", "units"), default="baseline")
    args = parser.parse_args()
    torch.set_num_threads(1)
    study = OUT if args.study == "baseline" else STUDY
    prefix = "R_ANCHOR_" if args.study == "baseline" else "R_ANCHOR_UNITS_"
    model_type = RGBAnchor if args.study == "baseline" else PhaseUnitAnchor
    assert (study / "V_RESULTS.json").exists(), "Internal fixed-endpoint analysis must finish first"
    endpoints = read(study / "ENDPOINTS_SEALED.json")
    models = {}
    for mode in ("EVENT", "ZERO"):
        p = OUT / f"fits/{prefix}{mode}_S7/checkpoint_last.pt"
        assert sha(p) == endpoints[mode]["checkpoint_sha256"]
        assert endpoints[mode]["completed_updates"] == 2500
        model = model_type(mode).eval()
        model.load_state_dict(
            torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
        )
        models[mode] = model
    directory = OUT / "external_states" / args.cohort
    seal = read(directory / "SEALED.json")
    assert sha(directory / "BINDING.json") == seal["binding_sha256"]
    records = []
    for qid, receipt in seal["queries"].items():
        path = directory / (hashlib.sha256(qid.encode()).hexdigest() + ".npz")
        assert receipt["binding_sha256"] == seal["binding_sha256"]
        assert receipt["sha256"] == sha(path) and receipt["query_id"] == qid
        row = {
            k: receipt[k]
            for k in ("query_id", "sequence_id", "anchor_us", "rgb_available", "reason")
        }
        with np.load(path, allow_pickle=False) as z, torch.no_grad():
            row["E_CTX_CPU"] = float(phase_to_ttc(torch.from_numpy(z["event_phase"]))[0])
            row["F_TRUE_CPU"] = float(phase_to_ttc(torch.from_numpy(z["fusion_phase"]))[0])
            row["R_CTX_CPU"] = float("nan")
            if receipt["rgb_available"]:
                row["R_CTX_CPU"] = float(phase_to_ttc(torch.from_numpy(z["rgb_phase"]))[0])
                x, widths = torch.from_numpy(z["features"]), torch.from_numpy(z["rgb_width_logits"])
                for mode, model in models.items():
                    row["R_ANCHOR_" + mode] = float(
                        phase_to_ttc(model(x, widths)["point_phase"])[0]
                    )
            else:
                for mode in models:
                    row["R_ANCHOR_" + mode] = row["E_CTX_CPU"]
                assert row["F_TRUE_CPU"] == row["E_CTX_CPU"]
        records.append(row)
    assert len(records) == seal["population"]
    frame = pd.DataFrame(records)
    assert not frame.duplicated(["sequence_id", "anchor_us"]).any()
    destination = study / "external_evaluation" / args.cohort
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "PREDICTIONS.csv"
    frame.to_csv(path, index=False)
    write(
        destination / "PREDICTIONS_SEALED.json",
        dict(
            sha256=sha(path),
            population=len(frame),
            state_seal_sha256=sha(directory / "SEALED.json"),
            endpoints=endpoints,
            targets_read=False,
            source_sha256=sha(__file__),
            fallback="exact same-route E_CTX whenever RGB unavailable; keep every query",
            scope="CPU frozen recomputation; exposed cohorts; descriptive, never model selection",
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
        "R_CTX_CPU",
        "F_TRUE_CPU",
        "E_CTX_CPU",
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
        ("R_CTX_CPU", common),
        ("F_TRUE_CPU", np.ones(len(merged), bool)),
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
            validation_decision=read(study / "V_RESULTS.json")["candidate"],
        ),
    )
    print(pd.DataFrame(metrics).query("band == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

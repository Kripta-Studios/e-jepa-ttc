"""Descriptive external error attribution; never selects or changes the frozen test model."""

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.common import ROOT, read, sha, write

from .common import OUT
from .phase_only_anchor import PhaseOnlyAnchor
from .phase_only_attribution import trace


def main() -> None:
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "docs/multimodal_next_handoff_20261012/scripts"))
    from metrics_core import phase

    freeze = OUT / "test12_phase_only/MODEL_INPUT_FREEZE.json"
    assert read(freeze)["candidate"] == "PHASE_ONLY_EVENT_S7"
    models, sources = {}, {str(freeze.relative_to(OUT)): sha(freeze)}
    for arm in ("EVENT", "ZERO"):
        path = OUT / f"fits/R_ANCHOR_PHASE_ONLY_{arm}_S7/checkpoint_last.pt"
        sources[str(path.relative_to(OUT))] = sha(path)
        model = PhaseOnlyAnchor(arm).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        )
        models[arm] = model
    rows = []
    for cohort, route in (("FCWD", "native_gpu"), ("DEV32", "cpu")):
        source = OUT / f"studies/PHASE_ONLY/S7/external_evaluation_v2/{route}/{cohort}"
        path = source / "SCORED_PREDICTIONS.csv"
        sources[str(path.relative_to(OUT))] = sha(path)
        frame = pd.read_csv(path, float_precision="round_trip")
        assert not frame.query_id.duplicated().any()
        root = "external_states_gpu_native" if cohort == "FCWD" else "external_states"
        for r in frame.to_dict("records"):
            path = (
                OUT / root / cohort / (hashlib.sha256(r["query_id"].encode()).hexdigest() + ".npz")
            )
            sources[str(path.relative_to(OUT))] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                for arm, model in models.items():
                    rgb = "features" in z
                    if rgb:
                        detail = trace(
                            model,
                            torch.from_numpy(z["features"]),
                            torch.from_numpy(z["rgb_width_logits"]),
                        )
                        agreement = (
                            "same_sign"
                            if (
                                np.sign(detail["R_emitted_phase"])
                                == np.sign(detail["E_emitted_phase"])
                            )
                            else "opposite_sign"
                        )
                    else:
                        detail = dict(
                            emitted_ttc=float(phase_to_ttc(torch.from_numpy(z["event_phase"]))[0])
                        )
                        agreement = "no_RGB"
                    assert detail["emitted_ttc"] == r["R_ANCHOR_" + arm]
                    rows.append(
                        dict(
                            cohort=cohort,
                            query_id=r["query_id"],
                            sequence_id=r["sequence_id"],
                            arm=arm,
                            rgb_available=rgb,
                            parent_agreement=agreement,
                            truth_ttc=r["truth_ttc_seconds"],
                            R_CTX=r["R_CTX_SAME_ROUTE"],
                            F_TRUE=r["F_TRUE_SAME_ROUTE"],
                            E_CTX=r["E_CTX_SAME_ROUTE"],
                            **detail,
                        )
                    )
        print("STRATA_PARITY", cohort, len(frame), flush=True)
    frame = pd.DataFrame(rows)
    records = []
    for cohort in ("FCWD", "DEV32"):
        for arm in ("EVENT", "ZERO"):
            sub = pd.DataFrame(frame[(frame.cohort == cohort) & (frame.arm == arm)])
            # Selection depends on input modality/parent signs, never the GT error.
            for group in ("all", "same_sign", "opposite_sign", "no_RGB"):
                selected = pd.DataFrame(
                    sub if group == "all" else sub[sub.parent_agreement == group]
                )
                labelled = pd.DataFrame(selected[selected.truth_ttc.notna()])
                if len(labelled) == 0:
                    continue
                for comparator in ("R_CTX", "F_TRUE", "E_CTX"):
                    valid = pd.DataFrame(labelled[labelled[comparator].notna()])
                    if len(valid) == 0:
                        continue
                    truth, a, b = (
                        valid[k].to_numpy() for k in ("truth_ttc", "emitted_ttc", comparator)
                    )
                    assert (truth > 0).all(), "External cohorts here contain positive TTC labels"
                    ma, mb = (
                        np.asarray(
                            [
                                1e4 * abs(phase(float(t)) - phase(float(p)))
                                for t, p in zip(truth, pred, strict=True)
                            ]
                        )
                        for pred in (a, b)
                    )
                    assert np.isfinite(ma).all() and np.isfinite(mb).all()
                    records.append(
                        dict(
                            cohort=cohort,
                            arm=arm,
                            stratum=group,
                            comparator=comparator,
                            queries=len(selected),
                            labelled_common=len(valid),
                            sequences=int(valid.sequence_id.nunique()),
                            delta_MiD=float((ma - mb).mean()),
                            delta_MAE=float((np.abs(a - truth) - np.abs(b - truth)).mean()),
                            MiD_wins=int((ma < mb).sum()),
                            MiD_ties=int((ma == mb).sum()),
                            MAE_wins=int((np.abs(a - truth) < np.abs(b - truth)).sum()),
                            candidate_wrong_sign=int((a < 0).sum()),
                            comparator_wrong_sign=int((b < 0).sum()),
                            candidate_capped=int((np.abs(a) >= 59.999).sum()),
                            comparator_capped=int((np.abs(b) >= 59.999).sum()),
                        )
                    )
    detail_path, table_path = OUT / "PHASE_ONLY_STRATA_DETAILS.csv", OUT / "PHASE_ONLY_STRATA.csv"
    frame.to_csv(detail_path, index=False)
    table = pd.DataFrame(records)
    table.to_csv(table_path, index=False)
    wrong_sign = []
    for r in rows:
        if r["arm"] != "EVENT" or not r["rgb_available"] or not r["truth_ttc"] > 0:
            continue
        if r["R_CTX"] < 0:
            median = float(np.median(r["R_expert_phases"]))
            wrong_sign.append(
                dict(
                    **r,
                    R_expert_median_phase=median,
                    R_context_residual_phase=r["R_raw_phase"] - median,
                    R_context_crossed_expert_median_sign=median > 0,
                    candidate_keeps_wrong_sign=r["emitted_ttc"] < 0,
                )
            )
    wrong_path = OUT / "PHASE_ONLY_WRONG_SIGN_CASES.csv"
    pd.DataFrame(wrong_sign).to_csv(wrong_path, index=False)
    write(
        OUT / "PHASE_ONLY_STRATA.json",
        dict(
            utc=datetime.now(UTC).isoformat(),
            source_sha256=sha(__file__),
            sources=sources,
            trace_source_sha256=sha(
                ROOT / "operational/multimodal_research/phase_only_attribution.py"
            ),
            detail_sha256=sha(detail_path),
            table_sha256=sha(table_path),
            wrong_sign_cases_sha256=sha(wrong_path),
            wrong_sign_summary={
                c: dict(
                    RGB_wrong_sign=sum(r["cohort"] == c for r in wrong_sign),
                    positive_expert_median=sum(
                        r["cohort"] == c and r["R_context_crossed_expert_median_sign"]
                        for r in wrong_sign
                    ),
                    candidate_keeps_wrong_sign=sum(
                        r["cohort"] == c and r["candidate_keeps_wrong_sign"] for r in wrong_sign
                    ),
                )
                for c in ("FCWD", "DEV32")
            },
            queries_per_arm={
                c: int((frame.cohort.eq(c) & frame.arm.eq("EVENT")).sum())
                for c in ("FCWD", "DEV32")
            },
            all_predictions_equal_sealed=True,
            instrumented_outputs_bitwise_equal=True,
            optimizer_updates=0,
            test_labels_read=False,
            scope="Posthoc diagnostics on exposed external cohorts. No threshold fitting, "
            "candidate selection, confidence calibration or claim of independent confirmation. "
            "All external labels are positive; wrong-sign counts cannot measure "
            "approaching/receding "
            "classification in a population with negative TTC. All means are query-weighted.",
        ),
    )
    print(
        table.query("arm == 'EVENT' and comparator == 'R_CTX'").to_string(index=False), flush=True
    )


if __name__ == "__main__":
    main()

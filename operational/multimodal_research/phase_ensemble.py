"""Fixed equal-phase ensemble of all three registered phase-unit seeds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
from operational.multimodal_validation.common import RUN, read, sha, write
from operational.rgb_port.evaluate import score_signed_ttc

from .common import OUT

DEST = OUT / "studies/PHASE_ENSEMBLE"
PARENTS = (
    OUT / "studies/PHASE_UNITS",
    OUT / "studies/UNIT_REPLICATION/S13",
    OUT / "studies/UNIT_REPLICATION/S23",
)


def combine(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 2 and values.shape[0] == 3
    phases = ttc_to_phase(torch.from_numpy(values.astype(np.float64)))
    return phase_to_ttc(phases.mean(0)).numpy()


def register() -> dict:
    binding = dict(
        source_sha256=sha(__file__),
        seeds=[7, 13, 23],
        weights=[1 / 3] * 3,
        operation="mean of native emitted phases, float64, native signed conversion and 60s cap",
        arms=["EVENT", "ZERO"],
        hypothesis="Reduce head-seed variation observed on exposed internal V",
        selection="All registered unit seeds; no seed dropping, fitted weights or TTC averaging",
        external_exposure="Individual external scores known; descriptive development only",
        original_protocol_sha256=sha(OUT / "PROTOCOL.json"),
        gates="Original 2%; separate 4% large-only sensitivity; paired group bootstrap 4000",
        optimizer_updates=0,
        cost="Three readouts with shared frozen features; no full-system replicas",
        endpoints={str(p.relative_to(OUT)): sha(p / "ENDPOINTS_SEALED.json") for p in PARENTS},
    )
    path = DEST / "PROTOCOL.json"
    if path.exists():
        result = read(path)
        assert result["binding"] == binding
        return result
    assert not (DEST / "V_RESULTS.json").exists()
    result = dict(utc=datetime.now(UTC).isoformat(), binding=binding)
    write(path, result)
    return result


def internal() -> None:
    register()
    frames = []
    sources = {}
    for parent in PARENTS:
        path = parent / "V_PREDICTIONS_UNSCORED.csv"
        assert sha(path) == read(parent / "V_PREDICTIONS_SEALED.json")["sha256"]
        frame = pd.read_csv(path).set_index("sample_token")
        assert frame.index.is_unique
        frames.append(frame)
        sources[str(path.relative_to(OUT))] = sha(path)
    tokens = frames[0].index
    assert all(set(f.index) == set(tokens) for f in frames)
    predictions = {}
    diagnostics = {}
    for mode in ("EVENT", "ZERO"):
        name = "R_ANCHOR_" + mode
        values = np.stack([f.loc[tokens, name].to_numpy() for f in frames])
        predictions[name] = combine(values)
        diagnostics[mode] = dict(
            disagree_sign=int(((values.min(0) < 0) & (values.max(0) > 0)).sum()),
            cap60=int((np.abs(predictions[name]) >= 60).sum()),
        )
    path = DEST / "V_PREDICTIONS_UNSCORED.csv"
    pd.DataFrame(dict(sample_token=tokens, **predictions)).to_csv(path, index=False)
    write(
        DEST / "V_PREDICTIONS_SEALED.json",
        dict(
            sha256=sha(path),
            sources=sources,
            protocol_sha256=sha(DEST / "PROTOCOL.json"),
            targets_read_for_combination=False,
            prior_V_exposure=True,
            diagnostics=diagnostics,
        ),
    )
    truth = groups = None
    for name in ("R_CTX", "F_TRUE", "E_CTX_MATCHED"):
        frame = pd.read_csv(RUN / f"predictions/V/{name}.csv").set_index("sample_token")
        assert frame.index.is_unique and set(frame.index) == set(tokens)
        frame = frame.loc[tokens]
        if truth is None:
            truth, groups = frame.target_ttc.to_numpy(), frame.group_id.to_numpy()
        assert groups is not None
        assert np.array_equal(truth, frame.target_ttc) and np.array_equal(groups, frame.group_id)
        predictions[name] = frame.prediction_ttc.to_numpy()
    assert truth is not None and groups is not None
    scores = {name: score_signed_ttc(truth, values, groups) for name, values in predictions.items()}
    pd.DataFrame(
        dict(sample_token=tokens, target_ttc=truth, group_id=groups, **predictions)
    ).to_csv(DEST / "V_SCORED_PREDICTIONS.csv", index=False)
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        rows.append(
            dict(
                group_id=group,
                **{
                    name: score_signed_ttc(truth[mask], values[mask], groups[mask])[
                        "group_macro_bucket_MiD"
                    ]
                    for name, values in predictions.items()
                },
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "V_GROUP_CONTRASTS.csv", index=False)
    for name, score in scores.items():
        assert np.isclose(table[name].mean(), score["group_macro_bucket_MiD"], atol=1e-10)
    delta = (table.R_ANCHOR_EVENT - table.R_ANCHOR_ZERO).to_numpy()
    ci = np.quantile(
        np.random.default_rng(20261012).choice(delta, (4000, len(delta))).mean(1), [0.025, 0.975]
    ).tolist()
    gates = {}
    for name in ("R_ANCHOR_EVENT", "R_ANCHOR_ZERO"):
        new, rgb, fusion = scores[name], scores["R_CTX"], scores["F_TRUE"]
        checks = dict(
            aggregate_vs_RGB=new["group_macro_bucket_MiD"] <= 0.97 * rgb["group_macro_bucket_MiD"],
            aggregate_vs_fusion=new["group_macro_bucket_MiD"] <= fusion["group_macro_bucket_MiD"],
            mae_vs_RGB=new["diagnostics"]["mae_s"] <= rgb["diagnostics"]["mae_s"],
            mae_vs_fusion=new["diagnostics"]["mae_s"] <= fusion["diagnostics"]["mae_s"],
            sign=new["diagnostics"]["sign_error_rate"] <= rgb["diagnostics"]["sign_error_rate"],
            no_failures=new["num_failed_predictions"] == 0,
        )
        for band in ("crucial", "small", "large", "negative"):
            checks[band] = (
                new["bins"][band]["group_macro_mid"] <= 1.02 * rgb["bins"][band]["group_macro_mid"]
            )
        relaxed = dict(checks)
        relaxed["large"] = (
            new["bins"]["large"]["group_macro_mid"]
            <= 1.04 * rgb["bins"]["large"]["group_macro_mid"]
        )
        gates[name] = dict(
            checks=checks, eligible=all(checks.values()), secondary_4pct=all(relaxed.values())
        )
    result = dict(
        study="PHASE_ENSEMBLE",
        scores=scores,
        gates=gates,
        event_minus_zero=dict(delta=float(delta.mean()), ci95=ci, draws=4000),
        complementary_event_gate=ci[1] < 0,
        candidate="R_ANCHOR_EVENT" if gates["R_ANCHOR_EVENT"]["eligible"] and ci[1] < 0 else None,
        optimizer_updates=0,
        official_SOTA=False,
        scope="Exposed development; not independent confirmation",
    )
    write(DEST / "V_RESULTS.json", result)
    print(
        {k: (v["group_macro_bucket_MiD"], v["diagnostics"]["mae_s"]) for k, v in scores.items()},
        flush=True,
    )
    print(gates, ci, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "internal"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    register() if args.action == "register" else internal()


if __name__ == "__main__":
    main()

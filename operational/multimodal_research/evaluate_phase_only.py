"""Seal both fixed endpoints before opening the exposed V targets for scoring."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.common import RUN, read, sha, write
from operational.rgb_port.evaluate import score_signed_ttc

from .common import OUT
from .fusion_readout import setup
from .phase_only_anchor import PhaseOnlyAnchor
from .phase_only_readout import STUDIES, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=tuple(STUDIES))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    assert args.seed in STUDIES[args.study]
    setup()
    contract(args.study)
    destination = OUT / "studies" / args.study / f"S{args.seed}"
    prefix = f"R_ANCHOR_{args.study}_"
    model_type = PhaseOnlyAnchor
    binding = dict(source=sha(__file__), protocol=sha(destination.parent / "PROTOCOL.json"))
    binding_path = destination / "ANALYSIS_BINDING.json"
    if binding_path.exists():
        assert read(binding_path) == binding
    else:
        write(binding_path, binding)
    recipe = read(OUT / "PROTOCOL.json")
    endpoints = {}
    for mode in ("EVENT", "ZERO"):
        directory = OUT / f"fits/{prefix}{mode}_S{args.seed}"
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        assert receipt["status"] == "COMPLETE" and receipt["completed_updates"] == 2500
        assert sha(directory / "checkpoint_last.pt") == receipt["checkpoint_sha256"]
        endpoints[mode] = receipt
    write(destination / "ENDPOINTS_SEALED.json", endpoints)
    path = OUT / "cache/V/states.npz"
    assert sha(path) == read(path.parent / "SEALED.json")["states_sha256"]
    with np.load(path, allow_pickle=False) as z:
        features = torch.from_numpy(z["features"]).to("cpu")
        width = torch.from_numpy(z["rgb_width_logits"]).to("cpu")
        tokens = z["sample_token"]
        predictions = {"R_CTX_SAME_CACHE": phase_to_ttc(torch.from_numpy(z["rgb_phase"])).numpy()}
    for mode in ("EVENT", "ZERO"):
        state = torch.load(
            OUT / f"fits/{prefix}{mode}_S{args.seed}/checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = model_type(mode).to("cpu").eval()
        model.load_state_dict(state["model_state_dict"], strict=True)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tokens), 128):
                output = model(features[start : start + 128], width[start : start + 128])
                chunks.append(phase_to_ttc(output["point_phase"]).cpu().numpy())
        predictions["R_ANCHOR_" + mode] = np.concatenate(chunks)
    data = pd.DataFrame({"sample_token": tokens, **predictions})
    path = destination / "V_PREDICTIONS_UNSCORED.csv"
    data.to_csv(path, index=False)
    write(
        destination / "V_PREDICTIONS_SEALED.json",
        dict(
            sha256=sha(path),
            endpoints=endpoints,
            targets_read_for_scoring=False,
            prior_V_exposure=True,
        ),
    )
    historical = {}
    for name in ("F_TRUE", "R_CTX", "E_CTX_MATCHED"):
        frame = pd.read_csv(RUN / f"predictions/V/{name}.csv").set_index("sample_token")
        assert frame.index.is_unique and set(frame.index) == set(tokens)
        historical[name] = frame.loc[tokens]
    truth = historical["R_CTX"].target_ttc.to_numpy()
    groups = historical["R_CTX"].group_id.to_numpy()
    for name, frame in historical.items():
        assert np.array_equal(frame.target_ttc, truth) and np.array_equal(frame.group_id, groups)
        predictions[name] = frame.prediction_ttc.to_numpy()
    scores = {name: score_signed_ttc(truth, pred, groups) for name, pred in predictions.items()}
    pd.DataFrame(
        dict(sample_token=tokens, group_id=groups, target_ttc=truth, **predictions)
    ).to_csv(destination / "V_SCORED_PREDICTIONS.csv", index=False)
    group_rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        row = {"group_id": group}
        for name, pred in predictions.items():
            row[name] = score_signed_ttc(truth[mask], pred[mask], groups[mask])[
                "group_macro_bucket_MiD"
            ]
        group_rows.append(row)
    table = pd.DataFrame(group_rows)
    table.to_csv(destination / "V_GROUP_CONTRASTS.csv", index=False)
    # Every V group has every native bucket: the group bootstrap then exactly
    # matches the registered bucket-weighted group-macro statistic.
    for name, score in scores.items():
        assert np.isfinite(table[name].to_numpy(dtype=float)).all()
        assert np.isclose(table[name].mean(), score["group_macro_bucket_MiD"], atol=1e-10)
    delta = table.R_ANCHOR_EVENT.to_numpy() - table.R_ANCHOR_ZERO.to_numpy()
    draws = np.random.default_rng(20261012).choice(delta, (4000, len(delta))).mean(1)
    ci = np.quantile(draws, [0.025, 0.975]).tolist()
    gates = {}
    for name in ("R_ANCHOR_EVENT", "R_ANCHOR_ZERO"):
        new, rgb, fusion = scores[name], scores["R_CTX"], scores["F_TRUE"]
        checks = dict(
            aggregate_vs_RGB=new["group_macro_bucket_MiD"] <= 0.97 * rgb["group_macro_bucket_MiD"],
            mae_vs_RGB=new["diagnostics"]["mae_s"] <= rgb["diagnostics"]["mae_s"],
            aggregate_vs_fusion=new["group_macro_bucket_MiD"] <= fusion["group_macro_bucket_MiD"],
            mae_vs_fusion=new["diagnostics"]["mae_s"] <= fusion["diagnostics"]["mae_s"],
            sign=new["diagnostics"]["sign_error_rate"] <= rgb["diagnostics"]["sign_error_rate"],
            no_failures=new["num_failed_predictions"] == 0,
        )
        for band in ("crucial", "small", "large", "negative"):
            checks[band] = (
                new["bins"][band]["group_macro_mid"] <= 1.02 * rgb["bins"][band]["group_macro_mid"]
            )
        gates[name] = dict(checks=checks, eligible=all(checks.values()))
    event_signal = ci[1] < 0
    write(
        destination / "V_RESULTS.json",
        dict(
            study=args.study,
            scores=scores,
            gates=gates,
            event_minus_zero=dict(delta=float(delta.mean()), ci95=ci, draws=4000),
            complementary_event_gate=event_signal,
            candidate="R_ANCHOR_EVENT"
            if gates["R_ANCHOR_EVENT"]["eligible"] and event_signal
            else None,
            selection_role=recipe["validation"],
            seed=args.seed,
            external_evaluation_used_for_selection=False,
        ),
    )
    print(
        {
            name: (value["group_macro_bucket_MiD"], value["diagnostics"]["mae_s"])
            for name, value in scores.items()
        },
        flush=True,
    )
    print("GATES", gates, "EVENT-ZERO", ci, flush=True)


if __name__ == "__main__":
    main()

"""Append exact readout-stage traces to the previously selected H and public failure episodes."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.multimodal_validation.common import read, sha, write

from .common import OUT
from .phase_only_anchor import PhaseOnlyAnchor


def trace(model: PhaseOnlyAnchor, features: torch.Tensor, widths: torch.Tensor) -> dict:
    with torch.inference_mode():
        plain = model(features, widths)
        captured = []
        handle = model.output.register_forward_hook(
            lambda _module, _inputs, value: captured.append(value.detach().clone())
        )
        try:
            instrumented = model(features, widths)
        finally:
            handle.remove()
        assert all(torch.equal(plain[k], instrumented[k]) for k in plain)
        assert torch.equal(captured[0], plain["correction"])
        rgb_raw = float(features[0, 160])
        raw, emitted = float(plain["raw_location"][0]), float(plain["point_phase"][0])
        return dict(
            R_raw_phase=rgb_raw,
            R_emitted_phase=float(features[0, 161]),
            R_expert_phases=features[0, 162:165].tolist(),
            E_raw_phase=float(features[0, 330]),
            E_emitted_phase=float(features[0, 331]),
            E_expert_phases=features[0, 332:335].tolist(),
            R_clock=features[0, 165:169].tolist(),
            E_clock=features[0, 335:339].tolist(),
            R_observations=int(round(float(features[0, 169]) * 8)),
            E_observations=int(round(float(features[0, 339]) * 8)),
            readout_correction_phase=float(0.03 * captured[0][0, 0]),
            new_raw_phase=raw,
            new_emitted_phase=emitted,
            raw_head_ttc=float(plain["ttc"][0]),
            emitted_ttc=float(phase_to_ttc(plain["point_phase"])[0]),
            sign_changed_from_R=bool(np.sign(raw) != np.sign(rgb_raw)),
            phase_moved_toward_zero=abs(raw) < abs(rgb_raw),
            conversion_changed_phase=raw != emitted,
            hook_output_bitwise_parity=True,
        )


def main() -> None:
    torch.set_num_threads(1)
    models, checkpoints = {}, {}
    for mode in ("EVENT", "ZERO"):
        path = OUT / f"fits/R_ANCHOR_PHASE_ONLY_{mode}_S7/checkpoint_last.pt"
        models[mode] = PhaseOnlyAnchor(mode).eval().requires_grad_(False)
        models[mode].load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        )
        checkpoints[mode] = sha(path)
    records = []
    selection_path = PREVIOUS / "H_CONTROLS_SEALED.json"
    selected_h = read(selection_path)["tokens"]
    with np.load(OUT / "cache/H/states.npz", allow_pickle=False) as z:
        lookup = {str(token): i for i, token in enumerate(z["sample_token"])}
        values, width = z["features"], z["rgb_width_logits"]
        for token in selected_h:
            index = lookup[token]
            for mode, model in models.items():
                record = trace(
                    model,
                    torch.from_numpy(values[index : index + 1]),
                    torch.from_numpy(width[index : index + 1]),
                )
                records.append(
                    dict(
                        cohort="H",
                        query_id=token,
                        arm=mode,
                        input_route="sealed H states",
                        **record,
                    )
                )
    previous_traces = read(PREVIOUS / "STAGE_TRACE_DETAILS.json")
    selected = sorted(
        {(r["scope"], r["query_id"]) for r in previous_traces if r["scope"] in {"FCWD", "DEV32"}}
    )
    for cohort, qid in selected:
        route = "native_gpu" if cohort == "FCWD" else "cpu"
        root = "external_states_gpu_native" if cohort == "FCWD" else "external_states"
        path = OUT / root / cohort / (hashlib.sha256(qid.encode()).hexdigest() + ".npz")
        source = OUT / f"studies/PHASE_ONLY/S7/external_evaluation_v2/{route}/{cohort}"
        reference = pd.read_csv(source / "SCORED_PREDICTIONS.csv", float_precision="round_trip")
        row = reference.set_index("query_id").loc[qid]
        with np.load(path, allow_pickle=False) as z:
            for mode, model in models.items():
                if "features" in z:
                    record = trace(
                        model,
                        torch.from_numpy(z["features"]),
                        torch.from_numpy(z["rgb_width_logits"]),
                    )
                else:
                    record = dict(
                        emitted_ttc=float(phase_to_ttc(torch.from_numpy(z["event_phase"]))[0]),
                        fallback="Exact E_CTX",
                        hook_output_bitwise_parity=True,
                    )
                assert record["emitted_ttc"] == float(row["R_ANCHOR_" + mode])
                records.append(
                    dict(
                        cohort=cohort,
                        query_id=qid,
                        arm=mode,
                        input_route=route + " parents; CPU readout",
                        state_sha256=sha(path),
                        sealed_prediction_exact=True,
                        truth_ttc=None
                        if pd.isna(row.truth_ttc_seconds)
                        else float(row.truth_ttc_seconds),
                        R_CTX=None
                        if pd.isna(row.R_CTX_SAME_ROUTE)
                        else float(row.R_CTX_SAME_ROUTE),
                        F_TRUE=float(row.F_TRUE_SAME_ROUTE),
                        **record,
                    )
                )
    write(
        OUT / "PHASE_ONLY_ATTRIBUTION.json",
        dict(
            source_sha256=sha(__file__),
            checkpoints=checkpoints,
            rows=records,
            old_trace_selection_sha256=sha(PREVIOUS / "STAGE_TRACE_DETAILS.json"),
            H_selection_sha256=sha(selection_path),
            all_hook_outputs_bitwise_equal=True,
            all_external_predictions_equal_sealed=True,
            optimizer_updates=0,
            selection="32 original hash-selected H controls and 20 prior public episodes",
            inference_scope="Frozen readout on sealed parent states; "
            "no new sensor replay or model selection",
        ),
    )
    pd.DataFrame(records).to_csv(OUT / "PHASE_ONLY_ATTRIBUTION.csv", index=False)
    shown = pd.DataFrame(records)
    print(
        pd.DataFrame(
            shown[
                shown.query_id.isin(
                    ["CCRs-1-medium-100-overlap-100:01", "CCRs-1-low-100-overlap-100:24"]
                )
            ][
                [
                    "query_id",
                    "arm",
                    "truth_ttc",
                    "R_CTX",
                    "F_TRUE",
                    "emitted_ttc",
                    "R_raw_phase",
                    "readout_correction_phase",
                    "new_raw_phase",
                ]
            ]
        ).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

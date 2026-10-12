"""Internal H/V phase traces of sealed readouts; no external label selection."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.common import read, sha, write

from .common import OUT
from .model import RGBAnchor
from .phase_units import STUDY, PhaseUnitAnchor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("baseline", "units"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    destination = OUT if args.study == "baseline" else STUDY
    model_type = RGBAnchor if args.study == "baseline" else PhaseUnitAnchor
    prefix = "R_ANCHOR_" if args.study == "baseline" else "R_ANCHOR_UNITS_"
    endpoints = read(destination / "ENDPOINTS_SEALED.json")
    summaries = []
    for role in ("H", "V"):
        path = OUT / f"cache/{role}/states.npz"
        assert sha(path) == read(path.parent / "SEALED.json")["states_sha256"]
        with np.load(path, allow_pickle=False) as z:
            x = torch.from_numpy(z["features"])
            widths = torch.from_numpy(z["rgb_width_logits"])
            ids, groups, target = z["sample_token"], z["group_id"], z["target_phase"]
        for mode in ("EVENT", "ZERO"):
            checkpoint = OUT / f"fits/{prefix}{mode}_S7/checkpoint_last.pt"
            assert sha(checkpoint) == endpoints[mode]["checkpoint_sha256"]
            model = model_type(mode).eval()
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
            )
            parts = []
            with torch.inference_mode():
                for start in range(0, len(x), 128):
                    features = x[start : start + 128]
                    output = model(features, widths[start : start + 128])
                    parts.append(
                        np.stack(
                            [
                                features[:, 160].numpy(),
                                features[:, 161].numpy(),
                                features[:, 331].numpy(),
                                output["correction"][:, 0].numpy(),
                                output["raw_location"].numpy(),
                                output["point_phase"].numpy(),
                                output["ttc"].numpy(),
                                output["q10"].numpy(),
                                output["q90"].numpy(),
                            ],
                            axis=1,
                        )
                    )
            frame = pd.DataFrame(
                np.concatenate(parts),
                columns=[
                    "raw_R",
                    "phase_R",
                    "phase_E",
                    "correction_coefficient",
                    "raw_new",
                    "phase_new",
                    "ttc_new",
                    "q10",
                    "q90",
                ],
            )
            frame.insert(0, "sample_token", ids)
            frame.insert(1, "group_id", groups)
            frame["target_phase"] = target
            frame["residual_phase"] = frame.raw_new - frame.raw_R
            frame["phase_error_R"] = (frame.phase_R - target).abs()
            frame["phase_error_new"] = (frame.phase_new - target).abs()
            frame["regret_phase"] = frame.phase_error_new - frame.phase_error_R
            frame["sign_changed"] = (frame.raw_R < 0) != (frame.raw_new < 0)
            frame["at_cap60"] = frame.ttc_new.abs() >= 59.99999
            frame["target_ttc"] = phase_to_ttc(torch.from_numpy(target)).numpy()
            target_path = destination / f"traces/{role}_{mode}.csv"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target_path, index=False)
            parity = None
            if role == "V":
                gpu = pd.read_csv(destination / "V_PREDICTIONS_UNSCORED.csv")
                assert np.array_equal(frame.sample_token, gpu.sample_token)
                delta = np.abs(frame.ttc_new - gpu["R_ANCHOR_" + mode])
                parity = dict(max_ttc_delta=float(delta.max()), mean_ttc_delta=float(delta.mean()))
            for group, part in frame.groupby("group_id"):
                summaries.append(
                    dict(
                        role=role,
                        mode=mode,
                        group_id=group,
                        queries=len(part),
                        improved=int((part.regret_phase < 0).sum()),
                        worsened=int((part.regret_phase > 0).sum()),
                        mean_phase_error_R=float(part.phase_error_R.mean()),
                        mean_phase_error_new=float(part.phase_error_new.mean()),
                        sign_changes=int(part.sign_changed.sum()),
                        cap60=int(part.at_cap60.sum()),
                    )
                )
            write(
                target_path.with_suffix(".json"),
                dict(
                    source_sha256=sha(__file__),
                    checkpoint_sha256=sha(checkpoint),
                    cache_sha256=sha(path),
                    trace_sha256=sha(target_path),
                    prediction_route="CPU FP32 batch128; GPU V parity measured separately",
                    V_numeric_parity=parity,
                    optimizer_updates=0,
                    role="Internal descriptive H/V; no causal attribution; H states not OOF",
                ),
            )
    pd.DataFrame(summaries).to_csv(destination / "traces/GROUP_SUMMARY.csv", index=False)


if __name__ == "__main__":
    main()

"""H-only complementarity and dead-zone diagnostics; oracle is not a predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from operational.multimodal_validation.common import admission, write

from .common import OUT


def main() -> None:
    torch.set_num_threads(1)
    admission()
    with np.load(OUT / "cache/H/states.npz", allow_pickle=False) as z:
        assert set(map(str, z["role"])) == {"H"}
        x, truth, groups, mass = (z[k] for k in ("features", "target_phase", "group_id", "mass"))
        r, e = z["rgb_phase"], z["event_phase"]
    loss_r, loss_e = abs(r - truth), abs(e - truth)
    low, high = -np.log1p(0.1 / 60), -np.log1p(-0.1 / 60)
    masks = {
        "ALL": np.ones(len(truth), bool),
        "true_positive": truth > 0,
        "true_negative": truth < 0,
        "prediction_sign_disagreement": (r < 0) != (e < 0),
        "RGB_wrong_sign": (r < 0) != (truth < 0),
        "RGB_raw_in_dead_zone": (x[:, 160] >= low) & (x[:, 160] <= high),
    }
    masks.update({"group_" + str(g): groups == g for g in sorted(set(groups))})
    rows = []
    for name, mask in masks.items():
        rows.append(
            dict(
                subset=name,
                rows=int(mask.sum()),
                event_better=int((loss_e[mask] < loss_r[mask]).sum()),
                rgb_mean_phase_error=float(loss_r[mask].mean()),
                event_mean_phase_error=float(loss_e[mask].mean()),
                oracle_min_phase_error=float(np.minimum(loss_r[mask], loss_e[mask]).mean()),
                global_mass_contribution_rgb=float((mass[mask] * loss_r[mask]).sum()),
                global_mass_contribution_oracle=float(
                    (mass[mask] * np.minimum(loss_r[mask], loss_e[mask])).sum()
                ),
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "H_COMPLEMENTARITY.csv", index=False)
    write(
        OUT / "H_COMPLEMENTARITY.json",
        dict(
            scope="in-sample supervised H; no V or external targets read",
            oracle_uses_GT=True,
            oracle_deployable=False,
            interpretation=(
                "descriptive ceiling for choosing these two predictions, not all possible fusion"
            ),
            rows=rows,
            optimizer_updates=0,
        ),
    )
    print(rows[0], flush=True)


if __name__ == "__main__":
    main()

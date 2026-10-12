"""Describe the admitted P geometry-MLP gradient by TTC range without fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.range_transfer.common import sha, write
from operational.representation_audit.nonlinear import geometry_parent, initialize
from operational.representation_audit.run import RUN
from operational.representation_audit.spatial_model import SpatialVectors


def main() -> None:
    torch.set_num_threads(1)
    receipt = geometry_parent(7)
    values = SpatialVectors("P").values
    model, *_ = initialize(7, "NLH", device="cpu")
    model.load_state_dict(
        torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=False)[
            "model_state_dict"
        ],
        strict=True,
    )
    model.eval()
    target = values["target_ttc"]
    masks = {
        "0-1": (target > 0) & (target <= 1),
        "1-3": (target > 1) & (target <= 3),
        "3-6": (target > 3) & (target <= 6),
        "6plus": target > 6,
        "negative": target < 0,
    }
    rng = np.random.default_rng(12345)
    gradients, rows, samples = {}, [], {}
    keys = (
        "height_features",
        "target_valid",
        "target_height_normalized",
        "visible_height",
        "analytic",
    )
    for band, mask in masks.items():
        ids = rng.choice(np.flatnonzero(mask), 128, replace=True)
        inputs = {key: torch.from_numpy(values[key][ids].copy()) for key in keys}
        loss = model.geometry_loss(inputs)
        mass = float(mask.mean())
        gradient = torch.autograd.grad(mass * loss, model.height_mlp.parameters())
        vector = torch.cat([v.flatten() for v in gradient]).double()
        gradients[band] = vector
        samples[band] = values["sample_token"][ids].astype(str).tolist()
        rows.append(
            {
                "band": band,
                "population": int(mask.sum()),
                "sampling_mass": mass,
                "conditional_loss": float(loss.detach()),
                "gradient_contribution_norm": float(vector.norm()),
            }
        )
    cosines = []
    for left, a in gradients.items():
        for right, b in gradients.items():
            denominator = a.norm() * b.norm()
            cosines.append(
                {
                    "left": left,
                    "right": right,
                    "cosine": float(a @ b / denominator) if denominator > 0 else None,
                }
            )
    folder = RUN / "geometry_gradient_I"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "CONTRIBUTIONS.csv", index=False)
    pd.DataFrame(cosines).to_csv(folder / "COSINES.csv", index=False)
    write(
        folder / "SCOPE.json",
        {
            "role": "P",
            "optimizer_updates": 0,
            "source_sha256": sha(Path(__file__)),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "objective": "registered normalized log-height and temporal box geometry only",
            "parameters": "height_mlp only",
            "device": "CPU",
            "norm_dot_accumulation": "float64",
            "estimator": "mass times 128-draw conditional mean; not exact full-P gradient",
            "TTC_labels": "descriptive stratification only; never passed to geometry_loss",
            "sample_tokens": samples,
            "norms_are_additive_masses": False,
        },
    )


if __name__ == "__main__":
    main()

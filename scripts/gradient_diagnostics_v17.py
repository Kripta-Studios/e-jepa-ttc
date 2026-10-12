"""Stratified H gradient contribution estimates at frozen A/D head endpoints."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import training_loss
from operational.range_transfer.common import read, sha, write
from operational.representation_audit.model import RepresentationRefiner
from operational.representation_audit.run import INPUTS, RUN, cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    h = cache("H")
    if args.directional:
        with np.load(RUN / "cache/D_H.npz", allow_pickle=False) as archive:
            h = {k: archive[k] for k in archive.files}
        names = [f"D-S{s}" for s in (7, 13, 23)]
        freeze = read(RUN / "D_ENDPOINTS.json")
    else:
        names = [f"{a}-S{s}" for s in (7, 13, 23) for a in ("Z", "G")]
        freeze = read(RUN / "QUEUE_FREEZE.json")
    truth = -0.1 / np.expm1(-h["target_phase"].astype(np.float64))
    masks = {
        "0-1": (truth > 0.1) & (truth <= 1),
        "1-3": (truth > 1) & (truth <= 3),
        "3-6": (truth > 3) & (truth <= 6),
        "6-10": (truth > 6) & (truth <= 10),
        "negative": truth < 0,
    }
    if not np.all(np.stack(list(masks.values())).sum(0) == 1):
        raise ValueError("Bands do not partition H")
    generator = np.random.default_rng(12345)
    samples = {
        band: generator.choice(np.flatnonzero(mask), 128, replace=True)
        for band, mask in masks.items()
    }
    rows, cosines = [], []
    for name in names:
        path = RUN / f"fits/{name}/checkpoint_last.pt"
        if sha(path) != freeze[name]["checkpoint_sha256"]:
            raise ValueError("Endpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = RepresentationRefiner(not name.startswith("Z"))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        parameters = list(model.parameters())
        vectors = {}
        for band, indices in samples.items():
            values = {
                k: torch.from_numpy(h[k][indices].copy()) for k in (*INPUTS, "mass", "target_phase")
            }
            output = model(*(values[k] for k in INPUTS))
            conditional = training_loss(
                output, values["target_phase"], values["expert_phase"], values["mass"], len(truth)
            )
            contribution = conditional * float(masks[band].mean())
            derivatives = torch.autograd.grad(contribution, parameters)
            vector = torch.cat([item.flatten() for item in derivatives]).detach().double()
            vectors[band] = vector
            rows.append(
                {
                    "fit": name,
                    "band": band,
                    "population": int(masks[band].sum()),
                    "native_mass": float(h["mass"][masks[band]].sum()),
                    "gradient_contribution_l2_estimate": float(vector.norm()),
                    "sampled_queries": 128,
                    "loss_contribution_estimate": float(contribution.detach()),
                }
            )
        for left, a in vectors.items():
            for right, b in vectors.items():
                cosines.append(
                    {
                        "fit": name,
                        "left": left,
                        "right": right,
                        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-20)),
                    }
                )
    folder = RUN / ("directional_gradient" if args.directional else "head_gradient")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "CONTRIBUTIONS.csv", index=False)
    pd.DataFrame(cosines).to_csv(folder / "COSINES.csv", index=False)
    write(
        folder / "SCOPE.json",
        {
            "optimizer_updates": 0,
            "role": "H",
            "draw_seed": 12345,
            "queries_per_band": 128,
            "estimator": "P(band) times conditional uniform minibatch gradient of N*mass*L0",
            "not_exact_full_H_gradient": True,
            "norms_are_not_additive_mass": True,
            "sample_tokens": {
                band: h["sample_token"][indices].astype(str).tolist()
                for band, indices in samples.items()
            },
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

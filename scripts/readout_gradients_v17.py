"""Sampled P-band NLL gradients, separate from nonadditive top-k/risk terms."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.models.causal_scale_ttc import target_log_ratio_from_ttc
from operational.range_transfer.common import read, sha, write
from operational.representation_audit.geometry import initialize as init_geometry
from operational.representation_audit.precision import PRECISION, Vectors
from operational.representation_audit.precision import initialize as init_precision
from operational.representation_audit.run import RUN
from operational.representation_audit.temporal_geometry import initialize as init_temporal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", choices=("B", "E", "F"))
    queue = parser.parse_args().queue
    torch.set_num_threads(1)
    source, init = {
        "B": ("PRECISION_ENDPOINTS.json", init_precision),
        "E": ("GEOMETRY_SCREEN_ENDPOINTS.json", init_geometry),
        "F": ("TEMP_GEOMETRY_SCREEN_ENDPOINTS.json", init_temporal),
    }[queue]
    endpoints = read(RUN / source)
    data = Vectors("P").values
    truth = data["target_ttc"]
    masks = {
        "0-1": (truth > 0) & (truth <= 1),
        "1-3": (truth > 1) & (truth <= 3),
        "3-6": (truth > 3) & (truth <= 6),
        "6plus": truth > 6,
        "negative": truth < 0,
    }
    if not np.all(np.stack(list(masks.values())).sum(0) == 1):
        raise ValueError("P bands do not partition the cohort")
    rng = np.random.default_rng(12345)
    samples = {
        band: rng.choice(np.flatnonzero(mask), 128, replace=True) for band, mask in masks.items()
    }
    rows, cosines = [], []
    for name, endpoint in endpoints.items():
        arm, seed = name.split("-S")
        model, *_ = (
            init(int(seed), device="cpu") if queue == "B" else init(int(seed), arm, device="cpu")
        )
        path = __import__("pathlib").Path(endpoint["checkpoint_path"])
        if sha(path) != endpoint["checkpoint_sha256"]:
            raise ValueError("Readout endpoint changed")
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"],
            strict=True,
        )
        model.eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        vectors = {}
        for band, ids in samples.items():
            inputs = {
                k: torch.from_numpy(v[ids].copy())
                for k, v in data.items()
                if k != "sample_token" and not k.startswith(("target_", "dino_"))
            }
            output = model(inputs)
            target, valid = target_log_ratio_from_ttc(
                torch.from_numpy(truth[ids]), inputs["delta"][:, -1]
            )
            if not valid.all():
                raise ValueError("Nonphysical sampled target")
            logvar = output.log_ratio_log_variance[:, -1]
            precision = torch.full_like(logvar, PRECISION) if arm == "U1" else (-logvar).exp()
            nll = (
                0.5 * precision * (output.pair_log_height_ratio[:, -1] - target).square()
                + 0.5 * logvar
            ).mean()
            mass = float(masks[band].mean())
            gradients = torch.autograd.grad(mass * nll, parameters)
            vector = torch.cat([g.flatten() for g in gradients]).double()
            vectors[band] = vector
            rows.append(
                {
                    "fit": name,
                    "band": band,
                    "population": int(masks[band].sum()),
                    "sampling_mass": mass,
                    "sample_draws": 128,
                    "mean_precision_sample": float(precision.detach().mean()),
                    "mean_nll_sample": float(nll.detach()),
                    "gradient_contribution_norm": float(vector.norm()),
                }
            )
        for left in vectors:
            for right in vectors:
                a, b = vectors[left], vectors[right]
                den = a.norm() * b.norm()
                cosines.append(
                    {
                        "fit": name,
                        "left": left,
                        "right": right,
                        "cosine": float(a @ b / den) if den > 0 else None,
                    }
                )
    folder = RUN / f"readout_gradient_{queue}"
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "CONTRIBUTIONS.csv", index=False)
    pd.DataFrame(cosines).to_csv(folder / "COSINES.csv", index=False)
    write(
        folder / "SCOPE.json",
        {
            "role": "P",
            "optimizer_updates": 0,
            "source": sha(__import__("pathlib").Path(__file__)),
            "objective": "NLL component only; excludes nonadditive top-k, risk and other terms",
            "estimator": (
                "band population mass times 128-draw conditional mean gradient; "
                "not exact full-P gradient"
            ),
            "dropout": "disabled for endpoint diagnosis",
            "accumulation": "float64 gradient norms/dots",
            "samples": {
                band: data["sample_token"][ids].astype(str).tolist()
                for band, ids in samples.items()
            },
        },
    )


if __name__ == "__main__":
    main()

"""H-only descriptive reliance audit of a durably saved RGB-anchor checkpoint."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from operational.multimodal_validation.common import admission, read, sha, write

from .common import OUT
from .model import BLOCK, PREDICTIVE, RGBAnchor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("EVENT", "ZERO"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    admission()
    directory = OUT / f"fits/R_ANCHOR_{args.mode}_S7"
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    path = directory / "checkpoint_last.pt"
    assert sha(path) == receipt["checkpoint_sha256"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = RGBAnchor(args.mode).eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    cache = OUT / "cache/H/states.npz"
    assert sha(cache) == read(cache.parent / "SEALED.json")["states_sha256"]
    with np.load(cache, allow_pickle=False) as z:
        x = torch.from_numpy(z["features"])
        width = torch.from_numpy(z["rgb_width_logits"])
        truth, mass, rgb = z["target_phase"], z["mass"], z["rgb_phase"]
        assert set(z["role"].tolist()) == {"H"}
    permutation = torch.from_numpy(np.random.default_rng(20261012).permutation(len(x)))
    result = {}
    for label in ("native", "event_predictive_permuted", "event_predictive_zero"):
        features = x.clone()
        section = slice(BLOCK, BLOCK + PREDICTIVE)
        if label.endswith("permuted"):
            features[:, section] = x[permutation, section]
        elif label.endswith("zero"):
            features[:, section] = 0
        phases, raw = [], []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                out = model(features[start : start + 128], width[start : start + 128])
                phases.append(out["point_phase"].numpy())
                raw.append(out["raw_location"].numpy())
        pred, locations = np.concatenate(phases), np.concatenate(raw)
        result[label] = dict(
            phase_mae=float(np.abs(pred - truth).mean()),
            phase_global_mass_error=float(np.dot(np.abs(pred - truth), mass)),
            sign_errors=int(((pred < 0) != (truth < 0)).sum()),
            sign_changes_vs_RGB=int(((pred < 0) != (rgb < 0)).sum()),
            raw_correction_abs_mean=float(np.abs(locations - x[:, 160].numpy()).mean()),
        )
    result["R_CTX"] = dict(
        phase_mae=float(np.abs(rgb - truth).mean()),
        phase_global_mass_error=float(np.dot(np.abs(rgb - truth), mass)),
        sign_errors=int(((rgb < 0) != (truth < 0)).sum()),
    )
    write(
        OUT / f"H_ATTRIBUTION_{args.mode}_{receipt['completed_updates']}.json",
        dict(
            checkpoint=receipt,
            population=len(x),
            results=result,
            optimizer_updates=0,
            source_sha256=sha(__file__),
            role="H only, supervised in-sample states",
            inference="CPU FP32; not the CUDA training loss trace",
            limitation="Perturbed states can be off-manifold; reliance is not causal attribution",
            scientific_endpoint=receipt["scientific_endpoint"],
        ),
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()

"""H-only first-linear decomposition; no targets, optimizer or causal claims."""

from __future__ import annotations

import numpy as np
import torch

from operational.multimodal_validation.common import read, sha, write

from .common import OUT
from .model import RGBAnchor


def main() -> None:
    torch.set_num_threads(1)
    directory = OUT / "fits/R_ANCHOR_EVENT_S7"
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    path = directory / "checkpoint_last.pt"
    assert sha(path) == receipt["checkpoint_sha256"]
    model = RGBAnchor("EVENT").eval()
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    cache = OUT / "cache/H/states.npz"
    assert sha(cache) == read(cache.parent / "SEALED.json")["states_sha256"]
    with np.load(cache, allow_pickle=False) as z:
        features = torch.from_numpy(z["features"])
    weight = model.get_submodule("mlp.0").get_parameter("weight").detach()
    rows = []
    with torch.no_grad():
        for modality, offset in (("R", 0), ("E", 170)):
            for group, lo, hi in (
                ("hidden", 0, 160),
                ("raw_and_emitted_phase", 160, 162),
                ("expert_phase", 162, 165),
                ("clock", 165, 169),
                ("count", 169, 170),
            ):
                x = features[:, offset + lo : offset + hi]
                w = weight[:, offset + lo : offset + hi]
                component = x @ w.T
                centered = (x - x.mean(0)) @ w.T
                rows.append(
                    dict(
                        modality=modality,
                        group=group,
                        dimensions=hi - lo,
                        median_input_std=float(x.std(0).median()),
                        weight_frobenius=float(w.norm()),
                        preactivation_rms=float(component.square().mean().sqrt()),
                        centered_preactivation_rms=float(centered.square().mean().sqrt()),
                    )
                )
    write(
        OUT / f"H_LINEAR_CONTRIBUTIONS_{receipt['completed_updates']}.json",
        dict(
            checkpoint=receipt,
            rows=rows,
            cache_sha256=sha(cache),
            source_sha256=sha(__file__),
            targets_read=False,
            optimizer_updates=0,
            stage="first linear before LayerNorm and SiLU",
            limitation="Correlated blocks; descriptive H-only decomposition",
        ),
    )
    print(rows, flush=True)


if __name__ == "__main__":
    main()

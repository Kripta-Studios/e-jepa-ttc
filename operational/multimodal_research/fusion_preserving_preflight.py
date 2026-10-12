"""Verify the new zero-initialized head against actual native F_TRUE forwards."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from operational.multimodal_validation.common import RUN, read, sha, write
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .common import OUT
from .fusion_preserving_anchor import FusionPreservingAnchor


def main() -> None:
    torch.set_num_threads(1)
    path = RUN / "fits/F_TRUE/checkpoint_last.pt"
    assert (
        sha(path)
        == read(OUT / "FUSION_PRESERVING_PREREGISTRATION.json")["parent_checkpoint_sha256"]
    )
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    native = DualClockFusion("F_TRUE").eval()
    native.load_state_dict(state)
    results = []
    for role in ("H", "V"):
        inputs = {}
        for modality in ("E", "R"):
            path = RUN / f"features/{role}/{modality}_HISTORY.npz"
            assert sha(path) == read(path.with_suffix(".json"))["output_sha256"]
            with np.load(path, allow_pickle=False) as z:
                inputs[modality] = {
                    key: z[key][:128].copy()
                    for key in ("features", "timing", "valid", "expert_phase", "sample_token")
                }
        e, r = inputs["E"], inputs["R"]
        assert np.array_equal(e["sample_token"], r["sample_token"])
        tensors = [
            torch.from_numpy(data[key])
            for data, key in (
                (e, "features"),
                (e, "timing"),
                (e, "valid"),
                (e, "expert_phase"),
                (r, "features"),
                (r, "timing"),
                (r, "valid"),
            )
        ]
        with torch.inference_mode():
            expected = deterministic_forward(native, *tensors)
        parts = []
        for directory in (OUT / f"cache/{role}", OUT / f"fusion_states/{role}"):
            assert sha(directory / "states.npz") == read(directory / "SEALED.json")["states_sha256"]
            with np.load(directory / "states.npz", allow_pickle=False) as z:
                assert np.array_equal(z["sample_token"][:128], e["sample_token"])
                parts.append(torch.from_numpy(z["features"][:128].copy()))
        features = torch.cat(parts, -1)
        torch.testing.assert_close(features[:, 340:500], expected["hidden"], rtol=0, atol=0)
        for mode in ("REFINED_RGB", "ZERO_RGB"):
            model = FusionPreservingAnchor(mode).eval()
            model.initialize_width(state)
            with torch.inference_mode():
                actual = model(features, torch.zeros(128, 2))
            deltas = {}
            for key in ("raw_location", "point_phase", "ttc", "q10", "q90"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                deltas[key] = float((actual[key] - expected[key]).abs().max())
            results.append(dict(role=role, mode=mode, queries=128, max_delta=deltas))
    write(
        OUT / "FUSION_PRESERVING_INITIAL_PARITY.json",
        dict(
            results=results,
            selection="first native batch128 in each sealed H/V cache; no labels",
            optimizer_updates=0,
            targets_read=False,
            source_sha256=sha(__file__),
        ),
    )
    print(results, flush=True)


if __name__ == "__main__":
    main()

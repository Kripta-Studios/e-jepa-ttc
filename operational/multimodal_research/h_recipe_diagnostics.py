"""Descriptive H-only losses of completed fixed readouts; no additional fitting."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import fusion_loss
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.multimodal_validation.common import admission, read, sha, write

from .common import OUT
from .fusion_anchor import FusionStateAnchor
from .model import RGBAnchor
from .phase_units import PhaseUnitAnchor
from .relative_phase import RelativePhaseAnchor


def main() -> None:
    torch.set_num_threads(1)
    cache = OUT / "cache/H/states.npz"
    fused = OUT / "fusion_states/H/states.npz"
    for path in (cache, fused):
        assert sha(path) == read(path.parent / "SEALED.json")["states_sha256"]
    with np.load(cache, allow_pickle=False) as z, np.load(fused, allow_pickle=False) as f:
        assert set(z["role"]) == {"H"}
        assert np.array_equal(z["sample_token"], f["sample_token"])
        features = torch.from_numpy(z["features"])
        extended = torch.cat([features, torch.from_numpy(f["features"])], -1)
        widths = torch.from_numpy(z["rgb_width_logits"])
        truth, mass = torch.from_numpy(z["target_phase"]), torch.from_numpy(z["mass"])
    rows = []
    recipes = [
        ("BASELINE", "R_ANCHOR_", RGBAnchor, ("EVENT", "ZERO")),
        ("PHASE_UNITS", "R_ANCHOR_UNITS_", PhaseUnitAnchor, ("EVENT", "ZERO")),
        ("RELATIVE_PHASE", "R_ANCHOR_RELATIVE_PHASE_", RelativePhaseAnchor, ("EVENT", "ZERO")),
        ("RGB_REGRET", "R_ANCHOR_RGB_REGRET_", PhaseUnitAnchor, ("EVENT", "ZERO")),
        ("FUSION_STATE", "R_ANCHOR_FUSION_STATE_", FusionStateAnchor, ("FUSED", "ZERO_FUSED")),
    ]
    parent_error = (emitted_phase(features[:, 160]) - truth).abs()
    for study, prefix, model_type, arms in recipes:
        for arm in arms:
            admission()
            directory = OUT / f"fits/{prefix}{arm}_S7"
            receipt = read(directory / "CHECKPOINT_RECEIPT.json")
            assert receipt["status"] == "COMPLETE" and receipt["completed_updates"] == 2500
            path = directory / "checkpoint_last.pt"
            assert sha(path) == receipt["checkpoint_sha256"]
            model = model_type(arm).eval()
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
            )
            x = extended if study == "FUSION_STATE" else features
            predictions = []
            loss_sum = 0.0
            with torch.inference_mode():
                for start in range(0, len(truth), 128):
                    stop = min(start + 128, len(truth))
                    output = model(x[start:stop], widths[start:stop])
                    predictions.append(output["point_phase"])
                    loss_sum += float(
                        fusion_loss(output, truth[start:stop], mass[start:stop], len(truth))
                    ) * (stop - start)
            point = torch.cat(predictions)
            error = (point - truth).abs()
            rows.append(
                dict(
                    study=study,
                    arm=arm,
                    seed=7,
                    checkpoint_sha256=sha(path),
                    queries=len(truth),
                    mean_abs_phase_error=float(error.mean()),
                    weighted_abs_phase_error=float((mass * error).sum()),
                    native_fusion_loss=loss_sum / len(truth),
                    improved_vs_R=int((error < parent_error).sum()),
                    worsened_vs_R=int((error > parent_error).sum()),
                    wrong_sign=int((point.sign() != truth.sign()).sum()),
                )
            )
    write(
        OUT / "H_RECIPE_DIAGNOSTICS.json",
        dict(
            rows=rows,
            source_sha256=sha(__file__),
            cache_sha256=sha(cache),
            fusion_cache_sha256=sha(fused),
            optimizer_updates=0,
            interpretation=(
                "In-sample H losses; all parents already H-supervised; not OOF or generalization"
            ),
            R_CTX_mean_abs_phase_error=float(parent_error.mean()),
            F_TRUE_mean_abs_phase_error=float((extended[:, -1] - truth).abs().mean()),
        ),
    )
    print(rows, flush=True)


if __name__ == "__main__":
    main()

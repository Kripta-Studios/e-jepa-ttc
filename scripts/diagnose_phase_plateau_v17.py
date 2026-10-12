"""Measure the native decoder's flat point-loss region on H, without optimizer updates."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import emitted_phase, phase_to_ttc


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    owner = root.parent / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    history = root.parent / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"
    out = root / "artifacts/representation_audit_20261012/diagnostics"
    with np.load(owner / "features/H/E_HISTORY.npz", allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    assert set(values["role"].astype(str)) == {"H"}
    target = -0.1 / np.expm1(-values["target_phase"].astype(np.float64))
    summaries = []
    for name in ("initialization", "RT-M0-L0-S7", "RT-M1-L0-S7"):
        torch.manual_seed(7)
        model = TemporalRefiner(TemporalConfig(feature_count=17, hidden=160, backbone="gru"))
        if name != "initialization":
            payload = torch.load(
                history / f"fits/{name}/checkpoint_last.pt", map_location="cpu", weights_only=False
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(target), 128):
                inputs = [
                    torch.from_numpy(values[k][start : start + 128].copy())
                    for k in ("features", "timing", "valid", "expert_phase")
                ]
                predictions.append(model(*inputs)["raw_location"])
        raw = torch.cat(predictions).requires_grad_(True)
        emission = emitted_phase(raw)
        derivative = torch.autograd.grad(emission.sum(), raw)[0]
        pred = phase_to_ttc(raw).detach().numpy()
        flat = derivative.numpy() == 0
        bands = {
            "all": np.ones(len(target), bool),
            "crucial": (target > 0.1) & (target <= 3),
            "small": (target > 3) & (target <= 6),
            "large": (target > 6) & (target <= 10),
            "negative": target < 0,
        }
        for band, mask in bands.items():
            summaries.append(
                {
                    "fit": name,
                    "band": band,
                    "queries": int(mask.sum()),
                    "flat_point_gradient_queries": int(flat[mask].sum()),
                    "flat_fraction": float(flat[mask].mean()),
                    "flat_native_mass": float(values["mass"][mask & flat].sum()),
                    "flat_wrong_sign": int(
                        (flat & mask & (np.sign(pred) != np.sign(target))).sum()
                    ),
                }
            )
    frame = pd.DataFrame(summaries)
    frame.to_csv(out / "PHASE_PLATEAU_H.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

import numpy as np
import torch

from scripts.train_phase_robust import fit


def test_fit_does_not_observe_heldout_targets_or_normalizer_inputs(tmp_path):
    rng = np.random.default_rng(7)
    a = {
        "native": rng.uniform(1, 4, (12, 3)).astype(np.float32),
        "geometry": rng.uniform(1, 4, 12).astype(np.float32),
        "quality": rng.uniform(0, 1, (12, 20)).astype(np.float32),
        "target": rng.uniform(1, 4, 12).astype(np.float32),
        "family": np.repeat(["A", "B", "reserve"], 4),
        "group": np.repeat(["recordA", "recordB", "reserve"], 4),
    }
    mask = a["family"] != "reserve"
    first = fit(a, mask, "quality", 7, tmp_path / "first")
    a["target"][~mask] = 1e6
    a["quality"][~mask] = 1e6
    a["native"][~mask] = -1e6
    second = fit(a, mask, "quality", 7, tmp_path / "second")
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])

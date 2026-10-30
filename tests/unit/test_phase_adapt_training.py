import numpy as np
import torch

from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_training import fit_head, load_head


def test_real_training_updates_weights_and_roundtrips(tmp_path):
    tmp_path = tmp_path / "run"
    torch.manual_seed(7)
    base = PhaseOnlyFinal()
    features = np.zeros((16, 340), dtype=np.float32)
    features[:, 160] = 0.04
    values = {
        "features": features,
        "rgb_width_logits": np.zeros((16, 2), dtype=np.float32),
        "target_phase": np.full(16, 0.05, dtype=np.float32),
        "domain": np.array(["a"] * 8 + ["b"] * 8),
        "group_id": np.array(["a"] * 8 + ["b"] * 8),
    }
    recipe = {
        "seed": 7,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "max_updates": 40,
        "max_epochs": 20,
        "grad_clip": 1.0,
    }
    result = fit_head(
        base.state_dict(), values, recipe, tmp_path, device="cpu", binding={"fixture": True}
    )
    assert result["updates"] == 40
    assert result["epochs"] == 20
    assert result["history"][-1]["loss"] < result["baseline"]["loss"]
    trained = load_head(tmp_path / "checkpoint.pt", result["checkpoint_sha256"])
    assert not torch.equal(trained.output.weight, base.output.weight)
    reloaded = load_head(tmp_path / "checkpoint.pt", result["checkpoint_sha256"])
    with torch.inference_mode():
        expected = trained(torch.tensor(features), torch.zeros(16, 2))["ttc"]
        actual = reloaded(torch.tensor(features), torch.zeros(16, 2))["ttc"]
    assert torch.equal(expected, actual)


def test_load_rejects_tampered_bytes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not the expected checkpoint")
    import pytest

    with pytest.raises(ValueError, match="hash"):
        load_head(path, "0" * 64)

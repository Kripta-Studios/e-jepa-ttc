import numpy as np
import pytest

from operational.phase_adapt_cache import save_npz


def test_recovery_preserves_declared_missing_predictions_and_rejects_changes(tmp_path):
    path = tmp_path / "predictions.npz"
    values = {"prediction": np.array([1.0, np.nan]), "tokens": np.array(["a", "b"])}
    save_npz(path, values)
    original = path.read_bytes()
    save_npz(path, values)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="different"):
        save_npz(path, {**values, "prediction": np.array([2.0, np.nan])})

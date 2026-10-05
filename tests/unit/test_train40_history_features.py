"""H8 must retain the original padded producer dispatch and target-free raw contract."""

import numpy as np
import torch

from operational.train40_system import history_features


def test_worker_retains_full_16_slots(monkeypatch):
    from e_jepa_ttc.efficient_context import mapped_union

    captured = {}

    class Pool:
        def get(self, path):
            return path

    def encode(reader, windows, lags, valid, square, **kwargs):
        captured.update(lags=lags, valid=valid, kwargs=kwargs)
        return torch.zeros(16, 3, 12, 128, 128)

    monkeypatch.setattr(history_features, "_reader_pool", Pool())
    monkeypatch.setattr(mapped_union, "encode_union", encode)
    result = history_features.prepare(
        {
            "path": "train/events.h5",
            "windows": np.asarray([[0, 1], [1, 2], [2, 3]]),
            "valid": np.asarray([False, False, True, True, True, True, True, True]),
            "square": [0.0, 0.0, 2.0, 2.0],
            "sequence": "train",
        }
    )
    assert result.shape == (16, 3, 12, 128, 128)
    assert result.dtype == np.float32
    assert not captured["valid"][:8].any()
    np.testing.assert_array_equal(captured["lags"][-8:], np.arange(7, -1, -1) * 50000)
    assert captured["kwargs"]["event_pixel_diff"] == 5

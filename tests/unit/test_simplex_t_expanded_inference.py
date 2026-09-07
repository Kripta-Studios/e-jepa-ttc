"""Adapter wiring with simulated producers/readers; no GPU or raw dataset access."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.expanded_inference import expanded_inference_family


@pytest.mark.parametrize("failure", ["", "authority", "role", "inactive", "weights"])
@pytest.mark.parametrize("scope", ["expanded", "original_qa"])
def test_adapter_uses_h16_layout_and_explicit_train_role(tmp_path, monkeypatch, failure, scope):
    import e_jepa_ttc.simplex_t.expanded_inference as module

    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"synthetic, never deserialized")
    digest = sha256(checkpoint)
    raw = tmp_path / "train"
    (raw / "seq").mkdir(parents=True)
    (raw / "seq/events.h5").write_bytes(b"synthetic, never decoded")
    families = [
        {"outer_fold": 0, "role": "inner0", "experts": dict.fromkeys(("A5", "C2F", "PAIR"), digest)}
    ]
    history = np.full((1, 16), -1, dtype=np.int64)
    history[0, -2:] = [0, 1]
    index = {
        "tokens": np.array(["q"]),
        "sequences": np.array(["seq"]),
        "producer_family": np.array([[0 if failure != "inactive" else -1], [-1], [-1]]),
        "valid": history >= 0,
        "base_windows_us": np.array([[[0, 100000], [100000, 200000], [200000, 300000]]]),
        "square_xyxy": np.array([[1, 2, 3, 4]]),
        "lag_us": np.arange(15, -1, -1, dtype=np.int64) * 50000,
        "anchor_us": np.array([300000], dtype=np.int64),
        "roi_available_us": np.array([300000], dtype=np.int64),
    }
    loads, reads, forwards = [], [], []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)

    def load(path, **kwargs):
        loads.append(path)
        return object()

    monkeypatch.setattr(module, "load_causal_scale_replay_checkpoint", load)
    monkeypatch.setattr(module, "load_pair_head", load)
    monkeypatch.setattr(module, "EAPEventReader", lambda path: nullcontext(path))

    def encode(reader, windows, lags, mask, roi, **kwargs):
        reads.append(reader)
        assert tuple(roi) == (1, 2, 3, 4)
        assert kwargs["sequence_id"] == "seq"
        return torch.zeros(16, 3, 12, 2, 2)

    def extract(a5, c2f, pair, tensor, delta):
        forwards.append(True)
        assert tensor.shape == (16, 3, 12, 2, 2)
        torch.testing.assert_close(delta, torch.full((16, 2), 0.1))
        return {"features145": np.zeros((16, 145), dtype=np.float32)}

    monkeypatch.setattr(module, "encode_context_union", encode)
    monkeypatch.setattr(module, "extract_family", extract)

    def validate():
        if failure == "authority":
            raise ValueError("missing time authority")

    if failure == "weights":
        checkpoint.write_bytes(b"changed")
    family_id = 0
    if scope == "original_qa":
        family_id = 3
        families = families * 4
        families[3] = dict(families[0], role="outer_dev")
        for key in tuple(index):
            if key == "lag_us":
                continue
            axis = 1 if key == "producer_family" else 0
            index[key] = np.repeat(index[key], 8192, axis=axis)
        index["producer_family"][0] = -1 if failure == "inactive" else family_id
        history = np.repeat(history, 8192, axis=0)
    args = dict(
        families=families,
        checkpoint_paths={digest: checkpoint},
        index=index,
        history=history,
        raw_train_root=raw,
        allowed_sequences=set() if failure == "role" else {"seq"},
        preprocessing={"roi_size": 2, "event_pixel_diff": 1},
        validate_prerequisites=validate,
        producer_scope=scope,
    )
    if failure:
        with pytest.raises(ValueError), expanded_inference_family(family_id, **args) as infer:
            infer(0)
        assert reads == forwards == []
        if failure in {"authority", "weights"}:
            assert loads == []
    else:
        with expanded_inference_family(family_id, **args) as infer:
            result = infer(0)
        assert len(loads) == 3 and len(reads) == 1 and len(forwards) == 1
        assert result["features145"].shape == (2, 145)
        np.testing.assert_array_equal(result["anchor_us"], [250000, 300000])
        np.testing.assert_array_equal(result["observation_ids"], [0, 1])

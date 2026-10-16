"""Inference prefetch must fail closed on corruption, identity or ordering changes."""

import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from operational.joint_measurement import prefetched_sensor as runtime
from operational.range_transfer.common import sha, write


def fixture_pack(tmp_path, *, duplicate=False, clock=0.1):
    path = tmp_path / "pack_00000.zip"
    pixels = np.arange(2 * 12 * 128 * 128, dtype=np.float32).reshape(2, 12, 128, 128)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in (0, 0) if duplicate else (0, 1):
            stream = io.BytesIO()
            np.savez(
                stream,
                events=pixels + index,
                delta=np.array(clock, dtype=np.float32),
                token=np.array(f"query-{index}"),
                unused_teacher=np.array([np.nan]),
            )
            archive.writestr(f"{index:06d}.npz", stream.getvalue())
    write(path.with_suffix(".json"), dict(start=0, stop=2, sha256=sha(path)))
    return path, pixels


def test_exact_pixels_clocks_and_selected_fields(tmp_path):
    path, pixels = fixture_pack(tmp_path)
    result, timing = runtime.decode_pack(path, ["query-0", "query-1"], workers=2)
    assert torch.equal(result.events[0], torch.from_numpy(pixels))
    assert torch.equal(result.events[1], torch.from_numpy(pixels + 1))
    assert result.delta_t_s.tolist() == [float(np.float32(0.1))] * 2
    assert timing["rows"] == 2
    assert not hasattr(result, "target_ttc_s")


def test_corrupt_pack_rejected(tmp_path):
    path, _ = fixture_pack(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        runtime.decode_pack(path, ["query-0", "query-1"])


def test_swapped_native_tokens_rejected(tmp_path):
    path, _ = fixture_pack(tmp_path)
    with pytest.raises(ValueError, match="token mismatch"):
        runtime.decode_pack(path, ["query-1", "query-0"])


def test_duplicate_member_rejected(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate"):
        path, _ = fixture_pack(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        runtime.decode_pack(path, ["query-0", "query-1"])


def test_zero_clock_rejected(tmp_path):
    path, _ = fixture_pack(tmp_path, clock=0)
    with pytest.raises(ValueError, match="clock"):
        runtime.decode_pack(path, ["query-0", "query-1"])


def test_tail_resume_and_invalid_orders(tmp_path, monkeypatch):
    fixture_pack(tmp_path)
    source = SimpleNamespace(
        rows=pd.DataFrame({"sample_token": ["query-0", "query-1"]}),
        folder=tmp_path,
        population=2,
        close=lambda: None,
    )
    monkeypatch.setattr(runtime, "SensorDataset", lambda role: source)
    reader = runtime.PrefetchedSensor("P")
    for indices in ([1, 0], [0, 0], [0], [-1, 0], [0, 1, 2, 3]):
        with pytest.raises(ValueError, match="original consecutive"):
            reader.batch(indices)
    assert reader.batch([0, 1]).events.shape == (2, 2, 12, 128, 128)
    assert len(reader.timings) == 1
    reader.close()
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.batch([0, 1])

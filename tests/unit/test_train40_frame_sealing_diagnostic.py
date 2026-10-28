"""The CPU diagnostic preserves the original identity formula and input strides."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from operational.train40_runtime.multiobject import RawRGBHistory
from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame
from scripts.diagnose_train40_frame_sealing import compare_pair, make_pair, split_sealing


@pytest.mark.parametrize("strided", [False, True])
def test_split_stages_match_factory_identity_and_immutable_pixels(strided):
    original = np.arange(11 * 17 * 3, dtype=np.uint8).reshape(11, 17, 3)
    frame = original[::2, ::2] if strided else original
    before, strides = frame.copy(), frame.strides
    source = RGBSourceIdentity("fixture", "a" * 64)
    token = freeze_rgb_frame(frame, 7, 1000, source)
    value = split_sealing(frame, 7, 1000, source)
    assert value.payload == token.payload
    assert value.checksum == token.content_sha256
    assert value.identity == token.identity_sha256
    assert value.total_ns == value.copy_ns + value.payload_sha256_ns + value.descriptor_ns
    assert min(value.copy_ns, value.payload_sha256_ns, value.descriptor_ns) >= 0
    assert frame.strides == strides
    np.testing.assert_array_equal(frame, before)
    frame[:] = 0
    np.testing.assert_array_equal(token.array, before)
    np.testing.assert_array_equal(
        np.frombuffer(value.payload, np.uint8).reshape(value.shape), before
    )


def test_both_pair_orders_cover_every_frame_and_reject_corrupted_descriptor():
    frames = {index: np.full((3, 5, 3), index, np.uint8) for index in (3, 5, 7)}
    raw = RawRGBHistory(900, frames, {3: 100, 5: 500, 7: 800}, [(3, 5, 7)])
    source = RGBSourceIdentity("fixture", "a" * 64)
    for order in (["original", "split"], ["split", "original"]):
        receipt = make_pair(raw, source, order=order)
        assert receipt["passed"] and receipt["order"] == order
        assert receipt["frames"] == 3 and receipt["bytes"] == 3 * 3 * 5 * 3
        assert receipt["no_digest_reused_between_routes"]
        assert receipt["split_wrapper_and_instrumentation_ms"] >= 0
    tokens = [
        freeze_rgb_frame(frames[index], index, raw.timestamps_us[index], source) for index in frames
    ]
    values = [
        split_sealing(frames[index], index, raw.timestamps_us[index], source) for index in frames
    ]
    values[0] = replace(values[0], identity="0" * 64)
    checks = compare_pair(raw, tokens, values)
    assert not checks[0]["descriptor_sha256_exact"]
    assert all(check["payload_sha256_exact"] for check in checks)
    with pytest.raises(ValueError, match="every original frame"):
        compare_pair(raw, tokens[:-1], values)

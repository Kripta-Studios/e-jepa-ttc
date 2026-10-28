"""Optional private-wheel correctness, thread contract and unchanged pixel checks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from operational.train40_runtime.multiobject import RawRGBHistory
from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame
from scripts.diagnose_train40_frame_hashes import (
    ROOT,
    blake3_incremental_reference,
    digest,
    load_dependency,
    measure_route,
)


@pytest.fixture(scope="module")
def private_blake3():
    directory = ROOT / "artifacts/train40_runtime_council_20261028/hash_probe_dependencies"
    if not (directory / "download_receipt.json").is_file():
        pytest.skip("Optional private BLAKE3 diagnostic wheel is not installed")
    module, receipt, files = load_dependency(directory)
    assert receipt["max_threads"] == 1 and receipt["digest_bytes"] == 32
    assert files and all(path.is_file() for path in files)
    return module


def test_official_empty_vector_and_explicit_single_thread(private_blake3):
    # BLAKE3-team/BLAKE3 test_vectors/test_vectors.json, first 32 bytes, input_len=0.
    expected = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return private_blake3.blake3(*args, **kwargs)

    wrapped = SimpleNamespace(blake3=observed)
    assert digest(b"", "blake3", wrapped) == expected
    assert blake3_incremental_reference(b"", wrapped) == expected
    assert calls == [{"max_threads": 1}, {"max_threads": 1}]
    with pytest.raises(ValueError, match="sha256 or blake3"):
        digest(b"", "unsupported", wrapped)


@pytest.mark.parametrize("copy_inside", [False, True])
def test_both_hashes_preserve_strided_pixels_and_original_sha(private_blake3, copy_inside):
    full = np.arange(11 * 17 * 3, dtype=np.uint8).reshape(11, 17, 3)
    frame = full[::2, ::2]
    raw = RawRGBHistory(1000, {3: frame}, {3: 900}, [(3,)])
    source = RGBSourceIdentity("fixture", "b" * 64)
    token = freeze_rgb_frame(frame, 3, 900, source)
    originals = {3: token}
    expected_b3 = {3: blake3_incremental_reference(token.payload, private_blake3)}
    results = []
    for algorithm in ("sha256", "blake3"):
        result = measure_route(
            raw,
            originals,
            expected_b3,
            private_blake3,
            algorithm=algorithm,
            copy_inside=copy_inside,
        )
        assert (
            result["passed"]
            and result["frame_checks"][0]["original_sha256"] == token.content_sha256
        )
        assert result["frame_checks"][0]["strides"] == list(frame.strides)
        assert result["copy_ms"] >= 0 if copy_inside else result["copy_ms"] == 0
        results.append(result["frame_checks"][0]["digest"])
    assert results[0] != results[1]  # Different algorithms are separately namespaced.

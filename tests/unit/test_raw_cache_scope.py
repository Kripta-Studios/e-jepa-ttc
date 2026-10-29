import os

import pytest

from scripts.runpod_raw_cache_scope_20261027 import advise_raw, identity


def test_advice_preserves_input_and_uses_read_only_descriptor(tmp_path):
    path = tmp_path / "events.h5"
    path.write_bytes(b"verified input")
    item = {"path": str(path), "stat": identity(path.stat())}
    calls = []

    def advice(fd, offset, length, kind):
        calls.append((offset, length))
        assert os.read(fd, 14) == b"verified input"
        with pytest.raises(OSError):
            os.write(fd, b"x")

    advise_raw(item, advice)
    assert calls == [(0, 0)] and path.read_bytes() == b"verified input"


def test_changed_input_rejected_before_advice(tmp_path):
    path = tmp_path / "events.h5"
    path.write_bytes(b"original")
    item = {"path": str(path), "stat": identity(path.stat())}
    path.write_bytes(b"changed length")
    with pytest.raises(ValueError, match="identity changed"):
        advise_raw(item, lambda *args: pytest.fail("Must not advise modified file"))

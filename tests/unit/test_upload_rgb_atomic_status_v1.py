from pathlib import Path

import pytest

from scripts.upload_rgb_atomic_status_v1 import DELAYS, retry_writer


def test_sharing_conflict_retries_same_receipt() -> None:
    calls = []
    sleeps = []

    def writer(path, value):
        calls.append((path, value))
        if len(calls) == 1:
            error = PermissionError("sharing")
            error.winerror = 5
            raise error

    value = {"verified_bytes": 10}
    retry_writer(writer, sleeps.append)(Path("status.json"), value)
    assert calls == [(Path("status.json"), value)] * 2
    assert sleeps == [DELAYS[0]]


def test_persistent_conflict_fails_after_bounded_retries() -> None:
    sleeps = []

    def writer(*_args):
        error = PermissionError("persistent")
        error.winerror = 32
        raise error

    with pytest.raises(PermissionError):
        retry_writer(writer, sleeps.append)(Path("status.json"), {})
    assert sleeps == list(DELAYS)


def test_other_permission_error_is_not_retried() -> None:
    sleeps = []

    def writer(*_args):
        raise PermissionError("not Windows sharing")

    with pytest.raises(PermissionError):
        retry_writer(writer, sleeps.append)(Path("status.json"), {})
    assert sleeps == []

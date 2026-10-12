"""The fast process scan must retain guard inputs and restore the native iterator."""

from types import SimpleNamespace
from unittest.mock import patch

import psutil
import pytest

from operational.multimodal_research import guard_scan


def test_python_metadata_preserved_and_other_metadata_not_read() -> None:
    queried = []

    def info():
        queried.append("python")
        return {"name": "Python.exe", "cmdline": ["python", "-m", "operational.example"], "ppid": 7}

    python = SimpleNamespace(pid=9, info={"name": "Python.exe"}, as_dict=lambda **kwargs: info())
    other = SimpleNamespace(pid=10, info={"name": "browser.exe"})
    with patch.object(guard_scan, "ORIGINAL_SCAN", lambda *args: iter([python, other])):
        result = list(guard_scan.python_first(["name", "cmdline", "ppid"]))
    assert result == [python] and queried == ["python"]
    assert python.info["ppid"] == 7


def test_unsupported_attribute_set_is_forwarded_without_filtering() -> None:
    sentinel = object()
    with patch.object(guard_scan, "ORIGINAL_SCAN", lambda *args: iter([sentinel])):
        assert list(guard_scan.python_first(["pid"])) == [sentinel]


def test_native_guard_failure_propagates_and_patch_is_restored() -> None:
    original = psutil.process_iter

    def fail(*, prospective_updates: int) -> None:
        assert prospective_updates == 25 and psutil.process_iter is guard_scan.python_first
        raise MemoryError("resource floor")

    with pytest.raises(MemoryError, match="resource floor"):
        guard_scan.wrap(fail)(prospective_updates=25)
    assert psutil.process_iter is original

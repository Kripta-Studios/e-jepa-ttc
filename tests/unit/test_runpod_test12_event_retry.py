"""Exercise resource retry decisions without importing Linux-only remote tooling."""

from __future__ import annotations

import argparse
import ast
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/runpod_test12_event_worker.py"


def load_function(name: str, namespace: dict) -> object:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "error,retries",
    [
        (RuntimeError("RESOURCE_WAIT: memory"), True),
        (RuntimeError("Execution deadline or manual stop"), False),
        (ValueError("Exact frozen tensor parity failed"), False),
    ],
)
def test_only_transient_resources_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    retries: bool,
) -> None:
    monkeypatch.setattr("sys.argv", ["worker"])
    calls, states, waits = [], [], []

    def run(_args: object) -> None:
        calls.append(True)
        if len(calls) == 1:
            raise error

    namespace = {
        "argparse": argparse,
        "__doc__": "test",
        "ROOT": tmp_path,
        "fcntl": SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_args: None),
        "run": run,
        "status": lambda **kwargs: states.append(kwargs),
        "wait_resources": waits.append,
    }
    main = load_function("main", namespace)
    if retries:
        main()
        assert len(calls) == 2
        assert waits == ["resume_after_resource_wait"]
        assert states[0]["status"] == "RETRYING_RESOURCE_WAIT"
    else:
        with pytest.raises(type(error), match=str(error)):
            main()
        assert len(calls) == 1
        assert not waits
        assert states[-1]["status"] == "STOPPED_REQUIRES_REVIEW"


def test_interrupted_download_releases_reservation_and_keeps_partial(tmp_path: Path) -> None:
    partial = tmp_path / "raw" / ".cache" / "fragment.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"verified partial remains")
    reservations, signals = [], []
    child = SimpleNamespace(pid=73, returncode=None)
    child.poll = lambda: child.returncode
    child.wait = lambda **_kwargs: setattr(child, "returncode", -15)

    def guard() -> None:
        raise RuntimeError("RESOURCE_WAIT: memory")

    namespace = {
        "PROBE": tmp_path,
        "Path": Path,
        "MEDIA_REVISION": "pinned",
        "wait_resources": lambda _sequence: {},
        "resource_guard": guard,
        "reserve_download": lambda *args: reservations.append(args),
        "status": lambda **_kwargs: None,
        "os": SimpleNamespace(environ={}, killpg=lambda *args: signals.append(args)),
        "signal": signal,
        "subprocess": SimpleNamespace(
            Popen=lambda *_args, **_kwargs: child, TimeoutExpired=TimeoutError
        ),
    }
    download = load_function("download", namespace)
    with pytest.raises(RuntimeError, match="RESOURCE_WAIT"):
        download("sequence", {"path": "data/test/sequence/events.h5", "bytes": 123})
    assert reservations == [("event_h5_sequence", 123), ("event_h5_sequence", 0)]
    assert signals == [(73, signal.SIGTERM)]
    assert partial.read_bytes() == b"verified partial remains"

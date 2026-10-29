"""Workspace quota accounting around RGB uploader V2 is narrow and fail closed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.upload_rgb_sparse_frames_v4 import (
    ATOMIC_STATUS_SHA256,
    V2_SHA256,
    digest,
    remote_inventory,
)


def _stat(free: int = 2 * 1024**5):
    return SimpleNamespace(
        stdout=json.dumps(
            {
                "root": "/workspace/ejepa_rgb_frames",
                "quota_root": "/workspace",
                "root_dev": 7,
                "quota_dev": 7,
                "statvfs_free": free,
            }
        )
    )


def test_frozen_dependencies_are_bound_by_hash() -> None:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    assert digest(scripts / "upload_rgb_sparse_frames_v2.py") == V2_SHA256
    assert digest(scripts / "upload_rgb_atomic_status_v1.py") == ATOMIC_STATUS_SHA256


def test_quota_caps_fake_two_pib_statvfs() -> None:
    replies = iter([_stat(), SimpleNamespace(stdout="18900000000\t/workspace\n")])
    free, report = remote_inventory(
        ["ssh"],
        "/workspace/ejepa_rgb_frames",
        "/workspace",
        50_000_000_000,
        runner=lambda *_args, **_kwargs: next(replies),
    )
    assert free == 31_100_000_000
    assert report["available_bytes"] == min(report["statvfs_free_bytes"], 31_100_000_000)


def test_gnu_du_enoent_is_retried_bounded_then_succeeds() -> None:
    race = subprocess.CalledProcessError(
        1, "du", stderr="du: cannot access '/workspace/gone': No such file or directory\n"
    )
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stat(40_000_000_000)
        if calls in {2, 3}:
            raise race
        return SimpleNamespace(stdout="10000000000\t/workspace\n")

    delays: list[float] = []
    free, report = remote_inventory(
        ["ssh"],
        "/workspace/ejepa_rgb_frames",
        "/workspace",
        50_000_000_000,
        runner=runner,
        delays=(0.1, 0.2),
        sleeper=delays.append,
    )
    assert free == 40_000_000_000
    assert report["du_enoent_retries"] == 2
    assert delays == [0.1, 0.2]


def test_permission_error_is_never_ignored() -> None:
    denied = subprocess.CalledProcessError(
        1, "du", stderr="du: cannot read directory '/workspace/private': Permission denied\n"
    )
    replies = iter([_stat(), denied])

    def runner(*_args, **_kwargs):
        value = next(replies)
        if isinstance(value, BaseException):
            raise value
        return value

    with pytest.raises(subprocess.CalledProcessError):
        remote_inventory(
            ["ssh"],
            "/workspace/ejepa_rgb_frames",
            "/workspace",
            50_000_000_000,
            runner=runner,
        )


def test_other_filesystem_or_root_fails_closed() -> None:
    bad = _stat()
    payload = json.loads(bad.stdout)
    payload["root_dev"] = 8
    bad.stdout = json.dumps(payload)
    with pytest.raises(ValueError, match="another filesystem"):
        remote_inventory(
            ["ssh"],
            "/workspace/ejepa_rgb_frames",
            "/workspace",
            50_000_000_000,
            runner=lambda *_args, **_kwargs: bad,
        )


def test_usage_above_authorized_quota_fails_closed() -> None:
    replies = iter([_stat(2 * 1024**5), SimpleNamespace(stdout="50000000001\t/workspace\n")])
    with pytest.raises(ValueError, match="outside authorization"):
        remote_inventory(
            ["ssh"],
            "/workspace/ejepa_rgb_frames",
            "/workspace",
            50_000_000_000,
            runner=lambda *_args, **_kwargs: next(replies),
        )

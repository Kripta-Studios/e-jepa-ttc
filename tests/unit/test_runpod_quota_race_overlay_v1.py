from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.runpod_quota_race_overlay_v1 import (
    RetryingStorageInventory,
    transient_workspace_enoent,
)


def failure(stderr: str, *, returncode: int = 1) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        returncode,
        ["du", "-sb", "/workspace"],
        output="123\t/workspace\n",
        stderr=stderr,
    )


def test_only_workspace_du_enoent_is_retryable() -> None:
    assert transient_workspace_enoent(
        failure(
            "du: cannot access '/workspace/run/cloud_snapshots/.old': No such file or directory\n"
        )
    )
    assert not transient_workspace_enoent(
        failure("du: cannot access '/etc/secret': No such file or directory\n")
    )
    assert not transient_workspace_enoent(
        failure("du: cannot read directory '/workspace/x': Permission denied\n")
    )
    assert not transient_workspace_enoent(failure("unclassified failure\n"))
    assert not transient_workspace_enoent(
        failure(
            "du: cannot access '/workspace/gone': No such file or directory\n"
            "du: cannot read directory '/workspace/x': Permission denied\n"
        )
    )
    assert not transient_workspace_enoent(
        failure("du: cannot access '/workspace/gone': No such file or directory\n", returncode=2)
    )


def test_transient_races_retry_then_publish_fresh_inventory() -> None:
    attempts = 0
    sleeps: list[float] = []

    def original(_root: Path) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise failure(
                f"du: cannot access '/workspace/cloud_snapshots/gone{attempts}': "
                "No such file or directory\n"
            )
        return 20_000_000_000, {"workspace_logical_bytes": 30_000_000_000}

    inventory = RetryingStorageInventory(original, delays=(0.1, 0.2), sleeper=sleeps.append)
    free, report = inventory(Path("/workspace/run"))

    assert free == 20_000_000_000
    assert attempts == 3
    assert sleeps == [0.1, 0.2]
    assert report["workspace_inventory_enoent_retries"] == 2
    assert report["workspace_inventory_stale_fallback_used"] is False


def test_non_enoent_failure_is_not_retried() -> None:
    attempts = 0

    def original(_root: Path) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        raise failure("du: cannot read directory '/workspace/x': Permission denied\n")

    inventory = RetryingStorageInventory(original, sleeper=lambda _delay: None)
    with pytest.raises(subprocess.CalledProcessError):
        inventory(Path("/workspace/run"))
    assert attempts == 1


def test_exhausted_enoent_retries_fail_closed_without_stale_value() -> None:
    attempts = 0

    def original(_root: Path) -> tuple[int, dict]:
        nonlocal attempts
        attempts += 1
        raise failure("du: cannot access '/workspace/gone': No such file or directory\n")

    inventory = RetryingStorageInventory(original, delays=(0.0, 0.0), sleeper=lambda _delay: None)
    with pytest.raises(RuntimeError, match="bounded ENOENT retries"):
        inventory(Path("/workspace/run"))
    assert attempts == 3

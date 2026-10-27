"""Resource authority tests for exact-sequence RGB staging."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "rgb_stage", Path(__file__).parents[2] / "scripts/runpod_test12_rgb_worker_v1.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot():
    return {
        "schema": "ejepa_event_volume_snapshot_v1",
        "run_root": "/workspace/ejepa_event_runs",
        "volume_root": "/workspace",
        "quota_bytes": 600_000_000_000,
        "used_bytes": 545_000_000_000,
        "observed_timestamp": 1000,
    }


def test_shared_reservation_and_fixed_reserve():
    assert module.quota_available(snapshot(), 1050, 3_000_000_000, 22_884_901_888) == 29_115_098_112


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("quota_bytes", 900_000_000_000),
        ("run_root", "/elsewhere"),
        ("volume_root", "/opt"),
        ("observed_timestamp", 0),
        ("observed_timestamp", 2000),
    ],
)
def test_invalid_authority_fails_closed(field, value):
    sample = snapshot()
    sample[field] = value
    with pytest.raises(ValueError):
        module.quota_available(sample, 1050, 0, 10)

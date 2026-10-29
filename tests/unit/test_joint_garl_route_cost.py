"""Guard paired latency evidence against population and numerical substitution."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement import garl_route_cost as cost
from operational.range_transfer.common import sha, write


def test_selection_is_group_balanced_and_ignores_frame_order() -> None:
    frame = pd.DataFrame(
        {"group_id": ["b", "a", "a", "b"], "sample_token": ["b2", "a2", "a1", "b1"]}
    )
    expected = ["a1", "b1", "a2"]
    assert [r["sample_token"] for r in cost.select_rows(frame, 3)] == expected
    assert [r["sample_token"] for r in cost.select_rows(frame.iloc[::-1], 3)] == expected


@pytest.mark.parametrize(
    "tokens,groups,count",
    [
        (["q", "q"], ["a", "b"], 2),
        (["q", None], ["a", "b"], 2),
        (["q", "r"], ["a", None], 2),
        (["q", "r"], ["a", "b"], 3),
    ],
)
def test_selection_rejects_ambiguous_populations(tokens, groups, count) -> None:
    with pytest.raises(ValueError):
        cost.select_rows(pd.DataFrame({"sample_token": tokens, "group_id": groups}), count)


def test_every_query_gets_both_measurement_orders() -> None:
    for index in range(64):
        assert cost.route_order(index, 0) == cost.route_order(index, 1)[::-1]


def test_sensor_parity_is_strict() -> None:
    sensor = np.zeros((46, 128, 128), np.float32)
    cost.ensure_pair(sensor, sensor.copy())
    changed = sensor.copy()
    changed[0, 0, 0] = 1e-10
    with pytest.raises(ValueError, match="parity"):
        cost.ensure_pair(sensor, changed)
    with pytest.raises(ValueError, match="precision"):
        cost.ensure_pair(sensor, sensor.astype(np.float64))
    sensor[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="parity"):
        cost.ensure_pair(sensor, sensor)


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_costs_are_not_dropped(values) -> None:
    with pytest.raises(ValueError):
        cost.summarize(values)


def test_statistics_use_total_service_time() -> None:
    result = cost.summarize([10.0, 30.0])
    assert result["mean_ms"] == result["p50_ms"] == 20.0
    assert result["queries_per_second"] == 50.0


def fixture_pairs(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(cost, "OUT", tmp_path)
    write(
        tmp_path / "REGISTRATION.json",
        {"passes": 1, "queries": [{"sample_token": "q"}], "cost_scope": "synthetic test"},
    )
    values = {
        "sensor_sha256": "a" * 64,
        "heights_ttc": [1.0, 1.0, "inf"],
        "preprocess_ms": 10.0,
        "model_emission_ms": 20.0,
        "total_ms": 30.0,
        "rss_bytes": 42,
        "lifetime_peak_rss_bytes": 60,
    }
    path = tmp_path / "pairs/0_000.json"
    write(
        path,
        {
            "status": "PASS",
            "registration_sha256": sha(tmp_path / "REGISTRATION.json"),
            "sample_token": "q",
            "group_id": "g",
            "repeat": 0,
            "index": 0,
            "session": "s",
            "order": list(cost.ROUTES),
            "routes": {route: dict(values) for route in cost.ROUTES},
        },
    )
    return path


def test_complete_cost_report_keeps_nonfinite_ttc(tmp_path: Path, monkeypatch) -> None:
    fixture_pairs(tmp_path, monkeypatch)
    result = cost.report()
    assert result["route_observations"] == 2
    assert result["summary"]["UPSTREAM"]["total_ms"]["mean_ms"] == 30.0


@pytest.mark.parametrize("mutation", ["query", "sensor", "prediction", "extra", "missing"])
def test_report_rejects_incompatible_evidence(tmp_path: Path, monkeypatch, mutation) -> None:
    path = fixture_pairs(tmp_path, monkeypatch)
    record = cost.read(path)
    if mutation == "query":
        record["sample_token"] = "wrong"
    elif mutation == "sensor":
        record["routes"]["LOCAL"]["sensor_sha256"] = "b" * 64
    elif mutation == "prediction":
        record["routes"]["LOCAL"]["heights_ttc"] = [1.0, 2.0, 0.2]
    elif mutation == "extra":
        write(path.with_name("extra.json"), record)
    write(path, record)
    if mutation == "missing":
        path.unlink()
    with pytest.raises(ValueError):
        cost.report()


def test_live_training_is_rejected_before_resource_writes(monkeypatch) -> None:
    process = SimpleNamespace(
        pid=987654321,
        info={
            "name": "python.exe",
            "cmdline": ["operational.joint_measurement.correspondence_isolated"],
        },
        cwd=lambda: str(cost.ROOT),
    )
    monkeypatch.setattr(cost.psutil, "process_iter", lambda *_: [process])
    monkeypatch.setattr(
        cost, "require_resources", lambda **_: pytest.fail("Touched live resources")
    )
    with pytest.raises(RuntimeError, match="compute still live"):
        cost.guard_idle()

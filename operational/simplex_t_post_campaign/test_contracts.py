"""No-optimizer tests of timer scope, recovery integrity and route exclusion."""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import pytest

from operational.simplex_t_post_campaign.contracts import (
    MODELS,
    checked,
    digest,
    fragment,
    order,
    read,
    route_admission,
    save,
    segmented_route,
    timed,
)


def test_fragment_resume_never_repeats_confirmed_work(tmp_path: Path) -> None:
    calls = []

    def produce() -> list[float]:
        calls.append(1)
        return [float(i + 1) for i in range(25)]

    binding = {"endpoint": "abc", "block": 1}
    first, reused = fragment(tmp_path / "part.json", binding, produce)
    resumed, reused_again = fragment(tmp_path / "part.json", binding, produce)
    assert first == resumed and not reused and reused_again and calls == [1]


def test_interrupted_fragment_has_no_committed_partial(tmp_path: Path) -> None:
    def fail() -> list[float]:
        raise InterruptedError("resource floor")

    path = tmp_path / "part.json"
    with pytest.raises(InterruptedError):
        fragment(path, {"source": 1}, fail)
    assert not path.exists()
    result, reused = fragment(path, {"source": 1}, lambda: [2.0] * 25)
    assert not reused and len(result["raw_ms"]) == 25


@pytest.mark.parametrize("values", [[1.0] * 24, [float("nan")] * 25, [0.0] * 25, [-1.0] * 25])
def test_invalid_measurements_not_published(tmp_path: Path, values: list[float]) -> None:
    path = tmp_path / "part.json"
    with pytest.raises(ValueError):
        fragment(path, {"source": 1}, lambda: values)
    assert not path.exists()


def test_changed_endpoint_or_query_binding_blocks_resume(tmp_path: Path) -> None:
    path = tmp_path / "part.json"
    fragment(path, {"source": 1}, lambda: [1.0] * 25)
    with pytest.raises(ValueError):
        fragment(path, {"source": 2}, lambda: [1.0] * 25)


def test_timer_measures_only_callable() -> None:
    events = []
    times = iter([100, 2000100])

    def clock() -> int:
        events.append("clock")
        return next(times)

    def call() -> str:
        events.append("head_and_emission")
        return "output"

    value, ms = timed(call, clock)
    assert value == "output" and ms == 2.0
    assert events == ["clock", "head_and_emission", "clock"]


def test_balanced_prespecified_order() -> None:
    counts = {label: Counter() for label in MODELS}
    for block in range(3):
        for frag in range(20):
            assert set(order(block, frag)) == set(MODELS)
            for position, label in enumerate(order(block, frag)):
                counts[label][position] += 1
    assert all(max(c.values()) - min(c.values()) <= 1 for c in counts.values())


def test_idle_gpu_does_not_grant_raw_access() -> None:
    a = route_admission({"resources": {"exclusive_gpu_and_heavy_io_slot": None}})
    assert a["status"].startswith("BLOCKED") and not a["raw_payloads_read"]


def test_unadmitted_route_never_calls_sources() -> None:
    def forbidden(*args: object) -> None:
        pytest.fail("raw or producer called without lease")

    with pytest.raises(PermissionError):
        segmented_route(
            admitted=False,
            supplied_roi=None,
            events=forbidden,
            context=forbidden,
            experts=forbidden,
            features=forbidden,
            head=forbidden,
            synchronize=forbidden,
        )


def test_segmented_interface_preserves_roi_and_synchronization() -> None:
    events = []
    clock = itertools.count(0, 1000000).__next__

    def context(value: object, roi: object) -> int:
        assert value == 1 and roi == "external_roi"
        return 2

    result, times = segmented_route(
        admitted=True,
        supplied_roi="external_roi",
        events=lambda: 1,
        context=context,
        experts=lambda x: x + 1,
        features=lambda x: x + 1,
        head=lambda x: x + 1,
        synchronize=lambda: events.append("sync"),
        clock=clock,
    )
    assert result == 5 and len(events) == 11
    assert set(times) == {
        "raw_slice",
        "roi_context",
        "experts_and_transfers",
        "features",
        "head_and_emission",
        "total_conditioned_on_supplied_roi",
    }
    assert all(value == 1 for key, value in times.items() if not key.startswith("total"))


def test_sha_mismatch_preserves_evidence(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    save(path, {"new_optimizer_updates": 0})
    original = digest(path)
    assert checked(path, original) == path
    with pytest.raises(ValueError):
        checked(path, "0" * 64)
    assert read(path) == {"new_optimizer_updates": 0}

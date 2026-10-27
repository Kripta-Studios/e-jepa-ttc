"""Power protection must release on every non-working or battery condition."""

import pytest

from operational.oof_awake import (
    CONTINUOUS,
    DISPLAY_REQUIRED,
    SYSTEM_REQUIRED,
    AwakeRequest,
    should_hold,
)


def test_request_acquire_release_is_idempotent():
    calls = []
    request = AwakeRequest(lambda flags: calls.append(flags) or CONTINUOUS)
    request.update(True)
    request.update(True)
    request.update(False)
    request.update(False)
    assert calls == [CONTINUOUS | SYSTEM_REQUIRED, CONTINUOUS]


def test_display_is_explicit():
    calls = []
    request = AwakeRequest(lambda flags: calls.append(flags) or CONTINUOUS, display=True)
    request.update(True)
    assert calls == [CONTINUOUS | SYSTEM_REQUIRED | DISPLAY_REQUIRED]


def test_failed_request_is_not_claimed_active():
    request = AwakeRequest(lambda _: 0)
    with pytest.raises(OSError):
        request.update(True)
    assert not request.active


@pytest.mark.parametrize("status", ["MANUAL_STOP", "COMPLETE", "FAILED", "WAITING_RESOURCES"])
def test_nonworking_queue_releases(status):
    assert not should_hold(queue_status=status, on_ac=True, stopped=False, alive=True)


def test_battery_stop_and_dead_queue_release():
    assert should_hold(queue_status="RUNNING", on_ac=True, stopped=False, alive=True)
    for on_ac, stopped, alive in [(False, False, True), (True, True, True), (True, False, False)]:
        assert not should_hold(queue_status="RUNNING", on_ac=on_ac, stopped=stopped, alive=alive)

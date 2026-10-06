"""Hourly recovery must never duplicate GPU work or override an explicit pause."""

from datetime import UTC, datetime
from pathlib import Path

from operational.train40_system.durable_io import atomic_json
from operational.train40_system.hourly_supervisor import CONTROLLER, FOLLOWUP, next_hour, review


def configuration(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    atomic_json(output / "AUTHORIZATION.json", {"deadline_utc": "2026-10-09T18:00:00+00:00"})
    return {
        "root": str(tmp_path),
        "output": str(output),
        "bound_files": [],
        "required_paths": {"controller": [str(tmp_path)], "followup": [str(tmp_path)]},
        "commands": {
            "controller": ["python", "-m", CONTROLLER],
            "followup": ["python", "-m", FOLLOWUP],
        },
    }


def run(config, processes=(), state="DEAD", errors=()):
    actions = []

    def start(config, command, tag):
        actions.append(tag)
        return {"pid": 999}

    result = review(
        config,
        now=datetime(2026, 10, 7, 1, tzinfo=UTC),
        inspect=lambda *_: (list(processes), list(errors)),
        owner=lambda _: state,
        start=start,
    )
    return result, actions


def test_live_supervisors_are_adopted(tmp_path):
    result, actions = run(configuration(tmp_path), [{"module": CONTROLLER}, {"module": FOLLOWUP}])
    assert actions == []
    assert result["status"] == "HEALTHY"


def test_missing_supervisors_restart_existing_queue_only(tmp_path):
    result, actions = run(
        configuration(tmp_path), [{"module": "operational.train40_system.engine"}]
    )
    assert actions == ["controller", "followup"]
    assert result["optimizer_updates"] == 0


def test_live_lease_blocks_duplicate_even_if_inventory_misses_it(tmp_path):
    assert run(configuration(tmp_path), state="LIVE")[1] == []


def test_pause_and_deadline_never_relaunch(tmp_path):
    config = configuration(tmp_path)
    pause = Path(config["output"]) / "STOP_REQUEST"
    pause.touch()
    assert run(config)[0]["status"] == "PAUSED_BY_REQUEST"
    assert run(config)[1] == []
    pause.unlink()
    atomic_json(
        Path(config["output"]) / "AUTHORIZATION.json", {"deadline_utc": "2026-10-06T00:00:00+00:00"}
    )
    assert run(config)[0]["status"] == "DEADLINE_STOP_PRESERVED"
    assert run(config)[1] == []


def test_completed_queue_only_needs_followup(tmp_path):
    config = configuration(tmp_path)
    atomic_json(Path(config["output"]) / "TRAINING_QUEUE_COMPLETION.json", {"status": "COMPLETE"})
    assert run(config)[1] == ["followup"]
    atomic_json(Path(config["output"]) / "BUNDLE_VERIFICATION.json", {"status": "PASSED"})
    assert run(config)[1] == []


def test_unknown_process_or_owner_fails_closed(tmp_path):
    config = configuration(tmp_path)
    assert run(config, errors=["AccessDenied"])[1] == []
    assert run(config, state="UNKNOWN")[1] == []


def test_missing_disk_preserves_dependency(tmp_path):
    config = configuration(tmp_path)
    config["required_paths"]["controller"] = [str(tmp_path / "unmounted")]
    result, actions = run(config)
    assert actions == ["followup"]
    assert result["missing_paths"] == [str(tmp_path / "unmounted")]


def test_timer_aligns_to_hour(tmp_path):
    now = datetime(2026, 10, 7, 0, 59, 58, tzinfo=UTC)
    assert next_hour(now) == datetime(2026, 10, 7, 1, tzinfo=UTC)

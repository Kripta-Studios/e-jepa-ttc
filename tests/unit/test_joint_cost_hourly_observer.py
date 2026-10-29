"""Keep amended cost monitoring separate from preserved historical parity failures."""

from types import SimpleNamespace

import pytest

from operational.joint_measurement import hourly_observe as observer


@pytest.mark.parametrize("job", sorted(observer.COST_JOBS))
def test_cost_observer_uses_its_own_rows_and_failure(tmp_path, monkeypatch, job):
    monkeypatch.setattr(observer, "RUN", tmp_path)
    monkeypatch.setattr(
        observer.psutil,
        "Process",
        lambda pid: SimpleNamespace(
            create_time=lambda: 123.0,
            is_running=lambda: True,
            memory_info=lambda: SimpleNamespace(rss=100),
        ),
    )
    monkeypatch.setattr(
        observer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="idle", stderr=""),
    )
    (tmp_path / "GPU_BUDGET.json").write_text('{"entries": [], "cap_s": 216000}')
    other = next(name for name in observer.COST_JOBS if name != job)
    other_dir = tmp_path / observer.JOBS[other][0]
    other_dir.mkdir(parents=True)
    (other_dir / "PARITY_FAILURE.json").write_text("{}")
    directory = tmp_path / observer.JOBS[job][0]
    directory.mkdir(parents=True)
    (directory / "row_000.json").write_text("{}")
    result = observer.observe(42, 123.0, None, job=job)
    assert result["status"] == "RUNNING"
    assert result["progress_signature"]["cost_rows"] == 1
    assert not result["progress_signature"]["parity_failure"]

    (directory / "PARITY_FAILURE.json").write_text("{}")
    result = observer.observe(42, 123.0, None, job=job)
    assert result["status"] == "ACTION_REQUIRED_PARITY_FAILURE"
    assert result["training_actions"] == []

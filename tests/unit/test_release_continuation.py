"""A long-running continuation must stop at scientific gates, not launch unconditional jobs."""

import json
from pathlib import Path

import pytest

from scripts import continue_release_research as continuation


@pytest.mark.parametrize(
    ("population_passes", "profile_passes", "expected_stages", "outcome"),
    [
        (False, True, 2, "cadence_not_admitted"),
        (True, False, 4, "native_cuda_profile_parity_failed_candidate_not_run"),
        (True, True, 6, "native_prediction_comparison_complete_accuracy_pending"),
    ],
)
def test_continuation_routes_full_pipeline_without_crossing_failed_gate(
    tmp_path, monkeypatch, population_passes, profile_passes, expected_stages, outcome
):
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "FREEZE.json").write_text(
        json.dumps({"schema": "phase_canonical_population2608_v1", "source_hashes": {}})
    )
    (screen / "COMPLETE.json").write_text("{}")
    public = tmp_path / "public.json"
    public.write_text("{}")
    identity = {
        "ProcessId": 123,
        "CreationDate": "fixed",
        "CommandLine": "python scripts/population_phase_canonical_pair.py",
    }
    states = iter([identity, None])
    monkeypatch.setattr(continuation, "worker_identity", lambda pid: next(states))
    calls = []
    original_run = continuation.subprocess.run

    def execute(command, **kwargs):
        if isinstance(command, str) or command[0] == "git":
            return original_run(command, **kwargs)
        calls.append(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        if command[2].endswith("evaluate_phase_canonical_screen.py"):
            (destination / "REPORT.json").write_text(
                json.dumps(
                    {
                        "population_size": 2608,
                        "primary_route": "F0",
                        "routes": {"F0": {"eligible_for_eap_validation": population_passes}},
                    }
                )
            )
        if command[2].endswith("release_native_pair_replay.py"):
            (destination / "RESULTS.json").write_text(
                json.dumps(
                    {
                        "historical_parity": "PASS" if profile_passes else "FAIL",
                        "candidate_executed": "--candidate" in command,
                    }
                )
            )

    monkeypatch.setattr(continuation.subprocess, "run", execute)
    output = tmp_path / "run"
    continuation.continue_work(screen, 123, public, output)
    assert len(calls) == expected_stages
    assert continuation.read(output / "DECISION.json")["outcome"] == outcome
    assert (output / "COMPLETE.json").exists()
    if not population_passes or not profile_passes:
        assert not any("--candidate" in command for command in calls)

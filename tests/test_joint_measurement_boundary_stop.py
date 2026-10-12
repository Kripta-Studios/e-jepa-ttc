"""A scheduling transition must never discard uncheckpointed work."""

import pytest

from operational.joint_measurement.boundary_stop import ROOT, check_owner, durable_boundary


def test_durable_boundary_requires_settled_post_scope_state():
    journal = {"completed_updates": 3010, "durable_updates": 3010, "pending_update_upper": 0}
    receipt = {"completed_updates": 3010, "complete_state": True}
    state = {"fit_id": "NATIVE-P2-FT-S7", "updates": 3010, "utc": "2026-10-12T19:37:01+00:00"}
    end = "2026-10-12T19:37:00+00:00"
    assert durable_boundary(journal, receipt, state, 3010, end)
    for changed in ({"pending_update_upper": 1}, {"durable_updates": 3000}):
        assert not durable_boundary({**journal, **changed}, receipt, state, 3010, end)
    assert not durable_boundary(journal, {**receipt, "complete_state": False}, state, 3010, end)
    assert not durable_boundary(journal, receipt, state, 3011, end)
    stale = {**state, "utc": "2026-10-12T19:36:59+00:00"}
    assert not durable_boundary(journal, receipt, stale, 3010, end)


def test_input_transition_requires_the_current_pair_fit():
    journal = {"completed_updates": 3000, "durable_updates": 3000, "pending_update_upper": 0}
    receipt = {"completed_updates": 3000, "complete_state": True}
    state = {"fit_id": "PAIR-DIRECT-S7", "updates": 3000, "utc": "2026-10-12T21:00:01+00:00"}
    end = "2026-10-12T21:00:00+00:00"
    assert durable_boundary(journal, receipt, state, 3000, end, "PAIR-DIRECT-S7")
    assert not durable_boundary(journal, receipt, state, 3000, end, "NATIVE-P2-FT-S7")


@pytest.mark.parametrize("change", ["pid_reused", "cwd", "command"])
def test_stop_refuses_foreign_process_identity(change):
    class Process:
        def create_time(self):
            return 2.0 if change == "pid_reused" else 1.0

        def cwd(self):
            return str(ROOT.parent if change == "cwd" else ROOT)

        def cmdline(self):
            module = (
                "another.runner" if change == "command" else "operational.joint_measurement.run"
            )
            return ["python.exe", "-m", module]

        def name(self):
            return "python.exe"

    with pytest.raises(ValueError, match="exact primary runner"):
        check_owner(Process(), 1.0)

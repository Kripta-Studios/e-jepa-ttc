"""Runtime routing must retain arm arguments and refuse an unregistered child command."""

import pytest

from operational.joint_measurement.adaptation_loss_graph_runtime import MODULE, child_command


@pytest.mark.parametrize("arm", ["LAST-BLOCK", "FULL"])
def test_only_child_module_changes(arm):
    original = [
        "python.exe",
        "-u",
        "-B",
        "-m",
        "operational.joint_measurement.adaptation_runtime",
        "child",
        "--arm",
        arm,
    ]
    updated = child_command(original)
    assert updated[:4] == original[:4] and updated[5:] == original[5:]
    assert updated[4] == MODULE and original[4] != MODULE


@pytest.mark.parametrize(
    ("position", "replacement"), [(4, "another.module"), (5, "run"), (7, "OTHER")]
)
def test_unexpected_child_is_rejected(position, replacement):
    command = [
        "python.exe",
        "-u",
        "-B",
        "-m",
        "operational.joint_measurement.adaptation_runtime",
        "child",
        "--arm",
        "FULL",
    ]
    command[position] = replacement
    with pytest.raises(ValueError):
        child_command(command)

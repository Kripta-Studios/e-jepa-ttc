"""Recovery must not mistake old failures or shell text for current evidence."""

import pytest

from operational.efficient_context.r1_supervisor import failure_kind, heavy_entrypoint, phase_stop


@pytest.mark.parametrize(
    "name,command,expected",
    [
        (
            "powershell.exe",
            ["powershell", "inspect operational.efficient_context.garl_train"],
            False,
        ),
        ("python.exe", ["python", "-m", "operational.efficient_context.garl_train_cached"], True),
        ("python.exe", ["python", "-m", "operational.train40_system.engine_graph_replay"], True),
        ("python.exe", ["python", "-m", "operational.efficient_context.r1_measure"], False),
        ("python.exe", ["python", "-m", "operational.stage76.evaluate"], True),
        ("python.exe", ["python", "-m"], False),
    ],
)
def test_r1_real_heavy_entrypoints_only(name, command, expected):
    assert heavy_entrypoint(name, command) == expected


def test_r1_last_exception_controls_retry():
    assert (
        failure_kind("InterruptedError: earlier\nValueError: parity failed")
        == "INTEGRITY_OR_CONTRACT_FAILURE"
    )
    assert failure_kind("noise\nInterruptedError: host RAM below threshold") == "TRANSIENT"
    assert failure_kind("FileNotFoundError: missing model") == "MISSING_DEPENDENCY"
    assert failure_kind("RuntimeError: CUDA failure") == "PERMANENT_FAILURE"


def test_r1_cpu_stop_reaches_real_child_scope(tmp_path):
    phase_stop(tmp_path, "window_parity", b"deadline")
    assert (tmp_path / "window_parity/STOP_REQUEST").read_bytes() == b"deadline"
    phase_stop(tmp_path, "measurement", b"explicit pause")
    phase_stop(tmp_path, "measurement", b"do not replace original reason")
    assert (tmp_path / "STOP_REQUEST").read_bytes() == b"explicit pause"

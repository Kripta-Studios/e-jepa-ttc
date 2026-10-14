"""A zero-update graph probe cannot admit training or compete with active drivers."""

from types import SimpleNamespace

import pytest

from operational.joint_measurement import geometry_graph_probe as probe

CHECKS = {
    "losses",
    "gradients",
    "rng_exact",
    "actual_cuda_graphs",
    "parameter_identity_unchanged",
    "weights_unchanged",
}


def test_all_gates_only_enable_separate_resume_qa():
    assert (
        probe.decision(
            {key: True for key in CHECKS}, 0.8, {"maximum_graph_to_eager_median_ratio": 0.9}
        )
        == "CANDIDATE_FOR_RESUME_QA"
    )


@pytest.mark.parametrize("failed", sorted(CHECKS))
def test_any_failed_or_missing_gate_rejects_graph(failed):
    checks = {key: key != failed for key in CHECKS}
    record = {"maximum_graph_to_eager_median_ratio": 0.9}
    assert probe.decision(checks, 0.5, record) == "REJECTED"
    del checks[failed]
    assert probe.decision(checks, 0.5, record) == "REJECTED"


@pytest.mark.parametrize("ratio", [0, -1, 0.91, float("inf"), float("nan")])
def test_invalid_or_insufficient_speed_does_not_enable_graph(ratio):
    assert (
        probe.decision(
            {key: True for key in CHECKS}, ratio, {"maximum_graph_to_eager_median_ratio": 0.9}
        )
        == "REJECTED"
    )


@pytest.mark.parametrize("module", ["rgb_run", "rgb_cached_run", "geometry_run"])
def test_running_driver_blocks_probe_even_between_gpu_scopes(monkeypatch, module):
    process = SimpleNamespace(
        info={
            "pid": 12345678,
            "cmdline": ["python", "-m", f"operational.joint_measurement.{module}"],
        },
        pid=12345678,
        cwd=lambda: str(probe.ROOT),
    )
    monkeypatch.setattr(probe.psutil, "process_iter", lambda fields: [process])
    with pytest.raises(RuntimeError, match="GPU_OWNER_WAIT"):
        probe.no_live_training_driver()

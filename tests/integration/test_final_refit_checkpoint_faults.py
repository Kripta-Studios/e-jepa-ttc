from __future__ import annotations

from pathlib import Path

from operational.final_refit.checkpoint_fault_probe import FAULTS, run_suite


def test_real_process_faults_resume_to_exact_cpu_state(tmp_path: Path) -> None:
    report = run_suite(tmp_path / "fault-probe")
    assert report["status"] == "PASS"
    assert [case["fault"] for case in report["cases"]] == list(FAULTS)
    assert all(case["all_exact"] for case in report["cases"])
    assert all(case["bounded_by_save_interval"] for case in report["cases"])
    assert all(case["completed_updates"] == 12 for case in report["cases"])
    assert all(case["scientific_endpoint"] is True for case in report["cases"])
    assert "machine_power_loss" in report["not_tested"]

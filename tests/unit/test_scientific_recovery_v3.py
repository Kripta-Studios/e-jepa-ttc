from __future__ import annotations

from pathlib import Path

from scripts.train_causal_scale_eap_screen import _blocking_worktree_state

ROOT = Path(__file__).resolve().parents[2]


def test_progress_monitor_tracked_edit_is_operational_not_scientific_dirty() -> None:
    state = _blocking_worktree_state(
        [
            " M scripts/monitor_scientific_recovery_v2.ps1",
            "?? historical.patch",
        ]
    )
    assert state["science_code_dirty"] is False
    assert state["ignored_operational_dirty_paths"] == [
        "scripts/monitor_scientific_recovery_v2.ps1"
    ]
    assert state["blocking_tracked_dirty_paths"] == []


def test_training_or_model_code_edit_remains_fail_closed() -> None:
    for line, expected in [
        (" M scripts/run_scientific_recovery_master_v2.ps1", "scripts/run_scientific_recovery_master_v2.ps1"),
        (" M src/e_jepa_ttc/models/causal_scale_ttc.py", "src/e_jepa_ttc/models/causal_scale_ttc.py"),
        (" M configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_legacy.yaml", "configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_legacy.yaml"),
    ]:
        state = _blocking_worktree_state([line])
        assert state["science_code_dirty"] is True
        assert state["blocking_tracked_dirty_paths"] == [expected]


def test_untracked_code_still_blocks_representative_screen() -> None:
    state = _blocking_worktree_state(["?? scripts/ad_hoc_training_change.py"])
    assert state["science_code_dirty"] is True
    assert state["untracked_code_paths"] == ["scripts/ad_hoc_training_change.py"]


def test_paired_bootstrap_falls_back_to_sequence_not_sample(tmp_path) -> None:
    import pandas as pd
    from scripts.paired_cluster_bootstrap import run
    rows = pd.DataFrame({
        "sample_token": ["a1", "a2", "b1", "b2"],
        "sequence_id": ["A", "A", "B", "B"],
        "target_ttc_s": [1.0, 2.0, 1.5, 2.5],
        "prediction_ttc_s": [1.1, 2.1, 1.4, 2.4],
    })
    e = tmp_path / "e.csv"; g = tmp_path / "g.csv"; out = tmp_path / "out.json"
    rows.to_csv(e, index=False); rows.to_csv(g, index=False)
    report = run(e, g, out, resamples=20, seed=1)
    assert report["cluster_definition"] == "sequence_only_fallback"
    assert report["clusters"] == 2


def test_paired_bootstrap_uses_external_track_metadata(tmp_path) -> None:
    import pandas as pd
    from scripts.paired_cluster_bootstrap import run
    rows = pd.DataFrame({
        "sample_token": ["a1", "a2", "b1", "b2"],
        "sequence_id": ["A", "A", "B", "B"],
        "target_ttc_s": [1.0, 2.0, 1.5, 2.5],
        "prediction_ttc_s": [1.1, 2.1, 1.4, 2.4],
    })
    meta = pd.DataFrame({
        "sample_token": ["a1", "a2", "b1", "b2"],
        "sequence_id": ["A", "A", "B", "B"],
        "track_id": ["t1", "t1", "t2", "t3"],
    })
    e = tmp_path / "e.csv"; g = tmp_path / "g.csv"; m = tmp_path / "meta.csv"; out = tmp_path / "out.json"
    rows.to_csv(e, index=False); rows.to_csv(g, index=False); meta.to_csv(m, index=False)
    report = run(e, g, out, resamples=20, seed=1, cluster_metadata=m)
    assert report["cluster_definition"] == "sequence_track_external_metadata"
    assert report["clusters"] == 3


def test_master_runner_treats_nvidia_smi_as_optional_observability() -> None:
    text = (ROOT / "scripts" / "run_scientific_recovery_master_v3.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "function Resolve-NvidiaSmi" in text
    assert "$script:NvidiaSmiExe" in text
    assert "nvidia_smi_available" in text
    assert "(& nvidia-smi)" not in text
    assert "(& nvidia-smi --query-gpu=" not in text


def test_master_runner_propagates_global_failures_after_packaging() -> None:
    text = (ROOT / "scripts" / "run_scientific_recovery_master_v3.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "$script:MasterExitCode = 1" in text
    assert "if ($script:MasterExitCode -ne 0)" in text
    assert "exit $script:MasterExitCode" in text
    assert 'throw "result packaging failed (exit $packageCode)"' in text

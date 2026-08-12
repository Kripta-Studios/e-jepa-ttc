from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

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
        (
            " M scripts/run_scientific_recovery_master_v2.ps1",
            "scripts/run_scientific_recovery_master_v2.ps1",
        ),
        (
            " M src/e_jepa_ttc/models/causal_scale_ttc.py",
            "src/e_jepa_ttc/models/causal_scale_ttc.py",
        ),
        (
            " M configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_legacy.yaml",
            "configs/model/e_jepa_causal_scale_event_v9_transport_r1_t002_legacy.yaml",
        ),
    ]:
        state = _blocking_worktree_state([line])
        assert state["science_code_dirty"] is True
        assert state["blocking_tracked_dirty_paths"] == [expected]


def test_untracked_code_still_blocks_representative_screen() -> None:
    state = _blocking_worktree_state(["?? scripts/ad_hoc_training_change.py"])
    assert state["science_code_dirty"] is True
    assert state["untracked_code_paths"] == ["scripts/ad_hoc_training_change.py"]


def test_paired_bootstrap_falls_back_to_sequence_not_sample(tmp_path) -> None:
    from scripts.paired_cluster_bootstrap import run

    rows = pd.DataFrame(
        {
            "sample_token": ["a1", "a2", "b1", "b2"],
            "sequence_id": ["A", "A", "B", "B"],
            "target_ttc_s": [1.0, 2.0, 1.5, 2.5],
            "prediction_ttc_s": [1.1, 2.1, 1.4, 2.4],
        }
    )
    e = tmp_path / "e.csv"
    g = tmp_path / "g.csv"
    out = tmp_path / "out.json"
    rows.to_csv(e, index=False)
    rows.to_csv(g, index=False)
    report = run(e, g, out, resamples=20, seed=1)
    assert report["cluster_definition"] == "sequence_only_fallback"
    assert report["clusters"] == 2


def test_paired_bootstrap_uses_external_track_metadata(tmp_path) -> None:
    from scripts.paired_cluster_bootstrap import run

    rows = pd.DataFrame(
        {
            "sample_token": ["a1", "a2", "b1", "b2"],
            "sequence_id": ["A", "A", "B", "B"],
            "target_ttc_s": [1.0, 2.0, 1.5, 2.5],
            "prediction_ttc_s": [1.1, 2.1, 1.4, 2.4],
        }
    )
    meta = pd.DataFrame(
        {
            "sample_token": ["a1", "a2", "b1", "b2"],
            "sequence_id": ["A", "A", "B", "B"],
            "track_id": ["t1", "t1", "t2", "t3"],
        }
    )
    e = tmp_path / "e.csv"
    g = tmp_path / "g.csv"
    m = tmp_path / "meta.csv"
    out = tmp_path / "out.json"
    rows.to_csv(e, index=False)
    rows.to_csv(g, index=False)
    meta.to_csv(m, index=False)
    report = run(e, g, out, resamples=20, seed=1, cluster_metadata=m)
    assert report["cluster_definition"] == "sequence_track_external_metadata"
    assert report["clusters"] == 3


def _prediction_rows(*, track_ids: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_token": ["a1", "a2", "b1", "b2"],
            "sequence_id": ["A", "A", "B", "B"],
            "target_ttc_s": [1.0, 4.0, 7.0, -1.0],
            "prediction_ttc_s": [1.1, 4.1, 7.1, -1.1],
        }
    )
    if track_ids is not None:
        frame["track_id"] = track_ids
    return frame


def _write_predictions(
    tmp_path: Path,
    *,
    ejepa_tracks: list[str] | None,
    garl_tracks: list[str] | None,
) -> tuple[Path, Path, Path]:
    ejepa = tmp_path / "ejepa.csv"
    garl = tmp_path / "garl.csv"
    output = tmp_path / "paired.json"
    _prediction_rows(track_ids=ejepa_tracks).to_csv(ejepa, index=False)
    _prediction_rows(track_ids=garl_tracks).to_csv(garl, index=False)
    return ejepa, garl, output


@pytest.mark.parametrize(
    ("ejepa_tracks", "garl_tracks", "expected_definition"),
    (
        (["t1", "t1", "t2", "t3"], ["t1", "t1", "t2", "t3"], "sequence_track"),
        (["t1", "t1", "t2", "t3"], None, "sequence_track_ejepa_only"),
        (None, ["t1", "t1", "t2", "t3"], "sequence_track_garl_only"),
    ),
)
def test_paired_bootstrap_namespaces_optional_track_columns(
    tmp_path: Path,
    ejepa_tracks: list[str] | None,
    garl_tracks: list[str] | None,
    expected_definition: str,
) -> None:
    from scripts.paired_cluster_bootstrap import run

    ejepa, garl, output = _write_predictions(
        tmp_path,
        ejepa_tracks=ejepa_tracks,
        garl_tracks=garl_tracks,
    )
    report = run(ejepa, garl, output, resamples=20, seed=1)
    assert report["cluster_definition"] == expected_definition


def test_paired_bootstrap_rejects_different_track_ids(tmp_path: Path) -> None:
    from scripts.paired_cluster_bootstrap import run

    ejepa, garl, output = _write_predictions(
        tmp_path,
        ejepa_tracks=["t1", "t1", "t2", "t3"],
        garl_tracks=["t1", "different", "t2", "t3"],
    )
    with pytest.raises(ValueError, match="Track IDs differ"):
        run(ejepa, garl, output, resamples=20, seed=1)


def test_paired_bootstrap_rejects_prediction_track_mismatch_with_metadata(
    tmp_path: Path,
) -> None:
    from scripts.paired_cluster_bootstrap import run

    ejepa, garl, output = _write_predictions(
        tmp_path,
        ejepa_tracks=["t1", "t1", "t2", "t3"],
        garl_tracks=None,
    )
    metadata = tmp_path / "metadata.csv"
    _prediction_rows(track_ids=["t1", "different", "t2", "t3"])[
        ["sample_token", "sequence_id", "track_id"]
    ].to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="external metadata track IDs differ"):
        run(
            ejepa,
            garl,
            output,
            resamples=20,
            seed=1,
            cluster_metadata=metadata,
        )


def test_paired_bootstrap_rejects_duplicate_sample_tokens(tmp_path: Path) -> None:
    from scripts.paired_cluster_bootstrap import run

    ejepa, garl, output = _write_predictions(
        tmp_path,
        ejepa_tracks=None,
        garl_tracks=None,
    )
    duplicate = _prediction_rows()
    duplicate.loc[1, "sample_token"] = "a1"
    duplicate.to_csv(ejepa, index=False)
    with pytest.raises(ValueError, match="duplicate sample_token"):
        run(ejepa, garl, output, resamples=20, seed=1)


def test_paired_bootstrap_records_input_hashes(tmp_path: Path) -> None:
    from scripts.paired_cluster_bootstrap import run

    ejepa, garl, output = _write_predictions(
        tmp_path,
        ejepa_tracks=None,
        garl_tracks=None,
    )
    metadata = tmp_path / "metadata.csv"
    _prediction_rows(track_ids=["t1", "t1", "t2", "t3"])[
        ["sample_token", "sequence_id", "track_id"]
    ].to_csv(metadata, index=False)
    report = run(
        ejepa,
        garl,
        output,
        resamples=20,
        seed=1,
        cluster_metadata=metadata,
    )
    for key, path in (
        ("ejepa_predictions", ejepa),
        ("garl_predictions", garl),
        ("cluster_metadata", metadata),
    ):
        assert report["sources"][key]["path"] == str(path.resolve())
        assert report["sources"][key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["comparison_scope"] == "current_candidate_public_validation"


def _readiness_inputs() -> tuple[dict, dict, dict, dict, dict]:
    from e_jepa_ttc.artifacts.hashing import sign_artifact

    contract = {
        "artifact_type": "scientific_recovery_contract_audit_v1",
        "claim_contract": {
            "matched_oracle_roi_comparison_allowed": True,
            "event_only_neural_forward_claim_allowed": True,
        },
        "sealed_sources": {"private_test_opened": False},
    }
    prefix = {
        "artifact_type": "scientific_recovery_prefix_causality_audit_v1",
        "results": {"causal_left": {"prefix_invariant": True}},
        "sealed_sources": {"private_test_opened": False},
    }
    candidate = {
        "artifact_type": "causal_scale_eap_public_validation_screen_v1",
        "status": "completed_public_validation_only",
        "official_test_opened": False,
        "selectable": True,
        "predictions": {"sha256": "a" * 64},
    }
    garl = {
        "artifact_type": "garl_event_only_matched_cached_training_v1",
        "status": "completed_max_epochs",
        "sealed_sources": {"private_test_opened": False},
        "artifacts": {"predictions": {"sha256": "b" * 64}},
    }
    paired = {
        "artifact_type": "paired_cluster_bootstrap_ejepa_vs_garl_v1",
        "status": "completed_public_validation_only",
        "comparison_scope": "current_candidate_public_validation",
        "cluster_definition": "sequence_track_external_metadata",
        "checks": {
            "exact_sample_tokens": True,
            "target_equality_atol_1e_5": True,
            "paired_evaluation": True,
            "input_provenance_complete": True,
            "private_test_opened": False,
        },
        "sources": {
            "ejepa_predictions": {"path": "ejepa.csv", "sha256": "a" * 64},
            "garl_predictions": {"path": "garl.parquet", "sha256": "b" * 64},
            "cluster_metadata": {"path": "metadata.parquet", "sha256": "c" * 64},
        },
    }
    for artifact in (contract, prefix, candidate, garl):
        sign_artifact(artifact)
    sign_artifact(paired)
    return contract, prefix, candidate, garl, paired


def test_claim_readiness_accepts_matching_paired_provenance() -> None:
    from scripts.build_scientific_claim_readiness import build_readiness

    result = build_readiness(*_readiness_inputs(), candidate_mode="causal_left")
    assert result["checks"]["paired_exact_sample_comparison_complete"] is True
    assert result["paired_provenance_validation"]["status"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (
            lambda paired: paired["sources"]["ejepa_predictions"].update(sha256="stale"),
            "candidate_predictions_sha_matches",
        ),
        (
            lambda paired: paired["sources"]["garl_predictions"].update(sha256="other-garl"),
            "garl_predictions_sha_matches",
        ),
        (lambda paired: paired.update(artifact_type="wrong"), "artifact_type"),
        (lambda paired: paired.update(status="incomplete"), "status"),
        (lambda paired: paired.update(comparison_scope="diagnostic_only"), "comparison_scope"),
    ),
)
def test_claim_readiness_rejects_stale_or_invalid_paired_artifact(
    mutation,
    reason: str,
) -> None:
    from e_jepa_ttc.artifacts.hashing import sign_artifact
    from scripts.build_scientific_claim_readiness import build_readiness

    contract, prefix, candidate, garl, paired = _readiness_inputs()
    mutation(paired)
    sign_artifact(paired)
    result = build_readiness(
        contract,
        prefix,
        candidate,
        garl,
        paired,
        candidate_mode="causal_left",
    )
    assert result["checks"]["paired_exact_sample_comparison_complete"] is False
    assert result["paired_provenance_validation"]["checks"][reason] is False
    assert result["readiness"] != "READY_FOR_ONE_SHOT_SEALED_MATCHED_ORACLE_ROI_TEST"


def test_claim_readiness_rejects_invalid_paired_artifact_signature() -> None:
    from scripts.build_scientific_claim_readiness import build_readiness

    contract, prefix, candidate, garl, paired = _readiness_inputs()
    paired["sources"]["ejepa_predictions"]["path"] = "tampered.csv"
    result = build_readiness(
        contract,
        prefix,
        candidate,
        garl,
        paired,
        candidate_mode="causal_left",
    )
    assert result["paired_provenance_validation"]["checks"]["artifact_signature"] is False
    assert result["checks"]["paired_exact_sample_comparison_complete"] is False


def test_master_runner_clears_paired_path_before_regeneration() -> None:
    text = (ROOT / "scripts" / "run_scientific_recovery_master_v3.ps1").read_text(
        encoding="utf-8-sig"
    )
    step73 = text[text.index("# 73:") : text.index("# Diagnostic-only paired comparison")]
    assert "$paired = $null" in step73
    assert "Move-StaleArtifactToQuarantine" in step73
    assert step73.index("$paired = $null") < step73.index('"73_paired_bootstrap"')
    assert step73.index("$paired = $pairedPath") > step73.index('"73_paired_bootstrap"')


def test_causal_replication_uses_one_fixed_a4_parent_summary() -> None:
    text = (ROOT / "scripts" / "run_scientific_recovery_master_v3.ps1").read_text(
        encoding="utf-8-sig"
    )
    start = text.index("$fixedCausalParentSummary")
    causal = text[start : text.index("$causalRep=", start)]
    assert causal.count('"--base-summary"') == 1
    assert "fixed A4 causal seed7 parent" in causal


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

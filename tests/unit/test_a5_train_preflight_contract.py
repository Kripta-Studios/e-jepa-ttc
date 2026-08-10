from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "a5_train_screen_contract", ROOT / "scripts" / "train_causal_scale_eap_screen.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _v3_payload(*, overlap: int = 0, selected_radius: int = 1) -> dict:
    return {
        "artifact_type": "a5_transport_preflight_train_only_v3_confirmation",
        "artifact_sha256": "signed-v3",
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "optimizer_steps": 0,
            "v2_v3_index_overlap": overlap,
            "v3_confirmation_rows": 1536,
        },
        "decision": {
            "a5_corr_authorized": True,
            "selected_radius": selected_radius,
            "selected_temperature": 0.02,
            "checks": {"student_soft_physics": True},
        },
        "discovery_contract": {
            "candidate_radius": 1,
            "candidate_temperature": 0.02,
            "no_candidate_reselection_in_v3": True,
        },
        "interpretation_contract": {
            "no_radius_or_temperature_search_in_v3": True,
            "no_training_or_optimizer_steps": True,
            "ttc_labels_are_not_used": True,
            "v2_rejection_is_preserved_and_not_overwritten": True,
            "v3_candidate_was_selected_from_v2_train_only_discovery": True,
            "v3_confirmation_indices_are_disjoint_from_v2_indices": True,
        },
    }


def _decision_contract() -> dict:
    return {
        "representation_change": {
            "type": "a4_endpoint_dino_plus_event_native_local_cross_time_transport",
            "dino_endpoint_teacher_unchanged_from_a4": True,
            "dino_temporal_delta_removed": True,
            "transport_model_input": "event_dense_features_only",
            "bbox_used_by_transport": False,
            "rgb_used_at_inference": False,
            "jepa_objective": False,
            "direct_ttc_regressor_from_flow": False,
            "analytic_height_ratio_remains_primary_backbone": True,
            "transport_radius": 1,
            "transport_pairs": ["t0_to_t1", "t1_to_t2"],
        },
        "preflight_contract": {
            "artifact_type": "a5_transport_preflight_train_only_v3_confirmation",
            "artifact": "artifacts/metrics/v3.json",
            "artifact_sha256": "signed-v3",
            "file_sha256": "file-v3",
            "selected_radius": 1,
            "selected_temperature": 0.02,
        },
    }


def _configs() -> tuple[SimpleNamespace, SimpleNamespace]:
    training = SimpleNamespace(
        representation_supervision="dinov3_local_relational",
        representation_temporal_delta_weight=0.0,
        initialization_mode="none",
        freeze_encoder=False,
    )
    model = SimpleNamespace(
        transport_enabled=True,
        transport_radius=1,
        transport_temperature=0.02,
    )
    return training, model


def _run_with_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict):
    artifact = tmp_path / "v3.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(MODULE, "_resolve", lambda _: artifact)
    monkeypatch.setattr(MODULE, "verify_artifact_hash", lambda _: True)
    monkeypatch.setattr(MODULE, "_sha256", lambda _: "file-v3")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    training, model = _configs()
    return MODULE._validate_a5_transport_change(training, model, _decision_contract())


def test_train_screen_accepts_signed_v3_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_with_payload(tmp_path, monkeypatch, _v3_payload())
    assert result is not None
    assert result["artifact_type"] == "a5_transport_preflight_train_only_v3_confirmation"
    assert result["a5_corr_authorized"] is True
    assert result["selected_radius"] == 1
    assert result["selected_temperature"] == 0.02


def test_train_screen_rejects_v3_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _run_with_payload(tmp_path, monkeypatch, _v3_payload(overlap=1))


def test_train_screen_rejects_v3_radius_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="radius"):
        _run_with_payload(tmp_path, monkeypatch, _v3_payload(selected_radius=2))

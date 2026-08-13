"""Tests for exact fold geometry audit evidence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.training.causal_scale_eap import _module_tensor_sha256
from scripts.audit_v5_fold_geometry import build_audit


def _checkpoint(path: Path, model: CausalScaleTTC) -> None:
    torch.save(
        {
            "artifact_type": "causal_scale_eap_grouped_dev_checkpoint_v1",
            "model_config": model.checkpoint_config(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def test_a6_geometry_audit_proves_exact_parent_state_and_probe(tmp_path: Path) -> None:
    torch.manual_seed(7)
    parent = CausalScaleTTC(CausalScaleTTCConfig(dropout=0.0))
    child = CausalScaleTTC(
        CausalScaleTTCConfig(
            dropout=0.0,
            transport_enabled=True,
            transport_adapter_enabled=True,
            transport_radius=1,
            transport_temperature=0.02,
        )
    )
    child.encoder.load_state_dict(parent.encoder.state_dict(), strict=True)
    parent_path = tmp_path / "parent.pt"
    child_path = tmp_path / "child.pt"
    _checkpoint(parent_path, parent)
    _checkpoint(child_path, child)
    parent_hash = _module_tensor_sha256(parent.encoder)
    frozen = [f"encoder.{name}" for name, _ in child.encoder.named_parameters()]
    optimizer = [name for name, _ in child.named_parameters() if not name.startswith("encoder.")]
    summary = {
        "artifact_type": "causal_scale_eap_train_only_grouped_dev_run_v1",
        "development_protocol": {"fold_identity": {"fold": 0}},
        "initialization": {
            "checkpoint_sha256": __import__("hashlib").sha256(parent_path.read_bytes()).hexdigest(),
            "initial_primary_encoder_sha256": parent_hash,
            "final_primary_encoder_sha256": parent_hash,
            "primary_encoder_exact_initial": True,
            "frozen_parameter_names": frozen,
            "optimizer_parameter_names": optimizer,
            "frozen_optimizer_overlap": [],
        },
    }
    sign_artifact(summary)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = build_audit(
        parent_checkpoint=parent_path,
        child_checkpoint=child_path,
        child_summary=summary_path,
        arm="a6",
        fold=0,
    )

    assert report["status"] == "completed_exact_primary_geometry"
    assert report["primary_geometry"]["exact_tensor_equality"] is True
    assert report["primary_geometry"]["fixed_probe_exact"] is True
    assert report["transport_encoder"] is None

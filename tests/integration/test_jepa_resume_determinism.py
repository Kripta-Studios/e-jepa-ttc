from __future__ import annotations

import torch

from tests.integration._artifact_helpers import artifact_path, read_artifact


def test_resume_contract_preserves_fingerprint_rng_and_checkpoint_state() -> None:
    metrics = read_artifact("artifacts/runs/carla_jepa_resume_contract_smoke_v1/metrics.json")
    checkpoint = torch.load(
        artifact_path(metrics["last_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    assert len(metrics["run_fingerprint"]) == 64
    assert checkpoint["checkpoint_role"] == "last"
    assert int(checkpoint["epoch"]) == metrics["epochs_completed"]
    assert checkpoint["split_manifest_sha256"] == metrics["split_manifest_sha256"]
    assert checkpoint["target_encoder_state_dict"]

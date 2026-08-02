from __future__ import annotations

from tests.integration._artifact_helpers import artifact_path, read_artifact


def test_zero_shot_evaluation_has_no_target_training_updates() -> None:
    validation = read_artifact(
        "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/zero_shot_validation.json"
    )
    aggregate = read_artifact("artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/zero_shot_aggregate.json")
    assert artifact_path(validation["checkpoint"]).is_file()
    assert validation["training_updates_on_target_dataset"] == 0
    assert validation["same_ttc_head_as_eap_training"] is True
    assert validation["sample_count"] == aggregate["sample_count"]
    assert aggregate["sequence_count"] >= 2

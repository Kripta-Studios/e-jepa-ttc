from __future__ import annotations

import math

from tests.integration._artifact_helpers import artifact_path, read_artifact


def test_garl_signed_training_smoke_uses_official_labels_and_emits_weights() -> None:
    summary = read_artifact("artifacts/runs/garl_v4_lhr_smoke_seed7_shardlocal_modern/summary.json")
    assert summary["uses_official_garl_ttc_labels"] is True
    assert summary["uses_reconstructed_public_eap_ttc"] is False
    assert summary["no_privileged_model_inputs"] is True
    assert math.isfinite(summary["best_validation_sequence_macro_paper_MiD_overall"])
    assert artifact_path(summary["weights_only_checkpoint"]).is_file()

from __future__ import annotations

from tests.integration._artifact_helpers import artifact_path, read_artifact


def test_jepa_tubelet_overfit_has_real_checkpoint_and_descending_validation_loss() -> None:
    metrics = read_artifact("artifacts/runs/carla_jepa_overfit_32_current_seed7/metrics.json")
    history = metrics["history"]
    assert metrics["epochs_completed"] >= 1
    assert artifact_path(metrics["best_checkpoint"]).is_file()
    assert history[-1]["validation"]["loss"] < history[0]["validation"]["loss"]
    assert all(row["validation"]["context_collapsed_dimension_fraction"] < 0.8 for row in history)

from __future__ import annotations

import math

import pytest

from e_jepa_ttc.evaluation.semantic_shortcuts import (
    BENCHMARK_ARMS,
    SemanticShortcutConfig,
    assess_eap_ssl_health,
    run_semantic_shortcut_benchmark,
)


def test_semantic_shortcut_config_rejects_non_block_aligned_latent() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SemanticShortcutConfig(latent_dim=10, block_size=4).validate()


def test_semantic_shortcut_benchmark_smoke_is_complete_and_finite() -> None:
    config = SemanticShortcutConfig(
        train_sequences=4,
        test_sequences=3,
        steps_per_sequence=4,
        hidden_dim=24,
        latent_dim=8,
        block_size=4,
        epochs=1,
        batch_size=8,
        rff_features=8,
    )
    result = run_semantic_shortcut_benchmark(
        config=config,
        seeds=(7,),
        device_name="cpu",
    )

    assert result["status"] == "complete"
    assert result["uses_real_dataset"] is False
    assert result["uses_ttc_labels_for_representation_training"] is False
    assert set(result["aggregate"]) == set(BENCHMARK_ARMS)
    assert len(result["runs"]) == len(BENCHMARK_ARMS)
    for run in result["runs"]:
        assert all(math.isfinite(value) for value in run["probes"].values())
    assert result["decision"]["scope"].startswith("Synthetic mechanistic")


def test_eap_health_distinguishes_rank_warning_from_semantic_diagnosis() -> None:
    payload = {
        "history": [
            {
                "validation": {
                    "context_effective_rank": 2.25,
                    "pred_effective_rank": 1.1,
                    "target_effective_rank": 5.1,
                    "context_collapsed_dimension_fraction": 0.03125,
                }
            }
        ]
    }
    result = assess_eap_ssl_health(payload, embedding_dim=192)

    assert result["rank_deficiency_warning"] is True
    assert result["statistical_collapse_guard_triggered"] is False
    assert result["semantic_shortcut_confirmed"] is False

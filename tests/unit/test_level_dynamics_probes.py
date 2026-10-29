from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.evaluation.level_dynamics_probes import (
    embedding_diagnostics,
    fit_frozen_probe,
    run_identity_shortcut_diagnostics,
    run_level_dynamics_probes,
)


def _fixture() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    embeddings = np.arange(48, dtype=np.float64).reshape(8, 6)
    metadata = {
        "split": np.asarray(["train"] * 4 + ["validation"] * 4),
        "sequence_id": np.asarray(["a"] * 4 + ["b"] * 4),
        "track_id": np.asarray(
            ["track-1", "track-1", "track-2", "invalid", "track-3", "track-3", "track-4", "track-4"]
        ),
        "expansion": np.arange(8, dtype=np.float64),
        "log_height_ratio": np.log1p(np.arange(8, dtype=np.float64)),
        "ttc_seconds": np.linspace(0.5, 4.0, 8),
        "event_count": np.arange(8, dtype=np.float64) + 1,
        "event_rate": np.arange(8, dtype=np.float64) + 2,
        "timestamp_s": np.arange(8, dtype=np.float64),
        "horizon_s": np.full(8, 0.1),
    }
    return embeddings, metadata


def test_frozen_probes_are_deterministic_and_sequence_disjoint() -> None:
    embeddings, metadata = _fixture()
    bindings = {
        "checkpoint_hash": "a" * 64,
        "manifest_hash": "b" * 64,
        "config_hash": "c" * 64,
        "code_commit": "d" * 40,
    }
    first = run_level_dynamics_probes(embeddings, metadata, **bindings)
    second = run_level_dynamics_probes(embeddings, metadata, **bindings)
    assert first == second
    assert first["diagnostics"]["effective_rank"] >= 1.0
    assert first["diagnostics"]["duplication_rate"] == 0.0
    assert any(probe["target"] == "expansion" for probe in first["probes"])
    assert all(
        probe["diagnostic_only"]
        for probe in first["probes"]
        if probe["target"] in {"ttc_seconds", "log_ttc_seconds"}
    )


def test_probe_rejects_overlap_and_validation_fit() -> None:
    embeddings, metadata = _fixture()
    bindings = {
        "checkpoint_hash": "a" * 64,
        "manifest_hash": "b" * 64,
        "config_hash": "c" * 64,
        "code_commit": "d" * 40,
    }
    overlapping = dict(metadata)
    overlapping["sequence_id"] = np.asarray(["a"] * 8)
    with pytest.raises(ValueError, match="overlap"):
        run_level_dynamics_probes(embeddings, overlapping, **bindings)
    with pytest.raises(ValueError, match="fit only on train"):
        fit_frozen_probe(embeddings, metadata, "expansion", fit_split="validation")


def test_embedding_diagnostics_constant_is_finite() -> None:
    result = embedding_diagnostics(np.ones((4, 3)))
    assert result["effective_rank"] == 0.0
    assert result["duplication_rate"] == 0.75


def test_unseen_categorical_validation_classes_are_unavailable_not_zero() -> None:
    embeddings = np.eye(6, dtype=np.float64)
    metadata = {
        "split": np.asarray(["train"] * 3 + ["validation"] * 3),
        "sequence_id": np.asarray(["s-train"] * 3 + ["s-val"] * 3),
    }
    result = fit_frozen_probe(embeddings, metadata, "sequence_id")
    assert result.metrics["status"] == "unavailable"
    assert result.metrics["unavailable_reason"] == "unseen_validation_classes"
    assert result.metrics["validation_accuracy"] is None


def test_numeric_probe_reports_macro_by_sequence() -> None:
    embeddings = np.eye(8, dtype=np.float64)
    metadata = {
        "split": np.asarray(["train"] * 4 + ["validation"] * 4),
        "sequence_id": np.asarray(["a"] * 4 + ["b", "b", "c", "c"]),
        "event_count": np.arange(8, dtype=np.float64),
    }
    result = fit_frozen_probe(embeddings, metadata, "event_count")
    assert result.metrics["validation_sequence_count"] == 2
    assert result.metrics["validation_mae_macro_by_sequence"] is not None
    assert set(result.metrics["validation_by_sequence"]) == {"b", "c"}


def test_identity_diagnostic_uses_guarded_duplicate_safe_temporal_folds() -> None:
    n = 18
    embeddings = np.random.default_rng(4).normal(size=(n, 5))
    windows = [[index * 1000, index * 1000 + 100] for index in range(n)]
    windows[1] = windows[0]
    metadata = {
        "sequence_id": np.asarray(["s1"] * 9 + ["s2"] * 9),
        "track_id": np.asarray(["t1", "t1", "t2"] * 6),
        "timestamp_s": np.arange(n, dtype=np.float64) * 0.1,
        "events_path": np.asarray(["events.h5"] * n),
        "event_windows_us": np.asarray(windows, dtype=object),
    }
    result = run_identity_shortcut_diagnostics(
        embeddings,
        metadata,
        checkpoint_hash="a" * 64,
        manifest_hash="b" * 64,
        config_hash="c" * 64,
        code_commit="d" * 40,
        seed=7,
        context_s=0.2,
        max_horizon_s=0.3,
        n_folds=3,
    )
    assert result["artifact_type"] == "identity_shortcut_diagnostics_v1"
    assert result["diagnostic_only"] is True
    assert result["excluded_from_ssl_selection_and_promotion"] is True
    assert result["hparams"]["guard_gap_s"] >= 0.5
    fold_by_index = {
        index: fold["fold"] for fold in result["folds"] for index in fold["test_indices"]
    }
    assert fold_by_index[0] == fold_by_index[1]
    assert len(result["fold_hash"]) == 64

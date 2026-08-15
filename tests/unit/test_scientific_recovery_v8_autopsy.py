"""Contracts for the fail-closed V8 mechanism-autopsy core."""

from __future__ import annotations

import hashlib
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (
    AGGREGATE_V8_REQUIRED_KEYS,
    OOF_V8_REQUIRED_COLUMNS,
    REPLAY_MECHANISM_REQUIRED_COLUMNS,
    REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS,
    align_oof_frames,
    assert_causal_prefix_invariance,
    classify_mechanism,
    general_gate,
    hierarchical_sequence_bootstrap,
    mechanism_cuts,
    mechanism_cuts_from_pruned_replay,
    raw_mid_per_sample,
    row_identity_sha256,
    target_sha256,
    validate_aggregate_payload,
    validate_counterfactual_identity,
    validate_oof_frame,
    validate_replay_frame,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from scripts.analyze_v5_a8_oof_failure_modes import raw_mid_per_sample as official_raw_mid


def _sha(letter: str) -> str:
    return hashlib.sha256(letter.encode("utf-8")).hexdigest()


def _frame(*, model_name: str = "a5", prediction_shift: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Every sequence covers the four signed Garl bins, making bootstrap draws
    # finite even when a single sequence is sampled repeatedly.
    for sequence in ("sequence-a", "sequence-b"):
        for index, target in enumerate((-1.0, 1.0, 4.0, 7.0)):
            rows.append(
                {
                    "token_id": f"{sequence}-token-{index}",
                    "sequence_id": sequence,
                    "track_id": f"track-{index % 2}",
                    "outer_fold": 0 if sequence == "sequence-a" else 1,
                    "seed": 7,
                    "target_ttc": target,
                    "sample_weight": 1.0,
                    "prediction_ttc": target + prediction_shift,
                    "prediction_log_variance": -1.0,
                    "finite": True,
                    "failure_reason": "",
                    "event_count": 100 + index,
                    "event_rate": 10.0 + index,
                    "support_ms": 100.0,
                    "model_name": model_name,
                    "config_sha256": _sha("config"),
                    "checkpoint_sha256": _sha(model_name),
                    "known_mask": True,
                    "sensor_support": 0.9,
                    "guard_margin": 0.1 + index,
                    "pair_log_height_ratio": 0.2,
                    "analytic_log_height_ratio": 0.1,
                    "residual_log_height_ratio": 0.1,
                    "pair_ttc": target,
                    "pair_inverse_ttc": 1.0 / target,
                    "pair_current_contribution": 0.2,
                    "pair_previous_contribution": 0.1,
                    "blend_output": 0.5,
                    "foreground_mass": 0.7,
                    "effective_mass": 0.6,
                    "geometry_tokens": "[0.1,0.2]",
                    "pair_tokens": "[0.3,0.4]",
                    "transport_raw": "[0.1,0.2]",
                    "transport_tokens": "[0.3,0.4]",
                    "endpoint_feature_norm": 1.0,
                    "occupancy": 0.2,
                    "occupancy_entropy": 0.3 + index,
                    "motion_magnitude": 0.4 + index,
                    "cycle_consistency": 0.0,
                    "category": "car" if index % 2 else "pedestrian",
                }
            )
    return pd.DataFrame(rows)


def test_oof_schema_is_strict_and_hashes_are_order_independent() -> None:
    frame = _frame()
    validated = validate_oof_frame(frame)
    assert len(validated) == 8
    assert row_identity_sha256(frame) == row_identity_sha256(frame.sample(frac=1.0, random_state=2))
    assert target_sha256(frame) == target_sha256(frame.sample(frac=1.0, random_state=3))
    with pytest.raises(ValueError, match="duplicate token_id"):
        validate_oof_frame(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    malformed = frame.drop(columns=["config_sha256"])
    with pytest.raises(ValueError, match="lacks required columns"):
        validate_oof_frame(malformed)


def test_raw_mid_matches_existing_official_formula() -> None:
    target = [-1.0, 1.0, 2.0, -2.0]
    prediction = [-1.1, 1.1, 2.1, -2.1]
    np.testing.assert_array_equal(
        raw_mid_per_sample(target, prediction), official_raw_mid(target, prediction)
    )


def test_autopsy_source_is_clean_utf8_without_mojibake() -> None:
    source = Path("src/e_jepa_ttc/evaluation/scientific_recovery_v8.py").read_text(encoding="utf-8")
    assert "Â" not in source
    assert "â" not in source
    assert "sequence-to-track" in source


def test_exact_row_identity_and_targets_are_required_for_alignment() -> None:
    a5 = _frame(model_name="a5")
    c2f = _frame(model_name="c2f", prediction_shift=0.1)
    aligned = align_oof_frames({"a5": a5, "c2f": c2f})
    assert len(aligned) == len(a5)
    assert {"a5_prediction_ttc", "c2f_prediction_ttc"}.issubset(aligned.columns)
    drifted = c2f.copy()
    drifted.loc[0, "target_ttc"] = 1.5
    with pytest.raises(ValueError, match="targets differ"):
        align_oof_frames({"a5": a5, "c2f": drifted})


def test_replay_and_counterfactual_identity_contract() -> None:
    reference = _frame()
    assert set(REPLAY_MECHANISM_REQUIRED_COLUMNS).issubset(reference.columns)
    validate_replay_frame(reference)
    intervention = reference.copy()
    intervention["prediction_ttc"] += 0.25
    hashes = validate_counterfactual_identity(reference, intervention)
    assert hashes["reference_prediction_sha256"] != hashes["counterfactual_prediction_sha256"]
    intervention.loc[0, "sequence_id"] = "wrong-sequence"
    with pytest.raises(ValueError, match="row identity"):
        validate_counterfactual_identity(reference, intervention)


def test_failed_rows_require_explicit_failure_tracking() -> None:
    frame = _frame()
    frame.loc[0, "prediction_ttc"] = float("nan")
    frame.loc[0, "prediction_log_variance"] = float("nan")
    frame.loc[0, "finite"] = False
    frame.loc[0, "failure_reason"] = "no_known_support"
    validate_oof_frame(frame)
    frame.loc[0, "failure_reason"] = ""
    with pytest.raises(ValueError, match="lacks failure_reason"):
        validate_oof_frame(frame)


def test_hierarchical_bootstrap_is_deterministic_and_sequence_clustered() -> None:
    frame = _frame()
    frame["reference_prediction"] = frame["target_ttc"] + 0.3
    first = hierarchical_sequence_bootstrap(
        frame,
        candidate_prediction_column="prediction_ttc",
        reference_prediction_column="reference_prediction",
        resamples=100,
        seed=11,
    )
    second = hierarchical_sequence_bootstrap(
        frame,
        candidate_prediction_column="prediction_ttc",
        reference_prediction_column="reference_prediction",
        resamples=100,
        seed=11,
    )
    assert first == second
    assert first["method"] == "hierarchical_sequence_then_track_cluster_bootstrap"
    assert first["delta_candidate_minus_reference"]["probability_candidate_lower_mid"] == 1.0
    with pytest.raises(ValueError, match="at least two sequences"):
        hierarchical_sequence_bootstrap(frame[frame["sequence_id"] == "sequence-a"], resamples=10)


def test_mechanism_rules_are_explicit_and_can_remain_inconclusive() -> None:
    h1 = classify_mechanism(
        {
            "a5_delta_mid_vs_reference": -4.0,
            "analytic_dynamic_spearman": 0.4,
            "residual_dynamic_spearman": 0.1,
            "sequence_concentration": 0.3,
            "innocuous_counterfactual_delta_mid": 0.2,
            "regime_complementarity": 0.0,
            "causal_regime_auroc": 0.5,
        }
    )
    assert h1["decision"] == "H1"
    h2 = classify_mechanism(
        {
            "a5_delta_mid_vs_reference": -4.0,
            "analytic_dynamic_spearman": 0.0,
            "residual_dynamic_spearman": 0.0,
            "sequence_concentration": 0.8,
            "innocuous_counterfactual_delta_mid": 0.2,
            "regime_complementarity": 0.8,
            "causal_regime_auroc": 0.9,
        }
    )
    assert h2["decision"] == "H2"
    h3 = classify_mechanism(
        {
            "a5_delta_mid_vs_reference": -1.0,
            "analytic_dynamic_spearman": 0.15,
            "residual_dynamic_spearman": 0.15,
            "sequence_concentration": 0.3,
            "innocuous_counterfactual_delta_mid": 0.2,
            "regime_complementarity": 0.2,
            "causal_regime_auroc": 0.7,
        }
    )
    assert h3["decision"] == "H3"
    assert classify_mechanism({})["decision"] == "INCONCLUSIVE"


def test_mechanism_cuts_include_prespecified_regimes_without_becoming_features() -> None:
    cuts = mechanism_cuts(_frame())
    assert {"ttc_bucket", "sequence", "track", "event_density_quartile"}.issubset(cuts)
    assert set(cuts["ttc_bucket"]) == {"0-3", "3-6", ">6", "negative_or_receding"}
    assert "category_analysis_only" in cuts




def test_pruned_mechanism_cuts_match_full_replay_for_scalar_diagnostics() -> None:
    full = _frame()
    scalar_columns = [
        *OOF_V8_REQUIRED_COLUMNS,
        "guard_margin",
        "occupancy_entropy",
        "motion_magnitude",
        "category",
    ]
    pruned = full.loc[:, list(dict.fromkeys(scalar_columns))].copy()

    assert mechanism_cuts_from_pruned_replay(pruned) == mechanism_cuts(full)


def test_pruned_mechanism_cuts_still_require_consumed_scalar_columns() -> None:
    full = _frame()
    pruned = full.loc[:, [
        *OOF_V8_REQUIRED_COLUMNS,
        "occupancy_entropy",
        "motion_magnitude",
    ]].copy()

    with pytest.raises(ValueError, match="guard_margin"):
        mechanism_cuts_from_pruned_replay(pruned)


def test_general_gate_and_aggregate_schema_fail_closed() -> None:
    bootstrap = {"delta_candidate_minus_reference": {"probability_candidate_lower_mid": 0.95}}
    decision = general_gate(
        candidate_metrics={
            "sequence_macro_MiD": 97.0,
            "finite_fraction": 1.0,
            "failure_rate_pct": 0.0,
            "coverage": 0.99,
        },
        baseline_metrics={"sequence_macro_MiD": 101.0, "coverage": 1.0},
        bootstrap=bootstrap,
        integrity_checks={name: True for name in REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS},
    )
    assert decision["passed"]
    incomplete = general_gate(
        candidate_metrics={
            "sequence_macro_MiD": 97.0,
            "finite_fraction": 1.0,
            "failure_rate_pct": 0.0,
            "coverage": 0.99,
        },
        baseline_metrics={"sequence_macro_MiD": 101.0, "coverage": 1.0},
        bootstrap=bootstrap,
        integrity_checks={},
    )
    assert not incomplete["passed"]
    assert set(incomplete["missing_integrity_checks"]) == REQUIRED_GENERAL_GATE_INTEGRITY_CHECKS
    payload = {key: {} for key in AGGREGATE_V8_REQUIRED_KEYS}
    for key in (
        "protocol_sha256",
        "config_sha256",
        "row_identity_sha256",
        "target_sha256",
        "prediction_sha256",
        "checkpoint_sha256",
        "artifact_sha256",
    ):
        payload[key] = _sha(key)
    payload.update(
        schema_version="scientific_recovery_v8_aggregate_v1",
        status="completed",
        git_commit="f9331b2",
        seed=7,
        folds=[0, 1, 2],
        metrics={},
        per_sequence={},
        per_bucket={},
        bootstrap={},
        integrity_checks={},
        gate_decision={},
    )
    sign_artifact(payload)
    validate_aggregate_payload(payload)
    payload["status"] = "tampered"
    with pytest.raises(ValueError, match="signature mismatch"):
        validate_aggregate_payload(payload)
    payload["status"] = "completed"
    sign_artifact(payload)
    payload["folds"] = []
    with pytest.raises(ValueError, match="non-empty"):
        validate_aggregate_payload(payload)


def test_factorial_replay_executes_one_real_checkpoint_graph_per_cell() -> None:
    """A factorial autopsy must obtain every cell by model replay, not CSV edits."""

    from e_jepa_ttc.evaluation.scientific_recovery_v8 import replay_factorial_a5

    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=8,
            geometry_dim=8,
            residual_depth=1,
            dropout=0.0,
            foreground_temporal_smoothing_mode="causal_left",
            transport_enabled=True,
            transport_radius=1,
        )
    ).eval()
    events = torch.rand(2, 3, 2, 16, 16)
    delta_t_s = torch.full((2, 2), 0.01)

    cells = replay_factorial_a5(model, events, delta_t_s)

    assert tuple(cells) == (
        "analytic_only",
        "analytic_residual",
        "analytic_transport",
        "analytic_residual_transport",
        "full",
    )
    assert all(value.ttc_mean_seconds.shape == (2,) for value in cells.values())
    assert torch.allclose(
        cells["analytic_only"].residual_log_height_ratio,
        torch.zeros_like(cells["analytic_only"].residual_log_height_ratio),
    )
    assert torch.allclose(
        cells["analytic_transport"].residual_log_height_ratio,
        torch.zeros_like(cells["analytic_transport"].residual_log_height_ratio),
    )
    assert torch.allclose(
        cells["full"].residual_log_height_ratio,
        model(events, delta_t_s).residual_log_height_ratio,
    )


def test_replay_cli_runner_loads_checkpoint_and_writes_signed_factorial_csvs(
    tmp_path: Path,
) -> None:
    """The runnable replay path must consume tensors plus checkpoint, never an OOF CSV."""

    torch = pytest.importorskip("torch")
    from e_jepa_ttc.evaluation.scientific_recovery_v8 import canonical_json_sha256

    module_spec = importlib.util.spec_from_file_location(
        "v8_replay", Path("scripts/replay_scientific_recovery_v8_mechanisms.py")
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config = CausalScaleTTCConfig(
        in_channels=2,
        hidden_dim=8,
        geometry_dim=8,
        residual_depth=1,
        dropout=0.0,
        foreground_temporal_smoothing_mode="causal_left",
        transport_enabled=True,
        transport_radius=1,
    )
    model = CausalScaleTTC(config).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model_config": config.__dict__, "model_state_dict": model.state_dict()}, checkpoint
    )
    replay_input = tmp_path / "frozen_event_rows.pt"
    torch.save(
        {
            "events": torch.rand(2, 3, 2, 16, 16),
            "delta_t_s": torch.full((2, 2), 0.01),
            "target_ttc": torch.tensor([2.0, 4.0]),
            "sample_weight": torch.ones(2),
            "token_id": ["a", "b"],
            "sequence_id": ["sequence-a", "sequence-b"],
            "track_id": ["track-a", "track-b"],
            "outer_fold": [0, 1],
            "seed": [7, 7],
            "endpoint_us": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        },
        replay_input,
    )
    manifest = module.run_replay(
        checkpoint=checkpoint,
        replay_input=replay_input,
        output_dir=tmp_path / "replay",
        model_name="a5",
        config_sha256=canonical_json_sha256(model.checkpoint_config()),
    )
    assert manifest["artifact_sha256"]
    assert len(manifest["factorial_cells"]) == 5
    assert (tmp_path / "replay" / "factorial_full.csv").is_file()


def test_sharded_replay_streams_signed_parts_without_fold_tensor(tmp_path: Path) -> None:
    """Production autopsy replay must accept small signed parts instead of one fold-sized tensor."""

    torch = pytest.importorskip("torch")
    from e_jepa_ttc.artifacts.hashing import sign_artifact
    from e_jepa_ttc.evaluation.scientific_recovery_v8 import canonical_json_sha256, sha256_file

    module_spec = importlib.util.spec_from_file_location(
        "v8_replay_sharded", Path("scripts/replay_scientific_recovery_v8_mechanisms.py")
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    config = CausalScaleTTCConfig(
        in_channels=2,
        hidden_dim=8,
        geometry_dim=8,
        residual_depth=1,
        dropout=0.0,
        foreground_temporal_smoothing_mode="causal_left",
        transport_enabled=True,
        transport_radius=1,
    )
    model = CausalScaleTTC(config).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model_config": config.__dict__, "model_state_dict": model.state_dict()}, checkpoint
    )

    part_entries = []
    for part_index in range(2):
        part = tmp_path / f"fold0.part{part_index:04d}.pt"
        start = part_index * 2
        torch.save(
            {
                "events": torch.rand(2, 3, 2, 16, 16).half(),
                "delta_t_s": torch.full((2, 2), 0.01),
                "target_ttc": torch.tensor([2.0 + start, 4.0 + start]),
                "sample_weight": torch.ones(2),
                "token_id": [f"token-{start}", f"token-{start + 1}"],
                "sequence_id": ["sequence-a", "sequence-a"],
                "track_id": [f"track-{start}", f"track-{start + 1}"],
                "outer_fold": [0, 0],
                "seed": [7, 7],
                "endpoint_us": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            },
            part,
        )
        part_entries.append(
            {
                "part": part_index,
                "path": part.name,
                "sha256": sha256_file(part),
                "rows": 2,
                "first_token": f"token-{start}",
                "last_token": f"token-{start + 1}",
            }
        )
    input_manifest = {
        "artifact_type": "scientific_recovery_v8_autopsy_replay_input_sharded_v2",
        "protocol_artifact_sha256": "0" * 64,
        "outer_fold": 0,
        "rows": 4,
        "chunk_rows": 2,
        "parts": part_entries,
        "sealed_splits_opened": False,
    }
    sign_artifact(input_manifest)
    manifest_path = tmp_path / "fold0.manifest.json"
    manifest_path.write_text(json.dumps(input_manifest), encoding="utf-8")

    result = module.run_replay_sharded(
        checkpoint=checkpoint,
        input_manifest_path=manifest_path,
        output_dir=tmp_path / "replay",
        model_name="a5",
        config_sha256=canonical_json_sha256(model.checkpoint_config()),
    )
    assert result["artifact_sha256"]
    assert result["replay_input"]["parts"] == 2
    assert result["replay_input"]["rows"] == 4
    frame = pd.read_csv(tmp_path / "replay" / "baseline.csv")
    assert len(frame) == 4
    assert set(frame["token_id"]) == {"token-0", "token-1", "token-2", "token-3"}



def test_replay_accepts_historical_checkpoint_config_with_new_default_fields(tmp_path: Path) -> None:
    """Raw historical config hash must survive backward-compatible dataclass defaults."""

    torch = pytest.importorskip("torch")
    from e_jepa_ttc.artifacts.hashing import sign_artifact
    from e_jepa_ttc.evaluation.scientific_recovery_v8 import canonical_json_sha256, sha256_file

    module_spec = importlib.util.spec_from_file_location(
        "v8_replay_historical_config", Path("scripts/replay_scientific_recovery_v8_mechanisms.py")
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    config = CausalScaleTTCConfig(
        in_channels=2,
        hidden_dim=8,
        geometry_dim=8,
        residual_depth=1,
        dropout=0.0,
        foreground_temporal_smoothing_mode="causal_left",
        transport_enabled=True,
        transport_radius=1,
    )
    model = CausalScaleTTC(config).eval()
    raw_config = dict(config.__dict__)
    # ``transport_mode`` was added after some historical A5/C2F checkpoints;
    # its default reproduces the old behavior exactly.
    raw_config.pop("transport_mode")
    checkpoint = tmp_path / "historical_checkpoint.pt"
    torch.save(
        {"model_config": raw_config, "model_state_dict": model.state_dict()}, checkpoint
    )

    part = tmp_path / "fold0.part0000.pt"
    torch.save(
        {
            "events": torch.rand(2, 3, 2, 16, 16).half(),
            "delta_t_s": torch.full((2, 2), 0.01),
            "target_ttc": torch.tensor([2.0, 4.0]),
            "sample_weight": torch.ones(2),
            "token_id": ["token-0", "token-1"],
            "sequence_id": ["sequence-a", "sequence-a"],
            "track_id": ["track-0", "track-1"],
            "outer_fold": [0, 0],
            "seed": [7, 7],
            "endpoint_us": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        },
        part,
    )
    manifest = {
        "artifact_type": "scientific_recovery_v8_autopsy_replay_input_sharded_v2",
        "protocol_artifact_sha256": "0" * 64,
        "outer_fold": 0,
        "rows": 2,
        "chunk_rows": 2,
        "parts": [
            {
                "part": 0,
                "path": part.name,
                "sha256": sha256_file(part),
                "rows": 2,
                "first_token": "token-0",
                "last_token": "token-1",
            }
        ],
        "sealed_splits_opened": False,
    }
    sign_artifact(manifest)
    manifest_path = tmp_path / "fold0.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = module.run_replay_sharded(
        checkpoint=checkpoint,
        input_manifest_path=manifest_path,
        output_dir=tmp_path / "replay",
        model_name="a5",
        config_sha256=canonical_json_sha256(raw_config),
    )
    assert result["config_sha256"] == canonical_json_sha256(raw_config)
    assert (
        result["config_provenance"]["raw_checkpoint_config_sha256"]
        == canonical_json_sha256(raw_config)
    )
    assert (
        result["config_provenance"]["effective_model_config_sha256"]
        == canonical_json_sha256(model.checkpoint_config())
    )

def test_autopsy_materializer_detects_pruned_historical_shards_and_builds_storage_only_recovery_config(tmp_path: Path) -> None:
    """Missing historical shards must trigger a train-only, value-equivalent recovery cache."""

    module_spec = importlib.util.spec_from_file_location(
        "v8_autopsy_materialize_recovery",
        Path("scripts/materialize_scientific_recovery_v8_autopsy_inputs.py"),
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "uses_official_garl_ttc_labels": True,
        "shards": [
            {"split": "train", "path": "train/shard-00000.pt", "count": 32},
            {"split": "validation", "path": "validation/shard-00000.pt", "count": 5},
        ],
        "config": {
            "full_width": 160,
            "full_height": 90,
            "full_bins": 5,
            "roi_size": 128,
            "roi_bins": 10,
            "shard_size": 256,
            "store_full_frame_events": False,
            "store_garl_event_roi": False,
            "store_jepa_event_roi": False,
            "store_event_v4_common_roi": True,
            "event_v4_margin_fraction": 0.25,
            "event_v4_require_precontext": True,
            "event_v4_precontext_fallback": "shifted_event_window",
            "event_v4_bins_per_polarity": 5,
            "event_v4_storage_dtype": "float32",
            "materialize_splits": ["train", "validation"],
            "include_rgb": False,
            "include_masks": False,
            "mask_required": False,
            "jepa_roi_bins": 5,
            "expected_train_rows": 88744,
            "allow_dataset_version_change": False,
            "target_delta_t_s": 0.1,
            "delta_t_tolerance_s": 0.025,
            "jepa_context_delta_t_s": 0.1,
            "jepa_context_tolerance_s": 0.05,
            "calibration_mode": "official_constant_fy",
            "selection_seed": 7,
            "event_pixel_diff": 5,
            "preprocessing_device": "cpu",
            "workers": 4,
            "compression": "none",
            "compression_level": 1,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    missing = module._missing_split_shards(manifest_path)
    assert missing == [tmp_path / "train" / "shard-00000.pt"]

    recovered = module._recovery_config(manifest, shard_size=32)
    assert recovered.materialize_splits == ("train",)
    assert recovered.shard_size == 32
    assert recovered.workers == 4
    assert recovered.roi_size == 128
    assert recovered.event_v4_bins_per_polarity == 5
    assert recovered.event_v4_margin_fraction == 0.25
    assert recovered.event_v4_precontext_fallback == "shifted_event_window"
    assert recovered.event_v4_storage_dtype == "float32"
    assert recovered.event_pixel_diff == 5


def test_autopsy_materializer_prefers_existing_train_shards(tmp_path: Path) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "v8_autopsy_materialize_existing",
        Path("scripts/materialize_scientific_recovery_v8_autopsy_inputs.py"),
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    shard = tmp_path / "train" / "shard-00000.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"present")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"shards": [{"split": "train", "path": "train/shard-00000.pt", "count": 1}]}),
        encoding="utf-8",
    )
    assert module._missing_split_shards(manifest_path) == []


def test_autopsy_materializer_loads_exact_sha_pinned_v7_population() -> None:
    module_spec = importlib.util.spec_from_file_location(
        "v8_autopsy_materialize_population",
        Path("scripts/materialize_scientific_recovery_v8_autopsy_inputs.py"),
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    protocol = json.loads(
        Path("configs/protocol/scientific_recovery_v8_temporal.json").read_text(encoding="utf-8")
    )
    rows = module._frozen_a5_rows(protocol)
    assert len(rows) == 8192
    assert {row["fold"] for row in rows.values()} == {0, 1, 2}
    assert all(row["seed"] == 7 for row in rows.values())


def test_future_prefix_invariance_uses_same_shape_future_perturbation() -> None:
    torch = pytest.importorskip("torch")
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=8,
            geometry_dim=8,
            residual_depth=1,
            dropout=0.0,
            foreground_temporal_smoothing=0.15,
            foreground_temporal_smoothing_mode="causal_left",
            transport_enabled=False,
        )
    ).eval()
    events = torch.rand(2, 3, 2, 16, 16)
    delta_t_s = torch.full((2, 2), 0.1)
    assert_causal_prefix_invariance(model, events, delta_t_s)


def test_future_prefix_invariance_rejects_symmetric_legacy_smoothing() -> None:
    torch = pytest.importorskip("torch")
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=8,
            geometry_dim=8,
            residual_depth=1,
            dropout=0.0,
            foreground_temporal_smoothing=0.15,
            foreground_temporal_smoothing_mode="symmetric_legacy",
            transport_enabled=False,
        )
    ).eval()
    events = torch.rand(1, 3, 2, 16, 16)
    delta_t_s = torch.full((1, 2), 0.1)
    with pytest.raises(ValueError, match="rejects symmetric_legacy"):
        assert_causal_prefix_invariance(model, events, delta_t_s)

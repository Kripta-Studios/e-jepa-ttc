"""Read-only feature replay for the preregistered E-Clock X0.5 campaign."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.evaluation.collision_clock_config import load_x0_config
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    load_signed_json,
    precheck_production_oof,
    require_reference_family,
    validate_protocol_reference_binding,
)
from e_jepa_ttc.evaluation.incremental_fusion import (
    DYNAMIC_SLOT_NAMES,
    FEATURE_COLUMNS,
    atomic_write_json,
)
from e_jepa_ttc.models.collision_clock_features import (
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
)
from e_jepa_ttc.models.collision_clock_ttc import (
    CollisionClockConfig,
    X0HeightBypassDirectPhase,
)
from e_jepa_ttc.training.collision_clock_eap import require_frozen_checkpoint

EXPECTED_X0_BUNDLE_SHA256 = "b8b228a6f0ea039238cf96d046228759d1a64f65fdda02b16353d63450beb7f9"
EXPECTED_X0_COMMIT = "57865bea943f7c1518341003170cec1c221aa093"
EXPECTED_X0_PROVENANCE_SHA256 = "7b8cd70f9dce14c531994bfe5adda18e389bff000329f49c1c341826f176f59d"
REPLAY_PHASE_ATOL = 0.0


def _configure_replay_fp32() -> None:
    """Restore the exact FP32 math policy active during X0 train/evaluation."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_x0_provenance_exception(
    campaign: Path, *, x0_bundle: Path | None = None
) -> dict[str, Any]:
    """Load the one exact signed BASE/DYN cross-commit adoption record."""

    path = campaign / "provenance_exception.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError("X0 cross-commit provenance signature mismatch")
    if (
        payload.get("artifact_type") != "eclock_x0_cross_commit_reuse_provenance_v1"
        or payload.get("artifact_sha256") != EXPECTED_X0_PROVENANCE_SHA256
        or payload.get("current_finalization_commit") != EXPECTED_X0_COMMIT
        or payload.get("source_training_commit") != "dc17643e4710b9fc525e3a8fdfb538e1eafeda76"
        or payload.get("reused_arms") != ["X0-BASE-U", "X0-DYN-U"]
        or payload.get("new_training_arms") != ["X0-PAIR-U"]
        or payload.get("official_external_replay_arms") != ["X0-A5-REPLAY"]
        or payload.get("source_artifacts_copied_byte_for_byte") is not True
        or payload.get("training_surface_git_diff_empty") is not True
        or payload.get("aggregates_and_comparisons_must_be_recomputed") is not True
        or payload.get("token_universe_count") != 8192
        or set(payload.get("arms", {})) != {"X0-BASE-U", "X0-DYN-U"}
    ):
        raise ValueError("X0 cross-commit provenance identity mismatch")
    if x0_bundle is not None:
        with zipfile.ZipFile(x0_bundle) as archive:
            if archive.read("provenance_exception.json") != raw:
                raise ValueError("X0 campaign provenance differs from the pinned bundle member")
    return payload


def _reused_fold_binding(provenance: dict[str, Any], arm: str, fold: int) -> dict[str, Any] | None:
    if arm not in provenance["reused_arms"]:
        return None
    arm_record = provenance["arms"].get(arm)
    if not isinstance(arm_record, dict):
        raise ValueError(f"X0 provenance lacks reused arm binding: {arm}")
    matches = [item for item in arm_record.get("folds", []) if item.get("outer_fold") == fold]
    if len(matches) != 1:
        raise ValueError(f"X0 provenance lacks unique fold binding: {arm}/fold-{fold}")
    return matches[0]


def _validate_reused_fold_binding(
    *,
    summary_path: Path,
    oof_path: Path,
    summary: dict[str, Any],
    binding: dict[str, Any],
    source_training_commit: str,
) -> None:
    """Require byte-level equality with every field signed by the X0 adoption."""

    checks = {
        "source commit": summary.get("git_commit") == source_training_commit,
        "summary file SHA": compute_file_hash(str(summary_path))
        == binding.get("fold_summary_file_sha256"),
        "summary artifact SHA": summary.get("artifact_sha256")
        == binding.get("fold_summary_sha256"),
        "OOF file SHA": compute_file_hash(str(oof_path)) == binding.get("oof_file_sha256"),
        "OOF bytes": oof_path.stat().st_size == binding.get("oof_bytes"),
        "summary OOF SHA": summary.get("oof_file_sha256") == binding.get("oof_file_sha256"),
        "checkpoint SHA": summary.get("checkpoint_file_sha256")
        == binding.get("checkpoint_file_sha256"),
        "checkpoint bytes": summary.get("checkpoint_bytes") == binding.get("checkpoint_bytes"),
        "checkpoint manifest SHA": summary.get("checkpoint_manifest_sha256")
        == binding.get("checkpoint_manifest_sha256"),
        "row count": summary.get("row_count") == binding.get("row_count"),
        "outer fold": summary.get("outer_fold") == binding.get("outer_fold"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("X0 reused fold differs from signed provenance: " + ", ".join(failed))


def _direct_model(config: dict[str, Any]) -> X0HeightBypassDirectPhase:
    values = dict(config["model"])
    values["feature_source"] = config["feature_source"]
    values["motion_feature_mode"] = config["motion_feature_mode"]
    clock = CollisionClockConfig(**values)
    encoder = HeightBypassEndpointEncoder(
        HeightBypassEncoderConfig(
            in_channels=clock.in_channels,
            hidden_dim=clock.encoder_hidden_dim,
            token_dim=clock.encoder_token_dim,
            residual_depth=clock.residual_depth,
            dropout=clock.dropout,
        )
    )
    return X0HeightBypassDirectPhase(encoder, clock)


def _load_frozen_model(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    device: torch.device,
    arm: str,
    fold: int,
    binding: dict[str, Any],
    source_training_commit: str,
) -> X0HeightBypassDirectPhase:
    manifest = require_frozen_checkpoint(checkpoint)
    if (
        manifest.get("completed_updates") != 6840
        or manifest.get("artifact_sha256") != binding.get("checkpoint_manifest_sha256")
        or manifest.get("checkpoint_file_sha256") != binding.get("checkpoint_file_sha256")
        or manifest.get("checkpoint_bytes") != binding.get("checkpoint_bytes")
        or manifest.get("scientific_identity", {}).get("git_commit_observed")
        != source_training_commit
        or manifest.get("scientific_identity", {}).get("arm_id") != arm
        or manifest.get("scientific_identity", {}).get("outer_fold") != fold
    ):
        raise ValueError("X0 replay checkpoint differs from signed cross-commit provenance")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != "eclock_x0_checkpoint_v2":
        raise ValueError("X0 checkpoint payload type mismatch")
    model = _direct_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def _read_signed_fold_oof(
    campaign: Path, arm: str, *, provenance: dict[str, Any]
) -> tuple[pd.DataFrame, dict[int, str]]:
    frames: list[pd.DataFrame] = []
    checkpoints: dict[int, str] = {}
    for fold in (0, 1, 2):
        root = campaign / arm / f"fold-{fold}"
        summary_path = root / "fold_summary.json"
        oof_path = root / "oof_predictions.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not verify_artifact_hash(summary):
            raise ValueError(f"X0 fold summary signature mismatch: {arm}/fold-{fold}")
        if (
            summary.get("arm_id") != arm
            or summary.get("outer_fold") != fold
            or summary.get("outer_dev_evaluations") != 1
            or compute_file_hash(str(oof_path)) != summary.get("oof_file_sha256")
        ):
            raise ValueError(f"X0 fold OOF identity mismatch: {arm}/fold-{fold}")
        binding = _reused_fold_binding(provenance, arm, fold)
        if binding is None:
            if summary.get("git_commit") != EXPECTED_X0_COMMIT:
                raise ValueError(f"X0 fold commit mismatch: {arm}/fold-{fold}")
        else:
            _validate_reused_fold_binding(
                summary_path=summary_path,
                oof_path=oof_path,
                summary=summary,
                binding=binding,
                source_training_commit=str(provenance["source_training_commit"]),
            )
        frame = pd.read_csv(oof_path, float_precision="round_trip")
        if set(frame["outer_fold"].astype(int)) != {fold}:
            raise ValueError(f"X0 OOF fold contamination: {arm}/fold-{fold}")
        frames.append(frame)
        checkpoints[fold] = str(summary["checkpoint_file_sha256"])
    return pd.concat(frames, ignore_index=True), checkpoints


def _checked_x0_oof(
    *,
    repo: Path,
    campaign: Path,
    arm: str,
    config_name: str,
    protocol: dict[str, Any],
    reference: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, str]]:
    config_path = repo / "configs" / "experiment" / "scientific_recovery_v9_eclock" / config_name
    load_x0_config(
        config_path,
        schema_path=repo / "schemas" / "scientific_recovery_v9_eclock_config_v2.schema.json",
    )
    binding = provenance["arms"].get(arm)
    if isinstance(binding, dict) and binding.get("config_file_sha256") != compute_file_hash(
        str(config_path)
    ):
        raise ValueError(f"X0 reused config differs from signed provenance: {arm}")
    frame, checkpoints = _read_signed_fold_oof(campaign, arm, provenance=provenance)
    checked = precheck_production_oof(
        frame,
        protocol=protocol,
        reference=reference,
        arm_id=arm,
        config_sha256=compute_file_hash(str(config_path)),
        checkpoint_sha256_by_fold=checkpoints,
    )
    return checked, checkpoints


def _source_hashes(repo: Path) -> dict[str, str]:
    relative_paths = (
        "src/e_jepa_ttc/data/collision_clock_cache.py",
        "src/e_jepa_ttc/models/collision_clock_features.py",
        "src/e_jepa_ttc/models/collision_clock_motion.py",
        "src/e_jepa_ttc/models/collision_clock_math.py",
        "src/e_jepa_ttc/models/collision_clock_ttc.py",
        "src/e_jepa_ttc/models/local_transport.py",
        "src/e_jepa_ttc/evaluation/collision_clock_protocol.py",
    )
    return {path: compute_file_hash(str(repo / Path(path))) for path in relative_paths}


def run_feature_replay(
    *,
    repo: Path,
    reference_root: Path,
    x0_campaign: Path,
    cache_root: Path,
    x0_bundle: Path,
    output_root: Path,
    device: torch.device,
    cache_mode: str = "shard_lru",
) -> dict[str, Any]:
    """Replay BASE/DYN once, hold slots until exact OOF equality, then sign them."""

    started = time.perf_counter()
    _configure_replay_fp32()
    if compute_file_hash(str(x0_bundle)) != EXPECTED_X0_BUNDLE_SHA256:
        raise ValueError("mandatory X0 bundle SHA-256 mismatch")
    provenance = load_x0_provenance_exception(x0_campaign, x0_bundle=x0_bundle)
    protocol_path = repo / "configs" / "protocol" / "scientific_recovery_v9_eclock_x0.json"
    reference_path = (
        repo / "configs" / "protocol" / "scientific_recovery_v9_eclock_x0_reference.json"
    )
    protocol = load_signed_json(
        protocol_path,
        schema_path=repo / "schemas" / "scientific_recovery_v9_eclock_protocol_v2.schema.json",
    )
    reference = load_signed_json(
        reference_path,
        schema_path=repo / "schemas" / "scientific_recovery_v9_eclock_reference_v2.schema.json",
    )
    validate_protocol_reference_binding(protocol, reference, protocol_path=protocol_path)

    arm_specs = {
        "X0-A5-REPLAY": "x0_a5_replay.yaml",
        "X0-BASE-U": "x0_base_u.yaml",
        "X0-DYN-U": "x0_dyn_u.yaml",
        "X0-PAIR-U": "x0_pair_u.yaml",
    }
    checked: dict[str, pd.DataFrame] = {}
    checkpoint_maps: dict[str, dict[int, str]] = {}
    for arm, config_name in arm_specs.items():
        checked[arm], checkpoint_maps[arm] = _checked_x0_oof(
            repo=repo,
            campaign=x0_campaign,
            arm=arm,
            config_name=config_name,
            protocol=protocol,
            reference=reference,
            provenance=provenance,
        )
    canonical = (
        checked["X0-A5-REPLAY"].sort_values("sample_token", kind="stable").reset_index(drop=True)
    )
    for arm, frame in checked.items():
        other = frame.sort_values("sample_token", kind="stable").reset_index(drop=True)
        for column in ("sample_token", "sequence_id", "track_id", "outer_fold"):
            if not np.array_equal(canonical[column].astype(str), other[column].astype(str)):
                raise ValueError(f"X0 cross-arm identity mismatch: {arm}/{column}")
        for column in ("target_ttc_s", "target_benchmark_phase", "sample_weight"):
            if not np.array_equal(
                canonical[column].to_numpy(dtype=np.float64),
                other[column].to_numpy(dtype=np.float64),
            ):
                raise ValueError(f"X0 cross-arm numeric identity mismatch: {arm}/{column}")

    base_config = load_x0_config(
        repo / "configs/experiment/scientific_recovery_v9_eclock/x0_base_u.yaml",
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_config_v2.schema.json",
    )
    dyn_config = load_x0_config(
        repo / "configs/experiment/scientific_recovery_v9_eclock/x0_dyn_u.yaml",
        schema_path=repo / "schemas/scientific_recovery_v9_eclock_config_v2.schema.json",
    )
    adapter = CollisionClockTrain8192Cache(
        cache_root,
        protocol,
        cache_mode=cache_mode,  # type: ignore[arg-type]
        canonical_supervision=load_canonical_supervision(reference, reference_root),
    )
    adapter.verify_and_index()
    replay_rows: list[dict[str, Any]] = []
    max_base_error = 0.0
    max_dyn_error = 0.0
    with torch.no_grad():
        for fold in (0, 1, 2):
            _train_view, dev_view = adapter.outer_views(fold)
            base_checkpoint = (
                x0_campaign / "X0-BASE-U" / f"fold-{fold}" / "milestones" / "update-006840.pt"
            )
            dyn_checkpoint = (
                x0_campaign / "X0-DYN-U" / f"fold-{fold}" / "milestones" / "update-006840.pt"
            )
            base_binding = _reused_fold_binding(provenance, "X0-BASE-U", fold)
            dyn_binding = _reused_fold_binding(provenance, "X0-DYN-U", fold)
            if base_binding is None or dyn_binding is None:
                raise ValueError("signed X0 provenance must bind BASE/DYN replay checkpoints")
            base_model = _load_frozen_model(
                base_config,
                base_checkpoint,
                device=device,
                arm="X0-BASE-U",
                fold=fold,
                binding=base_binding,
                source_training_commit=str(provenance["source_training_commit"]),
            )
            dyn_model = _load_frozen_model(
                dyn_config,
                dyn_checkpoint,
                device=device,
                arm="X0-DYN-U",
                fold=fold,
                binding=dyn_binding,
                source_training_commit=str(provenance["source_training_commit"]),
            )
            expected_base = checked["X0-BASE-U"].set_index("sample_token")
            expected_dyn = checked["X0-DYN-U"].set_index("sample_token")
            expected_a5 = checked["X0-A5-REPLAY"].set_index("sample_token")
            expected_pair = checked["X0-PAIR-U"].set_index("sample_token")
            adapter.stage_view(dev_view)
            try:
                for batch in adapter.iter_outer_dev_batches(dev_view, batch_size=32):
                    inputs = batch.inputs.to(device)
                    deltas = batch.delta_t_s.to(device)
                    base_output = base_model(inputs, deltas)
                    dyn_output = dyn_model(inputs, deltas)
                    base_phase = (
                        base_output.benchmark_phase_mean.detach().cpu().numpy().astype(np.float64)
                    )
                    dyn_phase = (
                        dyn_output.benchmark_phase_mean.detach().cpu().numpy().astype(np.float64)
                    )
                    slots = (
                        dyn_output.diagnostics["global_transport_12_observed"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    if slots.shape != (len(batch.sample_tokens), 9) or not np.isfinite(slots).all():
                        raise ValueError("DYN replay did not export nine finite m12 slots")
                    for index, token in enumerate(batch.sample_tokens):
                        base_expected = float(expected_base.loc[token, "predicted_benchmark_phase"])
                        dyn_expected = float(expected_dyn.loc[token, "predicted_benchmark_phase"])
                        base_error = abs(float(base_phase[index]) - base_expected)
                        dyn_error = abs(float(dyn_phase[index]) - dyn_expected)
                        max_base_error = max(max_base_error, base_error)
                        max_dyn_error = max(max_dyn_error, dyn_error)
                        row = expected_a5.loc[token]
                        slot_values = {
                            name: float(slots[index, slot_index])
                            for slot_index, name in enumerate(DYNAMIC_SLOT_NAMES)
                        }
                        replay_rows.append(
                            {
                                "sample_token": token,
                                "sequence_id": str(row["sequence_id"]),
                                "track_id": str(row["track_id"]),
                                "outer_fold": fold,
                                "target_ttc_s": float(row["target_ttc_s"]),
                                "target_benchmark_phase": float(row["target_benchmark_phase"]),
                                "sample_weight": float(row["sample_weight"]),
                                "a5_predicted_benchmark_phase": float(
                                    row["predicted_benchmark_phase"]
                                ),
                                "base_predicted_benchmark_phase": base_expected,
                                "dyn_predicted_benchmark_phase": dyn_expected,
                                "pair_predicted_benchmark_phase": float(
                                    expected_pair.loc[token, "predicted_benchmark_phase"]
                                ),
                                **slot_values,
                                "transport_valid": True,
                                "a5_checkpoint_sha256": checkpoint_maps["X0-A5-REPLAY"][fold],
                                "base_checkpoint_sha256": checkpoint_maps["X0-BASE-U"][fold],
                                "dyn_checkpoint_sha256": checkpoint_maps["X0-DYN-U"][fold],
                                "pair_checkpoint_sha256": checkpoint_maps["X0-PAIR-U"][fold],
                                "x0_protocol_sha256": protocol["artifact_sha256"],
                                "x0_reference_sha256": reference["artifact_sha256"],
                                "cache_manifest_sha256": protocol["cache_binding"]["file_sha256"],
                                "split_manifest_sha256": protocol["split_binding"]["file_sha256"],
                            }
                        )
            finally:
                adapter.release_staged_view()
            del base_model, dyn_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(replay_rows) != 8192:
        raise ValueError("feature replay did not cover exactly 8,192 rows")
    if max_base_error > REPLAY_PHASE_ATOL or max_dyn_error > REPLAY_PHASE_ATOL:
        raise ValueError(
            f"X0 exact replay failed before slot consumption: base={max_base_error}, "
            f"dyn={max_dyn_error}, tolerance={REPLAY_PHASE_ATOL}"
        )
    feature_table = (
        pd.DataFrame(replay_rows)
        .sort_values("sample_token", kind="stable")
        .reset_index(drop=True)
        .loc[:, FEATURE_COLUMNS]
    )
    output_root.mkdir(parents=True, exist_ok=False)
    feature_path = output_root / "x05_feature_table.csv"
    feature_table.to_csv(feature_path, index=False)
    a5_family = require_reference_family(reference, "official_a5_oof")
    manifest = atomic_write_json(
        output_root / "x05_feature_replay_manifest.json",
        {
            "artifact_type": "eclock_x05_feature_replay_manifest_v1",
            "row_count": 8192,
            "finite_fraction": 1.0,
            "failure_rate": 0.0,
            "identity_hashes_exact": True,
            "replay_matches_x0": True,
            "replay_phase_atol": REPLAY_PHASE_ATOL,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "base_replay_max_abs_phase_error": max_base_error,
            "dyn_replay_max_abs_phase_error": max_dyn_error,
            "a5_replay_semantics": "sha_bound_official_oof_per_x0_protocol",
            "a5_official_prediction_sha256": a5_family["prediction_sha256"],
            "target_not_passed_to_extractor": True,
            "slot_source": "X0-DYN-U global_transport_12_observed m12 before any zeroing",
            "slot_names": list(DYNAMIC_SLOT_NAMES),
            "upstream_roi_is_box_conditioned": True,
            "feature_table_path": str(feature_path),
            "feature_table_file_sha256": compute_file_hash(str(feature_path)),
            "feature_table_bytes": feature_path.stat().st_size,
            "x0_bundle_sha256": EXPECTED_X0_BUNDLE_SHA256,
            "x0_source_commit": EXPECTED_X0_COMMIT,
            "x0_reused_training_commit": provenance["source_training_commit"],
            "x0_provenance_exception_sha256": provenance["artifact_sha256"],
            "x0_provenance_exception_file_sha256": compute_file_hash(
                str(x0_campaign / "provenance_exception.json")
            ),
            "x0_protocol_sha256": protocol["artifact_sha256"],
            "x0_reference_sha256": reference["artifact_sha256"],
            "checkpoint_sha256_by_arm_fold": {
                arm: {str(fold): sha for fold, sha in values.items()}
                for arm, values in checkpoint_maps.items()
            },
            "x0_source_module_sha256": _source_hashes(repo),
            "cache_mode": adapter.cache_mode,
            "cache_engineering": adapter.engineering_stats(),
            "wall_seconds": time.perf_counter() - started,
            "rows_per_second": 8192 / max(time.perf_counter() - started, 1.0e-12),
            "sealed_evaluation_opened": False,
        },
    )
    return manifest


__all__ = [
    "EXPECTED_X0_BUNDLE_SHA256",
    "EXPECTED_X0_COMMIT",
    "EXPECTED_X0_PROVENANCE_SHA256",
    "load_x0_provenance_exception",
    "run_feature_replay",
]

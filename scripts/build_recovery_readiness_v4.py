"""Build the auditable recovery-v4 training readiness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(path: Path, expected: str) -> bool:
    return path.exists() and _read_json(path).get("status") == expected


SSL_PURE_SAMPLING_FIELDS = (
    "uses_ttc_for_sampling",
    "uses_boxes_for_sampling",
    "uses_category_for_sampling",
    "uses_depth_for_sampling",
    "uses_masks_for_sampling",
    "uses_3d_for_sampling",
    "uses_future_labels_for_sampling",
)


def _ssl_pure_sampling_provenance(provenance: dict[str, Any]) -> bool:
    """Fail closed unless every SSL-Pure sampling flag is explicitly false."""

    return all(provenance.get(field) is False for field in SSL_PURE_SAMPLING_FIELDS)


def _quality_gate(quality: dict[str, Any]) -> bool:
    commands = quality.get("commands", {})
    required = {
        "git_diff_check",
        "ruff_check",
        "ruff_format_check",
        "pyright",
        "targeted_unit_tests",
        "integration_no_fabricated_evidence",
    }
    return required.issubset(commands) and all(
        commands[name].get("exit_code") == 0 for name in required
    )


def _contract_gate(manifest: dict[str, Any], audit: dict[str, Any]) -> bool:
    schema = manifest.get("input_schema", {})
    split_counts = manifest.get("split_counts", {})
    return bool(
        manifest.get("artifact_type") == "garlttc_official_lhr_object_cache_v4"
        and manifest.get("schema_version") == "garlttc_cache_v4"
        and schema.get("version") == "garlttc_input_v4"
        and schema.get("normalization") == "official_timevolume20_grid_sample_v1"
        and schema.get("event_roi_shape") == [2, 20, 128, 128]
        and split_counts.get("train") == 12
        and split_counts.get("validation") == 12
        and manifest.get("discard_count") == 0
        and manifest.get("discard_fraction") == 0.0
        and manifest.get("protocol_sha256")
        and manifest.get("jepa_pair_valid_fraction") == 1.0
        and manifest.get("no_label_fallback") is True
        and manifest.get("uses_official_garl_ttc_labels") is True
        and manifest.get("uses_reconstructed_public_eap_ttc") is False
        and audit.get("status") == "PASS"
        and not audit.get("errors")
    )


def _pilot_gate(root: Path, audit_path: Path) -> bool:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = _read_json(manifest_path)
    counts = manifest.get("split_counts", {})
    return (
        manifest.get("artifact_type") == "garlttc_official_lhr_object_cache_v4"
        and counts.get("train", 0) >= 4096
        and counts.get("validation", 0) >= 4096
        and manifest.get("discard_fraction") == 0.0
        and _status(audit_path, "PASS")
    )


def _compressed_cache_gate(
    root: Path,
    audit_path: Path,
    *,
    train_count: int,
    validation_count: int,
) -> bool:
    """Validate a gzip cache without loading every tensor into readiness."""

    manifest_path = root / "manifest.json"
    if not manifest_path.exists() or not audit_path.exists():
        return False
    manifest = _read_json(manifest_path)
    counts = manifest.get("split_counts", {})
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        return False
    if not (
        manifest.get("artifact_type") == "garlttc_official_lhr_object_cache_v4"
        and manifest.get("schema_version") == "garlttc_cache_v4"
        and manifest.get("shard_compression") == "gzip"
        and manifest.get("config", {}).get("compression") == "gzip"
        and counts == {"train": train_count, "validation": validation_count}
        and manifest.get("discard_count") == 0
        and manifest.get("discard_fraction") == 0.0
        and manifest.get("jepa_pair_valid_fraction") == 1.0
        and manifest.get("no_label_fallback") is True
        and manifest.get("uses_official_garl_ttc_labels") is True
        and manifest.get("uses_reconstructed_public_eap_ttc") is False
        and shards
    ):
        return False
    for shard in shards:
        relative = shard.get("path")
        if not isinstance(relative, str) or not (root / relative).is_file():
            return False
        sidecar = (root / relative).with_suffix("").with_suffix(".meta.json")
        if not sidecar.is_file():
            return False
    return _status(audit_path, "PASS")


def _partial_compressed_cache_counts(root: Path) -> dict[str, int]:
    """Count completed gzip shards in an in-progress split-directory cache.

    The cache writer deliberately keeps ``train`` and ``validation`` in
    separate directories and only publishes ``manifest.json`` after both
    splits finish.  Readiness must still report truthful partial progress
    while that terminal manifest is absent.
    """

    counts: dict[str, int] = {}
    for split in ("train", "validation"):
        split_root = root / split
        counts[split] = (
            sum(1 for path in split_root.glob("shard-*.pt.gz") if path.is_file())
            if split_root.is_dir()
            else 0
        )
    return counts


def _manifest_compressed_cache_counts(manifest: dict[str, Any]) -> dict[str, int]:
    """Count gzip shards by split from a terminal cache manifest."""

    counts = {"train": 0, "validation": 0}
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        return counts
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        relative = shard.get("path")
        if not isinstance(relative, str):
            continue
        split = Path(relative).parts[0] if Path(relative).parts else ""
        if split in counts:
            counts[split] += 1
    return counts


def _is_cuda_report(value: object) -> bool:
    """Accept canonical CUDA reports such as ``cuda`` and ``cuda:0``."""

    return isinstance(value, str) and (value == "cuda" or value.startswith("cuda:"))


def _ssl_full_train_gate(summary: dict[str, Any]) -> bool:
    """Accept only a terminal SSL artifact covering the complete train40 split."""

    trainer_config = summary.get("trainer_config", {})
    provenance = summary.get("provenance", {})
    return (
        summary.get("artifact_type") == "eap_ssl_on_demand_pretraining_v1"
        and _is_cuda_report(summary.get("device"))
        and summary.get("epochs_completed", 0) > 0
        and summary.get("selected_train_samples") == 16_384
        and summary.get("selected_validation_samples") == 4_096
        and trainer_config.get("max_train_samples") is None
        and trainer_config.get("max_validation_samples") is None
        and provenance.get("uses_ttc_labels") is False
        and provenance.get("uses_object_bboxes") is False
        and provenance.get("uses_depth_track_derivatives") is False
        and provenance.get("uses_rgb") is False
        and provenance.get("uses_labels_for_window_sampling") is False
        and _ssl_pure_sampling_provenance(provenance)
    )


def _ssl_perf_screen_gate(summary: dict[str, Any]) -> bool:
    """Validate a bounded SSL throughput screen without treating it as full data."""

    provenance = summary.get("provenance", {})
    trainer_config = summary.get("trainer_config", {})
    return (
        summary.get("artifact_type") == "eap_ssl_on_demand_pretraining_v1"
        and _is_cuda_report(summary.get("device"))
        and trainer_config.get("event_chunk_size") == 250000
        and summary.get("selected_train_samples") == 128
        and summary.get("selected_validation_samples") == 32
        and provenance.get("uses_ttc_labels") is False
        and provenance.get("uses_object_bboxes") is False
        and provenance.get("uses_rgb") is False
        and _ssl_pure_sampling_provenance(provenance)
        and summary.get("epochs_completed", 0) > 0
        and summary.get("best_validation_loss") is not None
    )


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def build_readiness(repo_root: Path, output: Path) -> dict[str, Any]:
    quality_path = repo_root / "artifacts/audits/recovery_v4/quality_gate.json"
    release_path = repo_root / "artifacts/audits/garl_release_v1/audit.json"
    preprocessing_path = repo_root / "artifacts/parity/garl_preprocessing_v2/parity.json"
    model_path = repo_root / "artifacts/parity/garl_model_v1/parity.json"
    smoke_manifest_path = repo_root / "artifacts/cache/garlttc_lhr_v4_smoke_workers4/manifest.json"
    smoke_audit_path = repo_root / "artifacts/audits/garlttc_lhr_v4_smoke_workers4_audit.json"
    pilot_root = repo_root / "artifacts/cache/garlttc_lhr_v4_pilot_4096_workers4"
    pilot_audit_path = repo_root / "artifacts/audits/garlttc_lhr_v4_pilot_4096_workers4_audit.json"
    resolution_path = repo_root / "artifacts/audits/patch_resolution_v1/patch_resolution_audit.json"
    token_scaling_path = repo_root / "artifacts/benchmarks/highres_token_scaling_v1.json"
    architecture_smoke_path = repo_root / "artifacts/benchmarks/highres_architecture_screen_v1.json"
    temporal_sweep_path = repo_root / "artifacts/benchmarks/highres_temporal_sweep_v1.json"
    real_screen_path = (
        repo_root / "artifacts/benchmarks/highres_real_screen_v1_gpu_workers4/screen.json"
    )
    highres_matrix_aggregate_path = (
        repo_root / "artifacts/benchmarks/highres_matrix_aggregate_v1.json"
    )
    lhr_smoke_summary_path = (
        repo_root / "artifacts/runs/garl_v4_lhr_smoke_seed7_shardlocal_modern/summary.json"
    )
    ssl_smoke_metrics_path = repo_root / "artifacts/runs/eap_ssl_smoke_current_seed7/metrics.json"
    ssl_cache_capacity_smoke_path = repo_root / (
        "artifacts/runs/eap_ssl_cache_capacity_smoke_current_seed7/metrics.json"
    )
    ssl_multiwindow_smoke_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_smoke_multiwindow_current_seed7/metrics.json"
    )
    ssl_multiwindow_perf_screen_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_multiwindow_perf_screen_128_current_seed7/metrics.json"
    )
    ssl_multiwindow_perf_workers2_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_multiwindow_perf_screen_128_workers2_current_seed7/metrics.json"
    )
    ssl_multiwindow_perf_workers4_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_multiwindow_perf_screen_128_workers4_current_seed7/metrics.json"
    )
    ssl_temporal_cache_screen_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_multiwindow_perf_screen_128_workers4_temporal_cache_after_interruption_seed7/metrics.json"
    )
    ssl_screen_metrics_path = (
        repo_root / "artifacts/runs/eap_ssl_train40_screen_seed7_workers2/metrics.json"
    )
    ssl_full_failure_path = repo_root / "artifacts/runs/eap_ssl_train40_current_seed7/FAILURE.json"
    ssl_screen_failure_path = repo_root / "artifacts/runs/eap_ssl_train40_screen_seed7/FAILURE.json"
    ssl_full_workers2_failure_path = (
        repo_root / "artifacts/runs/eap_ssl_train40_full_cuda_workers2_current_seed7/FAILURE.json"
    )
    ssl_full_workers0_metrics_path = (
        repo_root / "artifacts/runs/eap_ssl_train40_full_cuda_workers0_current_seed7/metrics.json"
    )
    ssl_full_workers0_failure_path = (
        repo_root / "artifacts/runs/eap_ssl_train40_full_cuda_workers0_current_seed7/FAILURE.json"
    )
    ssl_full_chunked_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers4_chunked_current_seed7/metrics.json"
    )
    ssl_full_chunked_failure_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers4_chunked_current_seed7/FAILURE.json"
    )
    ssl_full_previous_interruption_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers2_chunk250k_current_seed7/FAILURE.json"
    )
    ssl_full_multiwindow_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers2_chunk250k_multiwindow_current_seed7/metrics.json"
    )
    ssl_full_multiwindow_failure_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers2_chunk250k_multiwindow_current_seed7/FAILURE.json"
    )
    ssl_full_current_metrics_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers0_chunk250k_current_seed7/metrics.json"
    )
    ssl_full_current_failure_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers0_chunk250k_current_seed7/FAILURE.json"
    )
    ssl_full_workers4_current_failure_path = repo_root / (
        "artifacts/runs/eap_ssl_train40_full_cuda_workers4_chunk250k_current_seed7/FAILURE.json"
    )
    robustness_smoke_path = (
        repo_root / "artifacts/metrics/robustness_synthetic_smoke_current_v1.json"
    )
    phase12_smoke_path = repo_root / "artifacts/audits/plan_phase12_smoke_v1/phase12_smoke.json"
    runtime_smoke_path = (
        repo_root / "artifacts/demos/runtime_smoke_current_v1/runtime_smoke_metrics.json"
    )
    pilot_audit_current_path = (
        repo_root / "artifacts/audits/garlttc_lhr_v4_pilot_4096_workers4_audit_current_v1.json"
    )
    full_cache_failure_path = (
        repo_root / "artifacts/cache/garlttc_lhr_v4_full_train40_v1/FAILURE.json"
    )
    rotating_cache_plan_path = (
        repo_root / "artifacts/audits/garlttc_rotating_cache_plan_v1/plan.json"
    )
    gzip_smoke_root = repo_root / "artifacts/cache/garlttc_lhr_v4_smoke_gzip_workers1"
    gzip_smoke_audit_path = (
        repo_root / "artifacts/audits/garlttc_lhr_v4_smoke_gzip_workers1_audit.json"
    )
    gzip_full_root = repo_root / "artifacts/cache/garlttc_lhr_v4_full_train40_gzip_v5"
    gzip_full_audit_path = (
        repo_root / "artifacts/audits/garlttc_lhr_v4_full_train40_gzip_v5_audit.json"
    )
    release_cache_smoke_root = repo_root / "artifacts/cache/garl_release_input_smoke_rgb_v4"
    release_cache_smoke_manifest_path = release_cache_smoke_root / "manifest.json"
    release_cache_smoke_parity_path = (
        repo_root / "artifacts/audits/garl_release_input_smoke_rgb_v4_parity.json"
    )
    # v1/v2 are preserved timeout/memory negatives.  Only the supervised v3
    # build may satisfy the full release-input gate once its terminal manifest
    # and independent parity audit exist.
    release_cache_full_root = repo_root / "artifacts/cache/garl_release_input_full_rgb_v4"
    release_cache_full_manifest_path = release_cache_full_root / "manifest.json"
    release_cache_full_parity_path = (
        repo_root / "artifacts/audits/garl_release_input_full_rgb_v4_parity.json"
    )
    rgb_manifest_path = (
        repo_root / "artifacts/cache/garlttc_lhr_v4_smoke_rgb_workers1/manifest.json"
    )
    rgb_audit_path = repo_root / "artifacts/audits/garlttc_lhr_v4_smoke_rgb_workers1_audit.json"
    rgbe_summary_path = repo_root / "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/summary.json"
    foreground_failure_path = (
        repo_root / "artifacts/runs/garl_v4_lhr_foreground_smoke_seed7/FAILURE.json"
    )
    zero_shot_validation_path = (
        repo_root / "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/zero_shot_validation.json"
    )
    zero_shot_aggregate_path = (
        repo_root / "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/zero_shot_aggregate.json"
    )
    submission_path = (
        repo_root / "artifacts/official/garl_release_inference_local_smoke_v1/submission.json"
    )
    submission_validation_path = (
        repo_root
        / "artifacts/official/garl_release_inference_local_smoke_v1/submission_validation.json"
    )
    freeze_protocol_path = repo_root / "artifacts/audit/recovery_v3/frozen_protocol.json"
    rgb_cache_failure_path = (
        repo_root / "artifacts/runs/garl_v4_lhr_rgb_cache_workers4/FAILURE.json"
    )
    ssl_window_budget_path = repo_root / "artifacts/audits/eap_ssl_window_budget_v1.json"
    phase4_overfit_summary_path = (
        repo_root / "artifacts/runs/carla_jepa_overfit_32_current_seed7/metrics.json"
    )
    phase4_health_path = (
        repo_root / "artifacts/figures/carla_jepa_overfit_32_current_v1/embedding_health.json"
    )
    report_manifest_path = repo_root / "artifacts/tables/regenerable_report/report_manifest.json"
    plan_audit_path = repo_root / "artifacts/audits/plan_execution/PLAN_AUDIT.json"

    quality = _read_json(quality_path)
    # Cache material is intentionally removable.  Treat a missing smoke
    # manifest as a red, auditable gate instead of crashing the readiness
    # builder before it can report the missing evidence.
    smoke_manifest = _read_json(smoke_manifest_path) if smoke_manifest_path.exists() else {}
    resolution = _read_json(resolution_path)
    token_scaling = _read_json(token_scaling_path)
    architecture_smoke = _read_json(architecture_smoke_path)
    temporal_sweep = _read_json(temporal_sweep_path)
    real_screen = _read_json(real_screen_path)
    highres_matrix_aggregate = (
        _read_json(highres_matrix_aggregate_path) if highres_matrix_aggregate_path.exists() else {}
    )
    lhr_smoke = _read_json(lhr_smoke_summary_path) if lhr_smoke_summary_path.exists() else {}
    ssl_smoke = _read_json(ssl_smoke_metrics_path) if ssl_smoke_metrics_path.exists() else {}
    ssl_multiwindow_smoke = (
        _read_json(ssl_multiwindow_smoke_metrics_path)
        if ssl_multiwindow_smoke_metrics_path.exists()
        else {}
    )
    ssl_multiwindow_perf_screen = (
        _read_json(ssl_multiwindow_perf_screen_metrics_path)
        if ssl_multiwindow_perf_screen_metrics_path.exists()
        else {}
    )
    ssl_multiwindow_perf_workers2 = (
        _read_json(ssl_multiwindow_perf_workers2_metrics_path)
        if ssl_multiwindow_perf_workers2_metrics_path.exists()
        else {}
    )
    ssl_multiwindow_perf_workers4 = (
        _read_json(ssl_multiwindow_perf_workers4_metrics_path)
        if ssl_multiwindow_perf_workers4_metrics_path.exists()
        else {}
    )
    ssl_temporal_cache_screen = (
        _read_json(ssl_temporal_cache_screen_metrics_path)
        if ssl_temporal_cache_screen_metrics_path.exists()
        else {}
    )
    ssl_screen = _read_json(ssl_screen_metrics_path) if ssl_screen_metrics_path.exists() else {}
    ssl_full_failure = _read_json(ssl_full_failure_path) if ssl_full_failure_path.exists() else {}
    ssl_screen_failure = (
        _read_json(ssl_screen_failure_path) if ssl_screen_failure_path.exists() else {}
    )
    ssl_full_workers2_failure = (
        _read_json(ssl_full_workers2_failure_path)
        if ssl_full_workers2_failure_path.exists()
        else {}
    )
    ssl_full_workers0 = (
        _read_json(ssl_full_workers0_metrics_path)
        if ssl_full_workers0_metrics_path.exists()
        else {}
    )
    ssl_full_workers0_failure = (
        _read_json(ssl_full_workers0_failure_path)
        if ssl_full_workers0_failure_path.exists()
        else {}
    )
    ssl_full_chunked = (
        _read_json(ssl_full_chunked_metrics_path) if ssl_full_chunked_metrics_path.exists() else {}
    )
    ssl_full_chunked_failure = (
        _read_json(ssl_full_chunked_failure_path) if ssl_full_chunked_failure_path.exists() else {}
    )
    ssl_full_previous_interruption = (
        _read_json(ssl_full_previous_interruption_path)
        if ssl_full_previous_interruption_path.exists()
        else {}
    )
    ssl_full_multiwindow = (
        _read_json(ssl_full_multiwindow_metrics_path)
        if ssl_full_multiwindow_metrics_path.exists()
        else {}
    )
    ssl_full_multiwindow_failure = (
        _read_json(ssl_full_multiwindow_failure_path)
        if ssl_full_multiwindow_failure_path.exists()
        else {}
    )
    ssl_full_current = (
        _read_json(ssl_full_current_metrics_path) if ssl_full_current_metrics_path.exists() else {}
    )
    ssl_full_current_failure = (
        _read_json(ssl_full_current_failure_path) if ssl_full_current_failure_path.exists() else {}
    )
    ssl_full_workers4_current_failure = (
        _read_json(ssl_full_workers4_current_failure_path)
        if ssl_full_workers4_current_failure_path.exists()
        else {}
    )
    robustness_smoke = _read_json(robustness_smoke_path) if robustness_smoke_path.exists() else {}
    phase12_smoke = _read_json(phase12_smoke_path) if phase12_smoke_path.exists() else {}
    runtime_smoke = _read_json(runtime_smoke_path) if runtime_smoke_path.exists() else {}
    pilot_audit_current = (
        _read_json(pilot_audit_current_path) if pilot_audit_current_path.exists() else {}
    )
    full_cache_failure = (
        _read_json(full_cache_failure_path) if full_cache_failure_path.exists() else {}
    )
    rotating_cache_plan = (
        _read_json(rotating_cache_plan_path) if rotating_cache_plan_path.exists() else {}
    )
    gzip_smoke_manifest = (
        _read_json(gzip_smoke_root / "manifest.json")
        if (gzip_smoke_root / "manifest.json").exists()
        else {}
    )
    gzip_full_manifest = (
        _read_json(gzip_full_root / "manifest.json")
        if (gzip_full_root / "manifest.json").exists()
        else {}
    )
    gzip_full_state = (
        _read_json(gzip_full_root / "build_state.json")
        if (gzip_full_root / "build_state.json").exists()
        else {}
    )
    release_cache_smoke_manifest = (
        _read_json(release_cache_smoke_manifest_path)
        if release_cache_smoke_manifest_path.exists()
        else {}
    )
    release_cache_smoke_parity = (
        _read_json(release_cache_smoke_parity_path)
        if release_cache_smoke_parity_path.exists()
        else {}
    )
    release_cache_full_manifest = (
        _read_json(release_cache_full_manifest_path)
        if release_cache_full_manifest_path.exists()
        else {}
    )
    release_cache_full_parity = (
        _read_json(release_cache_full_parity_path)
        if release_cache_full_parity_path.exists()
        else {}
    )
    gzip_full_partial_counts = _partial_compressed_cache_counts(gzip_full_root)
    gzip_full_manifest_counts = _manifest_compressed_cache_counts(gzip_full_manifest)
    gzip_full_shard_counts = (
        gzip_full_manifest_counts if gzip_full_manifest else gzip_full_partial_counts
    )
    rgb_manifest = _read_json(rgb_manifest_path) if rgb_manifest_path.exists() else {}
    rgb_audit = _read_json(rgb_audit_path) if rgb_audit_path.exists() else {}
    rgbe_summary = _read_json(rgbe_summary_path) if rgbe_summary_path.exists() else {}
    foreground_failure = (
        _read_json(foreground_failure_path) if foreground_failure_path.exists() else {}
    )
    zero_shot_validation = (
        _read_json(zero_shot_validation_path) if zero_shot_validation_path.exists() else {}
    )
    zero_shot_aggregate = (
        _read_json(zero_shot_aggregate_path) if zero_shot_aggregate_path.exists() else {}
    )
    submission_validation = (
        _read_json(submission_validation_path) if submission_validation_path.exists() else {}
    )
    ssl_window_budget = (
        _read_json(ssl_window_budget_path) if ssl_window_budget_path.exists() else {}
    )
    phase4_overfit = (
        _read_json(phase4_overfit_summary_path) if phase4_overfit_summary_path.exists() else {}
    )
    phase4_health = _read_json(phase4_health_path) if phase4_health_path.exists() else {}
    report_manifest = _read_json(report_manifest_path) if report_manifest_path.exists() else {}
    plan_audit = _read_json(plan_audit_path) if plan_audit_path.exists() else {}
    real_screen_decision = real_screen.get("s4_vs_s5", {}).get("decision")
    gates = {
        "ci_green": _quality_gate(quality),
        "artifact_contract_green": _contract_gate(smoke_manifest, _read_json(smoke_audit_path)),
        "cache_smoke_green": _status(smoke_audit_path, "PASS"),
        "signed_metrics_green": quality["commands"]["targeted_unit_tests"].get("exit_code") == 0,
        "official_preprocessing_parity_green": _status(preprocessing_path, "pass"),
        "official_model_parity_green": _status(model_path, "pass"),
        "official_release_audit_green": _status(release_path, "pass"),
        "cache_pilot_green": _pilot_gate(pilot_root, pilot_audit_path),
        "release_input_cache_smoke_green": (
            release_cache_smoke_manifest.get("artifact_type") == "garl_release_input_cache_v1"
            and release_cache_smoke_manifest.get("bbox_protocol") == "P0_oracle_bbox_roi"
            and release_cache_smoke_manifest.get("split_counts") == {"train": 12, "validation": 12}
            and release_cache_smoke_parity.get("status") == "pass"
            and release_cache_smoke_parity.get("checked_samples") == 24
        ),
        "release_input_cache_full_green": (
            release_cache_full_manifest.get("artifact_type") == "garl_release_input_cache_v1"
            and release_cache_full_manifest.get("bbox_protocol") == "P0_oracle_bbox_roi"
            and release_cache_full_manifest.get("split_counts")
            == {"train": 71_047, "validation": 17_697}
            and release_cache_full_parity.get("status") == "pass"
            and release_cache_full_parity.get("checked_samples", 0) > 0
        ),
        "highres_resolution_green": resolution.get("status") == "pass",
        "highres_token_scaling_green": (
            token_scaling.get("status") == "pass"
            and token_scaling.get("global_attention_over_r4_allocated") is False
        ),
        "highres_architecture_smoke_green": (
            architecture_smoke.get("status") == "pass"
            and architecture_smoke.get("selection_allowed") is False
        ),
        "highres_temporal_sweep_green": (
            temporal_sweep.get("status") == "pass"
            and temporal_sweep.get("selection_allowed") is False
            and temporal_sweep.get("temporal_steps") == [2, 5, 8, 16, 32]
        ),
        "highres_real_screen_green": (
            real_screen.get("status") == "pass" and real_screen.get("selection_allowed") is False
        ),
        "highres_matrix_aggregate_green": (
            highres_matrix_aggregate.get("artifact_type") == "highres_matrix_aggregate_v1"
            and highres_matrix_aggregate.get("status") == "pass"
            and highres_matrix_aggregate.get("selection_allowed") is False
            and highres_matrix_aggregate.get("decisions", {})
            .get("s5_kda", {})
            .get("promotion_allowed")
            is False
            and highres_matrix_aggregate.get("decisions", {})
            .get("historical_k1", {})
            .get("mixed_with_s3_s4_s5")
            is False
        ),
        "highres_kda_promoted": real_screen_decision == "no_regression_in_short_screen",
        "downstream_lhr_smoke_green": (
            lhr_smoke.get("artifact_type") == "eap_lhr_object_jepa_ttc_training_v3"
            and lhr_smoke.get("epochs_completed_this_invocation", 0) > 0
        ),
        "ssl_pure_smoke_green": (
            ssl_smoke.get("artifact_type") == "eap_ssl_on_demand_pretraining_v1"
            and _is_cuda_report(ssl_smoke.get("device"))
            and ssl_smoke.get("provenance", {}).get("uses_ttc_labels") is False
            and ssl_smoke.get("provenance", {}).get("uses_object_bboxes") is False
            and ssl_smoke.get("provenance", {}).get("uses_depth_track_derivatives") is False
            and _ssl_pure_sampling_provenance(ssl_smoke.get("provenance", {}))
            and ssl_smoke.get("epochs_completed", 0) > 0
        ),
        "ssl_multiwindow_smoke_green": (
            ssl_multiwindow_smoke.get("artifact_type") == "eap_ssl_on_demand_pretraining_v1"
            and _is_cuda_report(ssl_multiwindow_smoke.get("device"))
            and ssl_multiwindow_smoke.get("trainer_config", {}).get("event_chunk_size") == 250000
            and ssl_multiwindow_smoke.get("provenance", {}).get("uses_ttc_labels") is False
            and ssl_multiwindow_smoke.get("provenance", {}).get("uses_object_bboxes") is False
            and ssl_multiwindow_smoke.get("provenance", {}).get("uses_rgb") is False
            and _ssl_pure_sampling_provenance(ssl_multiwindow_smoke.get("provenance", {}))
            and ssl_multiwindow_smoke.get("epochs_completed", 0) > 0
        ),
        "ssl_multiwindow_perf_screen_green": _ssl_perf_screen_gate(ssl_multiwindow_perf_screen),
        "ssl_multiwindow_perf_workers2_green": _ssl_perf_screen_gate(ssl_multiwindow_perf_workers2),
        "ssl_multiwindow_perf_workers4_green": _ssl_perf_screen_gate(ssl_multiwindow_perf_workers4),
        "ssl_temporal_cache_screen_green": (
            _ssl_perf_screen_gate(ssl_temporal_cache_screen)
            and ssl_temporal_cache_screen.get("trainer_config", {}).get(
                "reuse_temporal_voxel_cache"
            )
            is True
        ),
        "ssl_pure_screen_green": (
            ssl_screen.get("artifact_type") == "eap_ssl_on_demand_pretraining_v1"
            and _is_cuda_report(ssl_screen.get("device"))
            and ssl_screen.get("provenance", {}).get("uses_ttc_labels") is False
            and ssl_screen.get("provenance", {}).get("uses_object_bboxes") is False
            and _ssl_pure_sampling_provenance(ssl_screen.get("provenance", {}))
            and ssl_screen.get("epochs_completed", 0) >= 3
            and ssl_screen.get("best_validation_loss") is not None
            and ssl_screen.get("history", [{}])[-1]
            .get("validation", {})
            .get("context_collapsed_dimension_fraction")
            == 0.0
        ),
        "ssl_full_train40_interrupted": ssl_full_failure.get("status") == "interrupted",
        "ssl_worker_budget_failure_preserved": ssl_screen_failure.get("status") == "interrupted",
        "ssl_full_workers2_failure_preserved": ssl_full_workers2_failure.get("status") == "failed",
        "ssl_full_train40_green": (
            any(
                _ssl_full_train_gate(summary)
                for summary in (
                    ssl_full_workers0,
                    ssl_full_chunked,
                    ssl_full_multiwindow,
                    ssl_full_current,
                )
            )
        ),
        "ssl_full_workers0_failure_preserved": ssl_full_workers0_failure.get("status")
        in {
            "failed",
            "interrupted",
        },
        "robustness_smoke_green": (
            robustness_smoke.get("artifact_type") == "synthetic_robustness_smoke_v1"
            and robustness_smoke.get("status") == "completed"
            and robustness_smoke.get("corruptions_tested", 0) > 0
            and all(
                item.get("status") == "completed"
                for item in robustness_smoke.get("results", {}).values()
            )
        ),
        "phase12_preparation_smoke_green": (
            phase12_smoke.get("artifact_type") == "plan_phase12_smoke_v1"
            and phase12_smoke.get("status") == "passed"
            and phase12_smoke.get("metrics_are_not_real_dataset_results") is True
            and phase12_smoke.get("scope", {}).get("external_data") is False
            and phase12_smoke.get("scope", {}).get("training") is False
            and phase12_smoke.get("scope", {}).get("codabench") is False
            and phase12_smoke.get("checks", {}).get("no_leakage", {}).get("test_split_not_opened")
            is True
        ),
        "runtime_smoke_green": (
            runtime_smoke.get("artifact_type") == "runtime_export_streaming_smoke_v1"
            and runtime_smoke.get("status") == "completed"
            and runtime_smoke.get("export", {}).get("verified_with_onnxruntime_cpu") is True
            and runtime_smoke.get("streaming", {}).get("event_count", 0) > 0
        ),
        "pilot_audit_current_green": pilot_audit_current.get("status") == "PASS",
        "report_regenerable_green": (
            report_manifest.get("artifact_type") == "regenerable_report_manifest_v1"
            and report_manifest.get("artifact_count", 0) > 0
        ),
        "plan_execution_audit_green": (
            plan_audit.get("artifact_type") == "plan_execution_audit_v1"
            and plan_audit.get("plan_path") == "PLAN.md"
            and plan_audit.get("plan_declared_complete") is False
            and sum(plan_audit.get("summary", {}).values()) == 14
            and all(
                item.get("status")
                in {"verified", "partial", "not_executed", "blocked_authorization"}
                for item in plan_audit.get("phases", [])
            )
        ),
        "full_train40_cache_blocked_storage": (
            full_cache_failure.get("status") == "blocked_preflight"
            and full_cache_failure.get("materialization_started") is False
        ),
        "rotating_cache_plan_green": (
            rotating_cache_plan.get("artifact_type") == "garlttc_rotating_cache_plan_v1"
            and rotating_cache_plan.get("status") == "pass"
            and rotating_cache_plan.get("selected_row_count") == 88_744
            and rotating_cache_plan.get("split_counts") == {"train": 71_047, "validation": 17_697}
            and rotating_cache_plan.get("storage", {}).get("rotating_peak_fits") is True
            and rotating_cache_plan.get("storage", {}).get("retained_full_fits") is False
            and rotating_cache_plan.get("materialization_started") is False
        ),
        "cache_gzip_smoke_green": _compressed_cache_gate(
            gzip_smoke_root,
            gzip_smoke_audit_path,
            train_count=12,
            validation_count=12,
        ),
        "full_train40_cache_gzip_green": _compressed_cache_gate(
            gzip_full_root,
            gzip_full_audit_path,
            train_count=71_047,
            validation_count=17_697,
        )
        and gzip_full_state.get("status") == "completed",
        "rgb_cache_smoke_green": (
            rgb_manifest.get("artifact_type") == "garlttc_official_lhr_object_cache_v4"
            and rgb_manifest.get("config", {}).get("include_rgb") is True
            and rgb_manifest.get("split_counts") == {"train": 12, "validation": 12}
            and rgb_audit.get("status") == "PASS"
            and "garl_rgb_pair" in rgb_audit.get("model_input_fields", [])
            and not set(rgb_audit.get("model_input_fields", [])).intersection(
                rgb_audit.get("forbidden_model_input_fields", [])
            )
        ),
        "rgbe_downstream_smoke_green": (
            rgbe_summary.get("artifact_type") == "eap_lhr_object_jepa_ttc_training_v3"
            and rgbe_summary.get("epochs_completed_this_invocation", 0) > 0
            and rgbe_summary.get("model_config", {}).get("use_rgb") is True
            and rgbe_summary.get("no_privileged_model_inputs") is True
        ),
        "foreground_missing_mask_guard_green": (
            foreground_failure.get("status") == "failed"
            and "no official/teacher mask target" in foreground_failure.get("error_message", "")
            and "Weak rectangular bbox masks" in foreground_failure.get("error_message", "")
        ),
        "zero_shot_pilot_green": (
            zero_shot_validation.get("artifact_type") == "eap_lhr_object_jepa_ttc_zero_shot_v3"
            and zero_shot_validation.get("sample_count", 0) > 0
            and zero_shot_validation.get("training_updates_on_target_dataset") == 0
            and zero_shot_validation.get("no_privileged_model_inputs") is True
            and zero_shot_validation.get("splits") == ["validation"]
        ),
        "zero_shot_aggregate_green": (
            zero_shot_aggregate.get("artifact_type") == "eap_lhr_zero_shot_oof_aggregate_v3"
            and zero_shot_aggregate.get("bootstrap", {}).get("status")
            == "sequence_cluster_bootstrap"
            and zero_shot_aggregate.get("benchmark10_opened") is False
        ),
        "submission_validation_green": (
            submission_validation.get("artifact_type") == "garlttc_submission_validation_v1"
            and submission_validation.get("status") == "PASS"
            and submission_validation.get("external_submission_sent") is False
        ),
        "submission_token_coverage_complete": (
            submission_validation.get("status") == "PASS"
            and submission_validation.get("complete_token_coverage") is True
        ),
        "recovery_protocol_freeze_smoke_green": freeze_protocol_path.exists(),
        "ssl_full_chunked_failure_preserved": ssl_full_chunked_failure.get("status")
        in {"failed", "interrupted"},
        "ssl_full_previous_interruption_preserved": ssl_full_previous_interruption.get("status")
        == "interrupted",
        "ssl_full_multiwindow_failure_preserved": ssl_full_multiwindow_failure.get("status")
        in {"failed", "interrupted"},
        "ssl_full_current_failure_preserved": any(
            failure.get("status") in {"failed", "interrupted"}
            for failure in (ssl_full_current_failure, ssl_full_workers4_current_failure)
        ),
        "ssl_window_memory_guard_green": (
            ssl_window_budget.get("artifact_type") == "eap_ssl_window_budget_audit_v1"
            and ssl_window_budget.get("status") == "pass"
            and ssl_window_budget.get("chunking_required") is True
            and ssl_window_budget.get("temporary_array_bytes_upper_bound", 0) > 0
            and ssl_window_budget.get("uses_ttc_labels") is False
            and ssl_window_budget.get("uses_object_bboxes") is False
            and _ssl_pure_sampling_provenance(ssl_window_budget)
        ),
        "phase4_jepa_gate_green": (
            phase4_overfit.get("artifact_type") == "carla_dvs_looming_jepa_pretraining_v1"
            and phase4_overfit.get("epochs_completed", 0) >= 2
            and phase4_overfit.get("history", [{}])[0]
            .get("validation", {})
            .get("loss", float("inf"))
            > phase4_overfit.get("best_validation_loss", float("inf"))
            and phase4_overfit.get("leakage_audit", {}).get("uses_ttc_labels") is False
            and phase4_overfit.get("leakage_audit", {}).get("train_validation_sequences_disjoint")
            is True
            and phase4_health.get("status") == "PASS"
            and phase4_health.get("last_validation", {}).get("collapsed_dimension_fraction") == 0.0
        ),
    }
    required = (
        "ci_green",
        "artifact_contract_green",
        "cache_smoke_green",
        "signed_metrics_green",
        "official_preprocessing_parity_green",
        "official_model_parity_green",
        "cache_pilot_green",
        "release_input_cache_smoke_green",
        "release_input_cache_full_green",
    )
    gates["long_training_authorized"] = all(gates[name] for name in required)

    evidence_paths = [
        quality_path,
        release_path,
        preprocessing_path,
        model_path,
        smoke_manifest_path,
        smoke_audit_path,
        pilot_root / "manifest.json",
        pilot_audit_path,
        resolution_path,
        token_scaling_path,
        architecture_smoke_path,
        temporal_sweep_path,
        real_screen_path,
        highres_matrix_aggregate_path,
        ssl_smoke_metrics_path,
        ssl_cache_capacity_smoke_path,
        ssl_multiwindow_smoke_metrics_path,
        ssl_multiwindow_perf_screen_metrics_path,
        ssl_multiwindow_perf_workers2_metrics_path,
        ssl_multiwindow_perf_workers4_metrics_path,
        ssl_temporal_cache_screen_metrics_path,
        ssl_screen_metrics_path,
        ssl_full_failure_path,
        ssl_screen_failure_path,
        ssl_full_workers2_failure_path,
        ssl_full_workers0_metrics_path,
        ssl_full_workers0_failure_path,
        ssl_full_chunked_metrics_path,
        ssl_full_chunked_failure_path,
        ssl_full_previous_interruption_path,
        ssl_full_multiwindow_metrics_path,
        ssl_full_multiwindow_failure_path,
        ssl_full_current_metrics_path,
        ssl_full_current_failure_path,
        robustness_smoke_path,
        phase12_smoke_path,
        runtime_smoke_path,
        report_manifest_path,
        pilot_audit_current_path,
        full_cache_failure_path,
        rotating_cache_plan_path,
        gzip_smoke_root / "manifest.json",
        gzip_smoke_audit_path,
        gzip_full_root / "manifest.json",
        gzip_full_root / "build_state.json",
        gzip_full_root / "FAILURE.json",
        gzip_full_audit_path,
        release_cache_smoke_manifest_path,
        release_cache_smoke_parity_path,
        release_cache_full_manifest_path,
        release_cache_full_parity_path,
        repo_root / "artifacts/cache/garlttc_lhr_v4_full_train40_gzip_v2/FAILURE.json",
        rgb_manifest_path,
        rgb_audit_path,
        rgbe_summary_path,
        foreground_failure_path,
        zero_shot_validation_path,
        zero_shot_aggregate_path,
        submission_path,
        submission_validation_path,
        freeze_protocol_path,
        rgb_cache_failure_path,
        ssl_window_budget_path,
        ssl_full_workers4_current_failure_path,
        phase4_overfit_summary_path,
        phase4_health_path,
        # PLAN_AUDIT is a consumer of readiness, not an input evidence file.
        # Including it here would create an impossible circular hash: the
        # audit records readiness's hash while readiness records the audit's
        # hash.  Keep the structural gate above, and let PLAN_AUDIT carry the
        # one-way readiness hash instead.
        repo_root / "artifacts/benchmarks/highres_real_screen_v1/FAILURE.json",
        repo_root / "artifacts/benchmarks/highres_real_screen_v1_gpu/FAILURE.json",
        repo_root / "artifacts/cache/garlttc_lhr_v4_pilot_4096/FAILURE.json",
        repo_root / "artifacts/cache/garlttc_lhr_v4_pilot_4096_cuda/FAILURE.json",
        repo_root / "artifacts/cache/garlttc_lhr_v4_pilot_4096_workers32/FAILURE.json",
        repo_root / "artifacts/runs/garl_v4_lhr_smoke_seed7/FAILURE.json",
        repo_root / "artifacts/runs/garl_v4_lhr_smoke_seed7_bounded/FAILURE.json",
        repo_root / "artifacts/runs/garl_v4_lhr_smoke_seed7_shardlocal/summary.json",
        lhr_smoke_summary_path,
        repo_root / "artifacts/runs/garl_v4_lhr_smoke_seed7_shardlocal_modern/weights_only.pt",
    ]
    evidence = {
        str(path.relative_to(repo_root)).replace("\\", "/"): _sha256(path)
        for path in evidence_paths
        if path.exists()
    }
    blocked_reasons = [name for name in required if not gates[name]]
    result: dict[str, Any] = {
        "artifact_type": "e_jepa_garl_readiness_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(repo_root),
        **gates,
        "gates": gates,
        "evidence": evidence,
        "metrics": {
            "smoke_train_samples": smoke_manifest.get("split_counts", {}).get("train"),
            "smoke_validation_samples": smoke_manifest.get("split_counts", {}).get("validation"),
            "smoke_jepa_pair_valid_fraction": smoke_manifest.get("jepa_pair_valid_fraction"),
            "smoke_discard_fraction": smoke_manifest.get("discard_fraction"),
            "preprocessing_samples": _read_json(preprocessing_path).get("samples"),
            "preprocessing_raw_max_abs": _read_json(preprocessing_path).get("raw_max_abs"),
            "preprocessing_resized_max_abs": _read_json(preprocessing_path).get("resized_max_abs"),
            "model_raw_height_max_abs": _read_json(model_path).get("raw_height_max_abs"),
            "model_ttc_max_abs": _read_json(model_path).get("ttc_max_abs"),
            "highres_r4_tokens": next(
                item["tokens"]
                for item in token_scaling.get("results", [])
                if item.get("name") == "R4"
            ),
            "highres_real_screen_validation_samples": real_screen.get("split_counts", {}).get(
                "validation"
            ),
            "highres_s4_paper_mid": next(
                item["validation_metrics"].get("paper_MiD_overall")
                for item in real_screen.get("arms", [])
                if item.get("arm") == "S4_R4_WINDOW_MERGE_TEMPORAL"
            ),
            "highres_s5_paper_mid": next(
                item["validation_metrics"].get("paper_MiD_overall")
                for item in real_screen.get("arms", [])
                if item.get("arm") == "S5_R4_WINDOW_MERGE_KDA"
            ),
            "highres_s4_weighted_rte_pct": next(
                item["validation_metrics"].get("weighted_RTE_pct")
                for item in real_screen.get("arms", [])
                if item.get("arm") == "S4_R4_WINDOW_MERGE_TEMPORAL"
            ),
            "highres_s5_weighted_rte_pct": next(
                item["validation_metrics"].get("weighted_RTE_pct")
                for item in real_screen.get("arms", [])
                if item.get("arm") == "S5_R4_WINDOW_MERGE_KDA"
            ),
            "highres_s4_vs_s5_decision": real_screen_decision,
            "highres_s4_t32_p95_ms": next(
                item["forward_p95_ms"]
                for item in temporal_sweep.get("results", [])
                if item.get("arm") == "S4_R4_WINDOW_MERGE_TEMPORAL"
                and item.get("temporal_steps") == 32
            ),
            "highres_s5_t32_p95_ms": next(
                item["forward_p95_ms"]
                for item in temporal_sweep.get("results", [])
                if item.get("arm") == "S5_R4_WINDOW_MERGE_KDA" and item.get("temporal_steps") == 32
            ),
            "downstream_lhr_smoke_validation_mid": lhr_smoke.get(
                "best_validation_sequence_macro_paper_MiD_overall"
            ),
            "downstream_lhr_smoke_epochs": lhr_smoke.get("epochs_completed_this_invocation"),
            "downstream_lhr_smoke_elapsed_seconds": lhr_smoke.get("elapsed_seconds"),
            "rgbe_smoke_validation_mid": rgbe_summary.get(
                "best_validation_sequence_macro_paper_MiD_overall"
            ),
            "rgbe_smoke_elapsed_seconds": rgbe_summary.get("elapsed_seconds"),
            "zero_shot_pilot_sample_count": zero_shot_validation.get("sample_count"),
            "zero_shot_pilot_sequence_count": zero_shot_aggregate.get("sequence_count"),
            "zero_shot_pilot_mae_s": zero_shot_aggregate.get("metrics", {}).get("mae_s"),
            "official_local_submission_prediction_count": submission_validation.get(
                "prediction_count"
            ),
            "foreground_smoke_status": foreground_failure.get("status"),
            "ssl_window_max_indexed_events": ssl_window_budget.get("maximum_indexed_window_events"),
            "ssl_window_chunk_size": ssl_window_budget.get("event_chunk_size"),
            "ssl_window_temp_array_bytes_upper_bound": ssl_window_budget.get(
                "temporary_array_bytes_upper_bound"
            ),
            "phase4_overfit_epochs": phase4_overfit.get("epochs_completed"),
            "phase4_overfit_first_validation_loss": (
                phase4_overfit.get("history", [{}])[0].get("validation", {}).get("loss")
            ),
            "phase4_overfit_best_validation_loss": phase4_overfit.get("best_validation_loss"),
            "ssl_full_chunked_epochs": ssl_full_chunked.get("epochs_completed"),
            "ssl_multiwindow_smoke_best_validation_loss": ssl_multiwindow_smoke.get(
                "best_validation_loss"
            ),
            "ssl_multiwindow_perf_screen_best_validation_loss": ssl_multiwindow_perf_screen.get(
                "best_validation_loss"
            ),
            "ssl_multiwindow_perf_screen_elapsed_seconds": ssl_multiwindow_perf_screen.get(
                "elapsed_seconds"
            ),
            "ssl_multiwindow_perf_screen_train_samples": ssl_multiwindow_perf_screen.get(
                "selected_train_samples"
            ),
            "ssl_multiwindow_perf_screen_validation_samples": ssl_multiwindow_perf_screen.get(
                "selected_validation_samples"
            ),
            "ssl_multiwindow_perf_workers2_elapsed_seconds": ssl_multiwindow_perf_workers2.get(
                "elapsed_seconds"
            ),
            "ssl_multiwindow_perf_workers4_elapsed_seconds": ssl_multiwindow_perf_workers4.get(
                "elapsed_seconds"
            ),
            "ssl_temporal_cache_screen_elapsed_seconds": ssl_temporal_cache_screen.get(
                "elapsed_seconds"
            ),
            "ssl_temporal_cache_screen_artifact_sha256": ssl_temporal_cache_screen.get(
                "artifact_sha256"
            ),
            "ssl_full_multiwindow_epochs": ssl_full_multiwindow.get("epochs_completed"),
            "rotating_cache_selected_rows": rotating_cache_plan.get("selected_row_count"),
            "rotating_cache_shard_count": rotating_cache_plan.get("shard_count"),
            "rotating_cache_peak_gib": rotating_cache_plan.get("storage", {}).get(
                "estimated_rotating_peak_gib"
            ),
            "rotating_cache_retained_full_gib": rotating_cache_plan.get("storage", {}).get(
                "estimated_retained_full_gib"
            ),
            "gzip_smoke_train_samples": gzip_smoke_manifest.get("split_counts", {}).get("train"),
            "gzip_smoke_validation_samples": gzip_smoke_manifest.get("split_counts", {}).get(
                "validation"
            ),
            "gzip_full_train_samples": gzip_full_manifest.get("split_counts", {}).get("train"),
            "gzip_full_validation_samples": gzip_full_manifest.get("split_counts", {}).get(
                "validation"
            ),
            "gzip_full_train_shard_count": (gzip_full_shard_counts["train"]),
            "gzip_full_validation_shard_count": (gzip_full_shard_counts["validation"]),
            "gzip_full_shard_count": (sum(gzip_full_shard_counts.values())),
            "release_cache_smoke_train_samples": release_cache_smoke_manifest.get(
                "split_counts", {}
            ).get("train"),
            "release_cache_smoke_validation_samples": release_cache_smoke_manifest.get(
                "split_counts", {}
            ).get("validation"),
            "release_cache_full_train_samples": release_cache_full_manifest.get(
                "split_counts", {}
            ).get("train"),
            "release_cache_full_validation_samples": release_cache_full_manifest.get(
                "split_counts", {}
            ).get("validation"),
            "release_cache_full_parity_checked_samples": release_cache_full_parity.get(
                "checked_samples"
            ),
        },
        "blocked_reason": (
            "Long training is not authorized because these gates are red: "
            + ", ".join(blocked_reasons)
            if blocked_reasons
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/recovery_v4/readiness.json"),
    )
    args = parser.parse_args()
    result = build_readiness(args.repo_root.resolve(), args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

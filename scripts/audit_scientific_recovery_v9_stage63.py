"""Execute Stage 63 physical raw/provenance readiness without training a model."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.data.crossfitted_a5_state import (  # noqa: E402
    build_crossfitted_a5_states,
    load_a5_state_split,
)
from e_jepa_ttc.data.raw_event_binding import (  # noqa: E402
    EXPECTED_SEQUENCES,
    ReadOnlyTrainAccess,
    build_raw_window_bindings,
    select_hash_probe_tokens,
    sha256_file,
)
from e_jepa_ttc.data.raw_temporal_cache import (  # noqa: E402
    build_raw_temporal_cache,
    load_raw_cache,
)
from e_jepa_ttc.evaluation.stage63_65 import (  # noqa: E402
    benchmark_phase,
    scientific_ttc,
    strict_score,
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _stage61_artifact_root(stage61_worktree: Path) -> Path:
    return stage61_worktree / "artifacts" / "scientific_recovery_v9_stage61_stage62"


def _replay_check(state_root: Path, stage61_root: Path) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    for outer in range(3):
        split = load_a5_state_split(state_root, outer, "eval")
        frame = cast(
            pd.DataFrame,
            split.metadata[["sample_token", "sequence_id", "track_id", "outer_fold"]].copy(),
        )
        frame["prediction_ttc_s"] = scientific_ttc(split.state[:, 0].astype(np.float64))
        rows.append(frame)
    replay = pd.concat(rows, ignore_index=True)
    historical_path = stage61_root / "stage62" / "aggregate_seed7" / "X2_A5_REPLAY_oof.csv"
    historical = pd.read_csv(historical_path, dtype={"sample_token": str})
    merged = replay.merge(
        historical[["sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"]],
        on=["sample_token", "sequence_id", "track_id"],
        suffixes=("_new", "_historical"),
        validate="one_to_one",
    )
    if len(merged) != 8192:
        raise ValueError("A5 replay comparison is not complete")
    new_phase = benchmark_phase(merged["prediction_ttc_s_new"].to_numpy(np.float64))
    old_phase = benchmark_phase(merged["prediction_ttc_s_historical"].to_numpy(np.float64))
    max_phase = float(np.max(np.abs(new_phase - old_phase)))
    new_frame = cast(
        pd.DataFrame,
        merged.rename(columns={"prediction_ttc_s_new": "prediction_ttc_s"})[
            ["sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"]
        ],
    )
    old_frame = cast(
        pd.DataFrame,
        merged.rename(columns={"prediction_ttc_s_historical": "prediction_ttc_s"})[
            ["sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"]
        ],
    )
    new_score = float(strict_score(new_frame)["score"])
    old_score = float(strict_score(old_frame)["score"])
    return {
        "historical_path": str(historical_path),
        "historical_sha256": sha256_file(historical_path),
        "rows": len(merged),
        "maximum_absolute_phase_difference": max_phase,
        "new_score": new_score,
        "historical_score": old_score,
        "absolute_score_difference": abs(new_score - old_score),
        "phase_tolerance": 1e-6,
        "score_tolerance": 0.01,
        "passed": max_phase <= 1e-6 and abs(new_score - old_score) <= 0.01,
    }


def _exposure_ledger(train_parquet: Path, output: Path) -> None:
    metadata = pd.read_parquet(train_parquet, columns=["sequence_id", "sample_token"])
    values: list[dict[str, object]] = []
    for sequence, group in metadata.groupby("sequence_id", sort=True):
        known = str(sequence) in EXPECTED_SEQUENCES
        values.append(
            {
                "sequence_id": str(sequence),
                "official_split": "train",
                "metadata_rows": len(group),
                "target_use": "known" if known else "unknown",
                "unsupervised_or_pretraining_use": "known" if known else "unknown",
                "model_input_use": "known" if known else "unknown",
                "metrics_observed": "yes" if known else "unknown",
                "role": "adaptive_development_8192" if known else "unopened_metadata_only",
                "exposure_status": "known" if known else "unknown",
            }
        )
    pd.DataFrame(values).to_csv(output, index=False, lineterminator="\n")


def _raw_unavailable_result(
    *, repo: Path, source_root: Path, stage_root: Path, output_root: Path, reason: str
) -> dict[str, Any]:
    old_x3 = source_root / "X3_FEASIBILITY.json"
    result = {
        "artifact_type": "scientific_recovery_v9_x3_feasibility_v2",
        "status": "completed_read_only",
        "decision": "RAW_DATA_UNAVAILABLE",
        "physical_data_ready": False,
        "integrity_passed": True,
        "training_ready": False,
        "authorization_id": "scientific_recovery_v9_stage63_65_rawtime_fullregret_v1",
        "raw_training_executed": False,
        "code_commit": _git_head(repo),
        "supersedes_artifact_path": str(old_x3),
        "supersedes_file_sha256": sha256_file(old_x3),
        "supersedes_artifact_sha256": json.loads(old_x3.read_text(encoding="utf-8"))[
            "artifact_sha256"
        ],
        "reason_for_revision": reason,
        "binding_manifest": {},
        "raw_cache_manifest": {},
        "crossfitted_state_manifest": {},
        "a5_replay": {"passed": False, "not_run_reason": "raw unavailable"},
        "support_readiness": {"passed": False, "not_run_reason": "raw unavailable"},
        "forbidden_paths_opened": None,
        "labels_in_raw_binding_or_cache": False,
        "elapsed_seconds": 0.0,
        "disk_free_bytes": shutil.disk_usage(output_root).free,
    }
    result = sign_stage63_65_artifact(result, evidence_type="physical_raw_feasibility")
    _atomic_json(stage_root / "X3_FEASIBILITY_V2.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    from e_jepa_ttc.training.campaign_budget import CampaignBudget, check_resource_margins

    repo = Path(__file__).resolve().parents[1]
    began = time.perf_counter()
    stage_root = args.output_root / "stage63"
    stage_root.mkdir(parents=True, exist_ok=args.resume)
    if args.max_hours != 12.0:
        raise ValueError("Stage63 budget differs from the authorized twelve-hour cap")
    budget = CampaignBudget(args.output_root / "budgets/stage63.json", hours=12.0)

    def resource_check() -> None:
        budget.check()
        check_resource_margins(args.output_root)

    resource_check()
    raw_cache_root = args.output_root / "raw_temporal_cache"
    state_root = args.output_root / "crossfitted_a5_state"
    source_root = _stage61_artifact_root(args.stage61_worktree)
    stage_metadata = source_root / "feature_cache" / "outer0_final.metadata.csv"
    old_x3 = source_root / "X3_FEASIBILITY.json"
    if not args.raw_train_root.is_dir() or not args.train_parquet.is_file():
        return _raw_unavailable_result(
            repo=repo,
            source_root=source_root,
            stage_root=stage_root,
            output_root=args.output_root,
            reason="authorized raw train root or train.parquet is physically unavailable",
        )
    access = ReadOnlyTrainAccess(
        args.raw_train_root,
        args.train_parquet,
        journal_path=stage_root / "READ_ACCESS_JOURNAL.jsonl",
        resource_check=resource_check,
    )
    try:
        binding, binding_manifest = build_raw_window_bindings(
            stage_metadata_path=stage_metadata,
            train_parquet=args.train_parquet,
            raw_train_root=args.raw_train_root,
            access=access,
            hash_progress_path=stage_root / "HDF5_HASH_PROGRESS.json",
        )
    except FileNotFoundError as exc:
        return _raw_unavailable_result(
            repo=repo,
            source_root=source_root,
            stage_root=stage_root,
            output_root=args.output_root,
            reason=f"authorized raw event file is physically unavailable: {exc}",
        )
    if time.perf_counter() - began > args.max_hours * 3600:
        raise TimeoutError("Stage 63 hashing exceeded its preregistered wall cap")
    state_manifest = state_root / "manifest.json"
    if state_root.exists() and not state_manifest.is_file():
        preserved = state_root.with_name(f"crossfitted_a5_state.failed_{time.time_ns()}")
        state_root.replace(preserved)
        preservation = sign_stage63_65_artifact(
            {
                "artifact_type": "scientific_recovery_v9_partial_state_preservation_v1",
                "status": "preserved_before_resume",
                "original_path": str(state_root),
                "preserved_path": str(preserved),
                "reason": "state directory existed without a signed completion manifest",
            },
            evidence_type="failed_partial_state",
        )
        _atomic_json(stage_root / "PARTIAL_STATE_PRESERVATION.json", preservation)
    if not state_manifest.is_file():
        build_crossfitted_a5_states(
            feature_cache_root=source_root / "feature_cache",
            router_root=args.reference_root
            / "artifacts"
            / "scientific_recovery_v8"
            / "results"
            / "router",
            output_root=state_root,
            coherent_state_root=args.coherent_a5_root,
            frozen_teacher_audit_path=args.frozen_teacher_audit,
        )
    cache_manifest = build_raw_temporal_cache(
        binding=binding,
        raw_train_root=args.raw_train_root,
        access=access,
        output_dir=raw_cache_root,
        resume=args.resume,
    )
    if time.perf_counter() - began > args.max_hours * 3600:
        raise TimeoutError("Stage 63 cache construction exceeded its preregistered wall cap")
    cache = load_raw_cache(raw_cache_root)
    roi_counts = np.load(raw_cache_root / "roi_event_counts.npy")
    token_to_row = {
        token: index for index, token in enumerate(cache.metadata["sample_token"].astype(str))
    }
    probe_tokens = set(select_hash_probe_tokens(binding, 64))
    binding["roi_event_count"] = [
        int(
            roi_counts[
                token_to_row[str(cast(Any, row).sample_token)],
                int(cast(Any, row).window_id),
            ]
        )
        for row in binding.itertuples()
    ]
    binding["read_probe_status"] = [
        "physically_read_and_validated" if str(token) in probe_tokens else "indexed_and_cache_read"
        for token in binding["sample_token"]
    ]
    binding_path = stage_root / "X3_RAW_BINDING_V2.csv"
    binding.to_csv(binding_path, index=False, lineterminator="\n")
    binding_manifest["binding_csv"] = {
        "path": str(binding_path),
        "sha256": sha256_file(binding_path),
        "rows": len(binding),
    }
    binding_manifest["cache_manifest"] = cache_manifest
    _atomic_json(stage_root / "X3_RAW_BINDING_MANIFEST_V2.json", binding_manifest)
    access.write_ledger(stage_root / "READ_ACCESS_LEDGER.csv")
    _exposure_ledger(args.train_parquet, stage_root / "TRAIN_EXPOSURE_LEDGER.csv")
    replay = _replay_check(state_root, source_root)
    support_details: dict[str, Any] = {}
    support_passed = cache_manifest["supported_fraction"] >= 0.5
    for outer in range(3):
        selected = cache.metadata["outer_fold"].to_numpy(int) != outer
        supported = cache.valid_patches[selected].any(axis=1)
        tracks = cache.metadata.loc[selected].loc[supported, "track_id"].nunique()
        item = {"supported_tokens": int(supported.sum()), "supported_tracks": int(tracks)}
        item["passed"] = item["supported_tokens"] >= 512 and item["supported_tracks"] >= 32
        support_passed = support_passed and bool(item["passed"])
        support_details[str(outer)] = item
    physical_ready = (
        len(binding) == 16384
        and binding["sample_token"].nunique() == 8192
        and binding["h5_file_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
        and bool(binding_manifest["common_roi"]["same_transform_for_both_windows"])
        and cache_manifest["read_probe"]["tokens"] == 64
    )
    integrity_passed = bool(replay["passed"])
    training_ready = physical_ready and support_passed and integrity_passed
    if not integrity_passed:
        decision = "INTEGRITY_BLOCKED"
    elif not physical_ready:
        decision = "RAW_DATA_UNAVAILABLE"
    elif not support_passed:
        decision = "RAW_SUPPORT_INSUFFICIENT"
    else:
        decision = "X3_DATA_READY"
    old_x3_sha = sha256_file(old_x3)
    result = {
        "artifact_type": "scientific_recovery_v9_x3_feasibility_v2",
        "status": "completed_read_only",
        "decision": decision,
        "physical_data_ready": bool(physical_ready),
        "integrity_passed": integrity_passed,
        "training_ready": bool(training_ready),
        "authorization_id": "scientific_recovery_v9_stage63_65_rawtime_fullregret_v1",
        "raw_training_executed": False,
        "code_commit": _git_head(repo),
        "supersedes_artifact_path": str(old_x3),
        "supersedes_file_sha256": old_x3_sha,
        "supersedes_artifact_sha256": json.loads(old_x3.read_text(encoding="utf-8"))[
            "artifact_sha256"
        ],
        "reason_for_revision": "v1 lacked full hashes, common-ROI proof and per-window schema",
        "binding_manifest": binding_manifest,
        "raw_cache_manifest": cache_manifest,
        "crossfitted_state_manifest": json.loads(
            (state_root / "manifest.json").read_text(encoding="utf-8")
        ),
        "a5_replay": replay,
        "support_readiness": {
            "global_supported_fraction": cache_manifest["supported_fraction"],
            "outer_train": support_details,
            "passed": support_passed,
        },
        "forbidden_paths_opened": None,
        "access_audit": access.audit_summary(),
        "labels_in_raw_binding_or_cache": False,
        "elapsed_seconds": time.perf_counter() - began,
        "disk_free_bytes": shutil.disk_usage(args.output_root).free,
    }
    result = sign_stage63_65_artifact(result, evidence_type="physical_raw_feasibility")
    _atomic_json(stage_root / "X3_FEASIBILITY_V2.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--raw-train-root", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coherent-a5-root", type=Path, required=True)
    parser.add_argument("--frozen-teacher-audit", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=12.0)
    from e_jepa_ttc.artifacts.campaign_session import CampaignLock, append_transition

    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with CampaignLock(args.output_root / "CAMPAIGN_LOCK.json"):
        source_files = [
            Path(__file__),
            _REPOSITORY_ROOT / "src/e_jepa_ttc/data/raw_temporal_cache.py",
            _REPOSITORY_ROOT / "src/e_jepa_ttc/data/raw_event_binding.py",
            _REPOSITORY_ROOT / "src/e_jepa_ttc/data/crossfitted_a5_state.py",
        ]
        source_hashes = {str(path): sha256_file(path) for path in source_files}
        append_transition(
            args.output_root / "RUN_LEDGER.jsonl",
            "stage63_raw_audit_running",
            implementation_source_sha256=source_hashes,
            python=sys.version,
            resume=args.resume,
        )
        result = run(args)
        if source_hashes != {str(path): sha256_file(path) for path in source_files}:
            raise ValueError("Stage63 source code changed during the physical audit")
        append_transition(
            args.output_root / "RUN_LEDGER.jsonl",
            "stage63_raw_audit_completed",
            decision=result["decision"],
            implementation_unchanged=True,
        )
    print(json.dumps({"decision": result["decision"], "training_ready": result["training_ready"]}))


if __name__ == "__main__":
    main()

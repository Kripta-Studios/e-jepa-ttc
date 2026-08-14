# ruff: noqa: E501
"""Unit contracts for the executable, fail-closed V8 job substrate."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
import torch

import scripts.run_scientific_recovery_v8_adaptive as adaptive
from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.training.scientific_recovery_v8_jobs import (
    V8JobIntegrityError,
    assess_v8_job,
    build_fold_jobs,
    derive_cache_selection,
    execute_jobs,
    write_v8_job_state,
)


def _signed(path: Path, payload: dict[str, object]) -> Path:
    sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fold_jobs_are_exact_resume_safe_and_do_not_open_sealed_splits(tmp_path: Path) -> None:
    configs = []
    for fold in range(3):
        config = tmp_path / f"timevol20_3_fold{fold}_seed7.yaml"
        config.write_text(
            "experiment:\n"
            f"  name: timevol20_3_fold{fold}_seed7\n"
            "  arm: timevol20_3\n"
            "  seed: 7\n"
            "data:\n"
            "  opened_splits: [train]\n"
            f"  outer_fold: {fold}\n",
            encoding="utf-8",
        )
        configs.append(config)
    jobs = build_fold_jobs(
        configs=configs,
        output_root=tmp_path / "runs",
        device="cuda",
        max_parallel=2,
    )
    assert len(jobs) == 3
    assert jobs[0].command[:4] == ("uv", "run", "--no-sync", "python")
    assert "scripts/train_scientific_recovery_v8_temporal.py" in jobs[0].command
    assert "--resume" not in jobs[0].command
    assert all("public_validation" not in " ".join(job.command) for job in jobs)

    run_dir = tmp_path / "runs" / "timevol20_3_fold0_seed7"
    (run_dir / "state").mkdir(parents=True)
    torch.save({"checkpoint": "valid"}, run_dir / "state" / "last.pt")
    resumed = build_fold_jobs(
        configs=configs,
        output_root=tmp_path / "runs",
        device="cuda",
        max_parallel=1,
    )
    assert "--resume" in resumed[0].command

    write_v8_job_state(
        job=resumed[0], status="planned", protocol_hash="a" * 64, manifest_hash="b" * 64
    )
    state = json.loads((run_dir / "job_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "planned"
    assert state["artifact_sha256"]


def test_resume_rejects_corrupt_checkpoint_without_deleting_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "state" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a torch checkpoint")
    outcome = assess_v8_job(run_dir)
    assert outcome.status == "failed_integrity"
    assert checkpoint.exists()
    assert (run_dir / "failed_integrity.json").is_file()


def test_execute_jobs_honors_max_parallel_and_returns_deterministic_order(tmp_path: Path) -> None:
    configs = []
    for fold in range(3):
        config = tmp_path / f"exp6_3_fold{fold}_seed7.yaml"
        config.write_text(
            "experiment:\n"
            f"  name: exp6_3_fold{fold}_seed7\n"
            "  arm: exp6_3\n"
            "  seed: 7\n"
            "data:\n"
            "  opened_splits: [train]\n"
            f"  outer_fold: {fold}\n",
            encoding="utf-8",
        )
        configs.append(config)
    jobs = build_fold_jobs(
        configs=configs, output_root=tmp_path / "runs", device="cuda", max_parallel=2
    )
    active = 0
    maximum = 0
    lock = threading.Lock()

    def runner(job) -> int:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        payload = {
            "artifact_type": "test_completed_run",
            "status": "completed_train_only_grouped_dev",
            "closed_evaluation": {
                "public_validation_used_for_selection": False,
                "private_test_opened": False,
                "evttc_test_opened": False,
                "codabench_opened": False,
            },
        }
        _signed(job.output_dir / "summary.json", payload)
        with lock:
            active -= 1
        return 0

    outputs = execute_jobs(
        jobs,
        protocol_hash="a" * 64,
        manifest_hash="b" * 64,
        dry_run=False,
        max_parallel=2,
        command_runner=runner,
    )
    assert maximum == 2
    assert [item["name"] for item in outputs] == [job.name for job in jobs]

    rerun = build_fold_jobs(
        configs=configs, output_root=tmp_path / "sequential", device="cuda", max_parallel=1
    )
    active = maximum = 0
    execute_jobs(
        rerun,
        protocol_hash="a" * 64,
        manifest_hash="b" * 64,
        dry_run=False,
        max_parallel=1,
        command_runner=runner,
    )
    assert maximum == 1


def test_cache_selection_derives_paired_train_oof_rows_and_rejects_sealed_path(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "sample_token": f"token-{index}",
            "sequence_id": "sequence-a",
            "track_id": f"track-{index}",
            "target_ttc_s": "1.0",
            "fold": "0",
        }
        for index in range(2)
    ]
    a5 = tmp_path / "a5_oof.csv"
    garl = tmp_path / "garl_oof.csv"
    header = ",".join(rows[0]) + "\n"
    a5.write_text(
        header + "\n".join(",".join(row[key] for key in rows[0]) for row in rows), encoding="utf-8"
    )
    garl.write_text(
        header + "\n".join(",".join(row[key] for key in rows[0]) for row in rows), encoding="utf-8"
    )
    protocol = {
        "sources": {
            "a5_oof_predictions": {"path": str(a5)},
            "garl_oof_predictions": {"path": str(garl)},
        }
    }
    selected = derive_cache_selection(protocol, expected_rows=2)
    assert [row["sample_token"] for row in selected] == ["token-0", "token-1"]
    protocol["sources"]["a5_oof_predictions"]["path"] = "public_validation/a5.csv"
    with pytest.raises(V8JobIntegrityError, match="sealed"):
        derive_cache_selection(protocol, expected_rows=2)


def test_adaptive_conditional_configs_require_three_hash_bound_paths(
    tmp_path: Path, monkeypatch
) -> None:
    paths = []
    for fold in range(3):
        path = tmp_path / f"gated_fold{fold}.yaml"
        path.write_text(f"fold: {fold}\n", encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(adaptive, "ROOT", tmp_path)
    template = {
        "fold_configs": [
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ]
    }
    assert adaptive.conditional_fold_configs(template) == paths
    template["fold_configs"].pop()
    with pytest.raises(Exception, match="exactly three"):
        adaptive.conditional_fold_configs(template)

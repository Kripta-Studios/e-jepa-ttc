from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from operational.final_refit.preflight import build_preflight


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_workspace(root: Path) -> Path:
    run12 = root / "e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005"
    run13 = root / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    run12.mkdir(parents=True)
    run13.mkdir(parents=True)
    tokens = [f"token-{index:05d}" for index in range(88744)]
    groups = [f"group-{index % 40:02d}" for index in range(88744)]
    native = pd.DataFrame({"sample_token": tokens, "sequence_id": groups})
    native_path = run12 / "TRAIN40_ROWS.parquet"
    native.to_parquet(native_path, index=False)
    native_sha256 = _digest(native_path)
    role_group_sets = {
        "P": {f"group-{index:02d}" for index in range(24)},
        "H": {f"group-{index:02d}" for index in range(24, 32)},
        "V": {f"group-{index:02d}" for index in range(32, 40)},
    }
    for role, role_groups in role_group_sets.items():
        mask = native["sequence_id"].isin(sorted(role_groups))
        frame = native.loc[mask, ["sample_token", "sequence_id"]].copy()
        frame.columns = ["sample_token", "group_id"]
        rows_path = root / f"metadata/{role}_ROWS.parquet"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(rows_path, index=False)
        _write_json(
            run13 / f"{role}_MANIFEST.json",
            {
                "status": "COMPLETE",
                "role": role,
                "group_ids": sorted(role_groups),
                "population_size": len(frame),
                "rows_path": str(rows_path),
                "rows_sha256": _digest(rows_path),
                "source_sha256": native_sha256,
                "eap_root": str(root / "eap"),
            },
        )
    (root / "eap").mkdir()

    input_dir = root / "cache/input"
    teacher_dir = root / "cache/teacher"
    input_dir.mkdir(parents=True)
    teacher_dir.mkdir(parents=True)
    (input_dir / "shard.npz").write_bytes(b"input")
    (teacher_dir / "shard.npz").write_bytes(b"teacher")
    index_sha256 = "1" * 64
    freeze_path = run12 / "TEACHER_FREEZE.json"
    _write_json(
        freeze_path,
        {
            "new_TRAIN40_metadata_sha256": index_sha256,
            "model_revision": "revision",
            "teacher_is_training_privilege_not_model_input": True,
        },
    )
    freeze_sha256 = _digest(freeze_path)
    base_manifest = {
        "status": "COMPLETE_VERIFIED",
        "row_count": 88744,
        "sequence_count": 40,
        "rows_sha256": native_sha256,
        "index_sha256": index_sha256,
    }
    input_manifest_path = run12 / "INPUT_MANIFEST.json"
    _write_json(
        input_manifest_path,
        {
            **base_manifest,
            "directory": str(input_dir),
            "ordered_shards": [
                {"name": "shard.npz", "start": 0, "stop": 88744, "bytes": 5}
            ],
        },
    )
    teacher_manifest_path = run12 / "TEACHER_MANIFEST.json"
    _write_json(
        teacher_manifest_path,
        {
            **base_manifest,
            "directory": str(teacher_dir),
            "freeze_sha256": freeze_sha256,
            "ordered_shards": [
                {"name": "shard.npz", "start": 0, "stop": 88744, "bytes": 7}
            ],
        },
    )
    _write_json(
        run12 / "TEACHER_COMPLETE_ADMISSION.json",
        {
            "status": "PASSED",
            "complete_rows": 88744,
            "sequences": 40,
            "shards_verified": 1,
            "verified_bytes": 7,
            "teacher_freeze_sha256": freeze_sha256,
            "all_ordinals_and_tokens_exactly_match": True,
            "all_targets_finite": True,
            "TTC_targets_read": False,
            "ordered_teacher_weights_sha256": "2" * 64,
        },
    )
    for name, mode in (("a5_seed7", "legacy"), ("c2f_seed7", "adaptive_pyramid")):
        fit = run12 / "fits" / name
        fit.mkdir(parents=True)
        checkpoint = fit / "checkpoint_last.pt"
        checkpoint.write_bytes(name.encode())
        receipt = {
            "status": "COMPLETE",
            "sha256": _digest(checkpoint),
            "committed_updates": 49932,
        }
        _write_json(fit / "CHECKPOINT_RECEIPT.json", receipt)
        _write_json(
            fit / "UPDATE_JOURNAL.json",
            {
                "durable_updates": 49932,
                "contract": {
                    "updates_limit": 49932,
                    "split_version": "public_TRAIN40_all_train",
                    "dataset_manifest_sha256": _digest(input_manifest_path),
                    "teacher_manifest_sha256": _digest(teacher_manifest_path),
                    "model_config": {"modality": "event", "transport_mode": mode},
                    "training_config": {"epochs": 18, "initialization_checkpoint": None},
                    "seed": 7,
                    "protocol_sha256": "3" * 64,
                },
            },
        )
        _write_json(fit / "PROGRESS.json", {"status": "COMPLETE", "durable_updates": 49932})
        _write_json(fit / "epoch_18.json", {"updates": 49932, "examples": 88744})
    return root


def test_build_preflight_verifies_union_cache_and_all_train_parents(tmp_path: Path) -> None:
    workspace = _fixture_workspace(tmp_path)
    output = tmp_path / "result/preflight.json"

    result = build_preflight(workspace, output)

    assert result["status"] == "PASS"
    assert result["population"]["queries"] == 88744
    assert result["population"]["groups"] == 40
    assert result["population"]["union_equals_native_train40"] is True
    assert result["availability"]["producer_input_cache"]["shards"] == 1
    assert result["availability"]["dino_teacher_cache"]["payload_sha256_recomputed"] is False
    assert result["parent_ancestry"]["final_chain_reuse"] == "ADMISSIBLE_BY_DATA_ANCESTRY"
    assert result["parent_ancestry"]["oof_fold_reuse"].startswith("FORBIDDEN")
    assert {item["fit"] for item in result["historical_event_producers"]} == {
        "a5_seed7",
        "c2f_seed7",
    }
    assert json.loads(output.read_text(encoding="utf-8"))["test_ground_truth_read"] is False


def test_build_preflight_fails_on_role_group_leakage(tmp_path: Path) -> None:
    workspace = _fixture_workspace(tmp_path)
    run13 = workspace / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    h_manifest_path = run13 / "H_MANIFEST.json"
    h_manifest = json.loads(h_manifest_path.read_text(encoding="utf-8"))
    h_manifest["group_ids"].append("group-00")
    _write_json(h_manifest_path, h_manifest)

    with pytest.raises(ValueError, match="group list differs"):
        build_preflight(workspace)


def test_build_preflight_fails_on_missing_declared_cache_shard(tmp_path: Path) -> None:
    workspace = _fixture_workspace(tmp_path)
    (workspace / "cache/input/shard.npz").unlink()

    with pytest.raises(FileNotFoundError, match="producer input shard is missing"):
        build_preflight(workspace)

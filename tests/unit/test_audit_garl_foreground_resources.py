from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).parents[2] / "scripts" / "audit_garl_foreground_resources.py"
    spec = importlib.util.spec_from_file_location("audit_garl_foreground_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_script()


def test_rejects_non_train_parquet(tmp_path: Path) -> None:
    path = tmp_path / "test" / "train.parquet"
    path.parent.mkdir()
    path.touch()
    try:
        audit._require_public_train_parquet(path)
    except ValueError as exc:
        assert "public train.parquet" in str(exc)
    else:
        raise AssertionError("test path must be rejected")


def test_mask_and_rgb_audits_are_exact(tmp_path: Path) -> None:
    eap_root = tmp_path / "eap"
    shard_reference = "data/train/seq/rgb_shards/rgb-00000.tar"
    shard_path = eap_root / shard_reference
    shard_path.parent.mkdir(parents=True)
    payload = tmp_path / "frame.png"
    payload.write_bytes(b"png")
    with tarfile.open(shard_path, "w") as archive:
        archive.add(payload, arcname="rgb/frame.png")
    mask_root = tmp_path / "masks"
    mask = mask_root / "seq" / "image_blobs" / "frame.npy"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"npy")
    frame = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "mask_paths": [["image_blobs/frame.npy"]],
            "rgb_shard_paths": [[shard_reference]],
            "rgb_member_paths": [["rgb/frame.png"]],
        }
    )

    masks = audit.audit_mask_paths(frame, [mask_root])
    rgb = audit.audit_rgb_tar_members(frame, eap_root)

    assert masks["official_masks_available"] is True
    assert masks["resolved_unique_count"] == 1
    assert rgb["all_train_rgb_inputs_available"] is True
    assert rgb["found_unique_member_count"] == 1


def test_teacher_snapshot_requires_processor_and_license(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "sam", "architectures": ["SamModel"]}), encoding="utf-8"
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    incomplete = audit.audit_teacher_snapshot("example/model", "revision", snapshot)
    assert incomplete["self_contained_for_offline_loading_and_license_audit"] is False

    (snapshot / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "SamImageProcessor"}), encoding="utf-8"
    )
    (snapshot / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    complete = audit.audit_teacher_snapshot("example/model", "revision", snapshot)
    assert complete["self_contained_for_offline_loading_and_license_audit"] is True
    assert complete["license"]["metadata_license"] == "apache-2.0"

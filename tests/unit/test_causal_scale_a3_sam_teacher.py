from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import Dataset

import e_jepa_ttc.training.causal_scale_eap as training
import scripts.train_causal_scale_eap_screen as runner
from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4
from e_jepa_ttc.data.sam_teacher_cache import SAMTeacherMaskDataset
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.training.causal_scale_eap import CausalScaleEAPTrainingConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Dataset(Dataset[dict[str, Any]]):
    def __init__(self, square: torch.Tensor | None = None) -> None:
        self.square = square if square is not None else torch.tensor([0.0, 0.0, 16.0, 16.0])

    def __len__(self) -> int:
        return 1

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        return ((0,),)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return {
            "event_v4_common_roi": torch.rand(3, 12, 16, 16),
            "garl_delta_t_s": 0.1,
            "observable_motion": torch.zeros(18),
            "garl_visible_heights_px": torch.tensor([8.0, 9.0]),
            "ttc_s": 2.0,
            "event_v4_boxes_xyxy": torch.tensor(
                [[2.0, 2.0, 12.0, 12.0], [2.0, 2.0, 12.0, 12.0], [1.0, 1.0, 13.0, 13.0]]
            ),
            "event_v4_common_square_xyxy": self.square,
            "sequence_id": "train-seq",
            "sample_token": "train-token",
            "track_id": "track",
        }


def _teacher_cache(tmp_path: Path) -> tuple[Path, str, str]:
    mask = np.zeros((1, 2, 16, 16), dtype=np.uint8)
    mask[:, :, 3:13, 4:12] = 1
    packed = np.packbits(mask.reshape(1, 2, -1), axis=-1, bitorder="little")
    npz = tmp_path / "shard-00000.npz"
    np.savez(
        npz,
        sample_tokens=np.asarray(["train-token"]),
        sequence_ids=np.asarray(["train-seq"]),
        masks_packbits=packed,
        training_mask_valid=np.asarray([[True, False]]),
        common_square_xyxy=np.asarray([[0.0, 0.0, 16.0, 16.0]], dtype=np.float32),
    )
    manifest: dict[str, Any] = {
        "artifact_type": "sam_train_bbox_prompt_cache_manifest_v1",
        "status": "passed",
        "scope": {
            "row_count": 1,
            "public_train_only": True,
            "validation_or_test_opened": False,
        },
        "claim_boundary": {
            "validation_or_test_teacher_generation": False,
            "ttc_labels_read": False,
        },
        "cache": {
            "roi_size": 16,
            "packbits_bitorder": "little",
            "shards": [{"npz_path": npz.name, "npz_sha256": _sha256(npz)}],
        },
    }
    sign_artifact(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path, str(manifest["artifact_sha256"]), _sha256(path)


def test_sam_teacher_cache_joins_exact_token_and_collates(tmp_path: Path) -> None:
    path, artifact, file_hash = _teacher_cache(tmp_path)
    wrapped = SAMTeacherMaskDataset(
        _Dataset(),
        manifest_path=path,
        expected_artifact_sha256=artifact,
        expected_manifest_sha256=file_hash,
    )
    batch = collate_object_event_v4([wrapped[0]])

    assert wrapped.shard_index_groups() == ((0,),)
    assert batch.sam_teacher_masks is not None
    assert batch.sam_teacher_masks.shape == (1, 2, 1, 16, 16)
    assert batch.sam_teacher_masks.sum().item() == 160
    assert batch.sam_teacher_mask_valid is not None
    assert batch.sam_teacher_mask_valid.tolist() == [[True, False]]
    assert set(batch.event_inputs()) == {"events", "delta_t_s"}


def test_sam_teacher_cache_rejects_crop_mismatch(tmp_path: Path) -> None:
    path, artifact, file_hash = _teacher_cache(tmp_path)
    wrapped = SAMTeacherMaskDataset(
        _Dataset(torch.tensor([1.0, 0.0, 17.0, 16.0])),
        manifest_path=path,
        expected_artifact_sha256=artifact,
        expected_manifest_sha256=file_hash,
    )
    with pytest.raises(ValueError, match="common crop mismatch"):
        wrapped[0]


def test_a3_targets_add_train_teacher_to_a1_geometry_only() -> None:
    record = _Dataset()[0]
    record["sam_teacher_masks"] = torch.ones(2, 1, 16, 16)
    record["sam_teacher_mask_valid"] = torch.tensor([True, False])
    batch = collate_object_event_v4([record])
    targets = training._targets(
        batch,
        mask_t0_as_proxy=True,
        foreground_supervision="bbox_geometry_sam_teacher",
    )

    assert targets.geometry is not None
    assert targets.geometry.valid.tolist() == [[False, True, True]]
    assert targets.target_masks is not None
    assert targets.target_masks.shape == (1, 3, 1, 16, 16)
    assert targets.target_masks[:, 0].sum().item() == 0
    assert targets.mask_valid is not None
    assert targets.mask_valid.tolist() == [[False, True, False]]


def test_a3_validation_without_teacher_remains_geometry_only() -> None:
    batch = collate_object_event_v4([_Dataset()[0]])
    targets = training._targets(
        batch,
        mask_t0_as_proxy=True,
        foreground_supervision="bbox_geometry_sam_teacher",
    )
    assert targets.geometry is not None
    assert targets.target_masks is None
    assert targets.mask_valid is None


def test_a3_config_requires_frozen_teacher_identity() -> None:
    with pytest.raises(ValueError, match="declared together"):
        CausalScaleEAPTrainingConfig(
            foreground_supervision="bbox_geometry_sam_teacher"
        )
    config = CausalScaleEAPTrainingConfig(
        foreground_supervision="bbox_geometry_sam_teacher",
        teacher_cache_artifact_sha256="a" * 64,
    )
    assert config.teacher_cache_artifact_sha256 == "a" * 64


def test_a3_preregistered_config_is_single_teacher_difference() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a3_sam_teacher_v1.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    train = CausalScaleEAPTrainingConfig(**raw["training"])
    loss = CausalScaleTTCLossConfig(**raw["loss"])

    runner._validate_bbox_geometry_loss(train, loss, raw["decision_contract"])
    assert train.foreground_supervision == "bbox_geometry_sam_teacher"
    assert loss.foreground_bce_weight == 1.0
    assert loss.foreground_dice_weight == 0.5
    assert loss.foreground_pair_ratio_weight == 0.0
    assert raw["decision_contract"]["validation_teacher_loaded"] is False
    assert raw["experiment"]["modality"] == "event-only inference with RGB distillation"

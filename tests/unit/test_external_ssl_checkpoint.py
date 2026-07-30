from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.data.carla_looming import CARLA_LOOMING_DATASET_ID
from e_jepa_ttc.training.checkpoints import validate_external_ssl_checkpoint


def _checkpoint(split: Path) -> dict[str, object]:
    return {
        "checkpoint_role": "best",
        "checkpoint_selected_by": "validation_loss",
        "seed": 42,
        "external_pretraining": True,
        "pretraining_dataset_id": CARLA_LOOMING_DATASET_ID,
        "model_name": "event-tubelet-transformer",
        "in_channels": 21,
        "bins": 5,
        "encoder_state_dict": {"event_embed.weight": torch.zeros(2, 2)},
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "uses_velocity_feature": False,
        "uses_object_diameter_feature": False,
        "benchmark10_opened": False,
    }


def test_external_ssl_checkpoint_accepts_signed_label_free_carla(tmp_path: Path) -> None:
    split = tmp_path / "carla_split.json"
    split.write_text('{"split_version":"test"}\n', encoding="utf-8")
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint identity")

    result = validate_external_ssl_checkpoint(
        path,
        _checkpoint(split),
        source_split_path=split,
    )

    assert result["recommended_for_downstream"] is True
    assert result["uses_evttc_pretraining_events"] is False
    assert result["benchmark10_opened"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("uses_ttc_labels", True), ("benchmark10_opened", True)),
)
def test_external_ssl_checkpoint_rejects_label_or_test_exposure(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    split = tmp_path / "carla_split.json"
    split.write_text('{"split_version":"test"}\n', encoding="utf-8")
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint identity")
    checkpoint = _checkpoint(split)
    checkpoint[field] = value

    with pytest.raises(ValueError, match="violates provenance"):
        validate_external_ssl_checkpoint(path, checkpoint, source_split_path=split)


def test_external_ssl_checkpoint_rejects_split_mismatch(tmp_path: Path) -> None:
    split = tmp_path / "carla_split.json"
    split.write_text('{"split_version":"test"}\n', encoding="utf-8")
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint identity")
    checkpoint = _checkpoint(split)
    split.write_text('{"split_version":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source split hash"):
        validate_external_ssl_checkpoint(path, checkpoint, source_split_path=split)

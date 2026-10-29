from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.data.carla_looming import CARLA_LOOMING_DATASET_ID
from e_jepa_ttc.training.checkpoints import (
    validate_external_eap_checkpoint,
    validate_external_ssl_checkpoint,
    validate_external_ttc_checkpoint,
)
from e_jepa_ttc.utils.io import read_structured, write_structured


def _checkpoint(split: Path) -> dict[str, object]:
    split_payload = read_structured(split)
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
        "split_artifact_sha256": split_payload["artifact_sha256"],
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "uses_velocity_feature": False,
        "uses_object_diameter_feature": False,
        "benchmark10_opened": False,
    }


def test_external_ssl_checkpoint_accepts_signed_label_free_carla(tmp_path: Path) -> None:
    split = tmp_path / "carla_split.json"
    write_structured(split, {"artifact_type": "test_split", "split_version": "test"})
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
    write_structured(split, {"artifact_type": "test_split", "split_version": "test"})
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint identity")
    checkpoint = _checkpoint(split)
    checkpoint[field] = value

    with pytest.raises(ValueError, match="violates provenance"):
        validate_external_ssl_checkpoint(path, checkpoint, source_split_path=split)


def test_external_ssl_checkpoint_rejects_split_mismatch(tmp_path: Path) -> None:
    split = tmp_path / "carla_split.json"
    write_structured(split, {"artifact_type": "test_split", "split_version": "test"})
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint identity")
    checkpoint = _checkpoint(split)
    write_structured(split, {"artifact_type": "test_split", "split_version": "changed"})

    with pytest.raises(ValueError, match="source split artifact"):
        validate_external_ssl_checkpoint(path, checkpoint, source_split_path=split)


def test_external_ttc_checkpoint_accepts_disclosed_carla_auxiliary(
    tmp_path: Path,
) -> None:
    split = tmp_path / "carla_split.json"
    write_structured(split, {"artifact_type": "test_split", "split_version": "test"})
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint identity")
    checkpoint = _checkpoint(split)
    checkpoint.update(
        {
            "pretraining_regime": "ssl_plus_synthetic_ttc",
            "uses_ttc_labels": True,
            "uses_collision_labels": True,
            "synthetic_ttc_definition": (
                "collision_end_timestamp_minus_causal_reference_timestamp"
            ),
        }
    )

    result = validate_external_ttc_checkpoint(
        path,
        checkpoint,
        source_split_path=split,
    )

    assert result["uses_ttc_labels"] is True
    assert result["pretraining_regime"] == "ssl_plus_synthetic_ttc"


@pytest.mark.parametrize("regime", ("eap_ssl", "eap_geo", "eap_geo_v2"))
def test_external_eap_checkpoint_accepts_train_only_pretraining(
    tmp_path: Path,
    regime: str,
) -> None:
    inventory_hash = "a" * 64
    split = tmp_path / "eap_split.json"
    write_structured(
        split,
        {
            "artifact_type": "eap_pilot12_sequence_split_v1",
            "inventory_artifact_sha256": inventory_hash,
        },
    )
    split_payload = read_structured(split)
    path = tmp_path / "best.pt"
    path.write_bytes(b"eap checkpoint identity")
    geometry = regime in {"eap_geo", "eap_geo_v2"}
    checkpoint = {
        "checkpoint_role": "best",
        "checkpoint_selected_by": "validation_loss",
        "seed": 42,
        "external_pretraining": True,
        "pretraining_dataset_id": "EAP_PUBLIC_TRAIN40",
        "pretraining_regime": regime,
        "model_name": "event-tubelet-transformer",
        "in_channels": 21,
        "bins": 5,
        "encoder_state_dict": {"event_embed.weight": torch.zeros(2, 2)},
        "split_artifact_sha256": split_payload["artifact_sha256"],
        "inventory_artifact_sha256": inventory_hash,
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "uses_object_bboxes": geometry,
        "uses_depth_track_derivatives": geometry,
        "uses_labels_for_window_sampling": False,
        "uses_ttc_for_sampling": False,
        "uses_boxes_for_sampling": geometry,
        "uses_category_for_sampling": regime == "eap_geo_v2",
        "uses_depth_for_sampling": regime == "eap_geo",
        "uses_masks_for_sampling": False,
        "uses_3d_for_sampling": geometry,
        "uses_future_labels_for_sampling": False,
        "uses_rgb": False,
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
    }

    result = validate_external_eap_checkpoint(
        path,
        checkpoint,
        source_split_path=split,
        expected_regime=regime,
    )

    assert result["pretraining_regime"] == regime
    assert result["uses_object_bboxes"] is geometry


@pytest.mark.parametrize(
    "field",
    (
        "uses_ttc_for_sampling",
        "uses_boxes_for_sampling",
        "uses_category_for_sampling",
        "uses_depth_for_sampling",
        "uses_masks_for_sampling",
        "uses_3d_for_sampling",
        "uses_future_labels_for_sampling",
    ),
)
def test_external_eap_ssl_checkpoint_fails_closed_on_sampling_provenance(
    tmp_path: Path,
    field: str,
) -> None:
    inventory_hash = "a" * 64
    split = tmp_path / "eap_split.json"
    write_structured(
        split,
        {
            "artifact_type": "eap_pilot12_sequence_split_v1",
            "inventory_artifact_sha256": inventory_hash,
        },
    )
    split_payload = read_structured(split)
    path = tmp_path / "best.pt"
    path.write_bytes(b"eap checkpoint identity")
    checkpoint = {
        "checkpoint_role": "best",
        "checkpoint_selected_by": "validation_loss",
        "external_pretraining": True,
        "pretraining_dataset_id": "EAP_PUBLIC_TRAIN40",
        "pretraining_regime": "eap_ssl",
        "model_name": "event-tubelet-transformer",
        "in_channels": 21,
        "bins": 5,
        "encoder_state_dict": {"event_embed.weight": torch.zeros(2, 2)},
        "split_artifact_sha256": split_payload["artifact_sha256"],
        "inventory_artifact_sha256": inventory_hash,
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "uses_object_bboxes": False,
        "uses_depth_track_derivatives": False,
        "uses_labels_for_window_sampling": False,
        "uses_rgb": False,
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
        "uses_ttc_for_sampling": False,
        "uses_boxes_for_sampling": False,
        "uses_category_for_sampling": False,
        "uses_depth_for_sampling": False,
        "uses_masks_for_sampling": False,
        "uses_3d_for_sampling": False,
        "uses_future_labels_for_sampling": False,
    }
    checkpoint.pop(field)

    with pytest.raises(ValueError, match="explicit booleans"):
        validate_external_eap_checkpoint(
            path,
            checkpoint,
            source_split_path=split,
            expected_regime="eap_ssl",
        )

    checkpoint[field] = True
    with pytest.raises(ValueError, match="uses labels for sampling"):
        validate_external_eap_checkpoint(
            path,
            checkpoint,
            source_split_path=split,
            expected_regime="eap_ssl",
        )

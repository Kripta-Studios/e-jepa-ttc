from __future__ import annotations

import json

import numpy as np
import pytest

from e_jepa_ttc.detection.campaign_data import (
    _resized_intrinsic,
    _select_train,
    build_campaign_manifest,
    evenly_spaced,
    load_candidate_split,
)


def rows(sequence: str, count: int, role: str, rare: bool = False):
    result = []
    for index in range(count):
        result.append(
            {
                "sample_token": f"{sequence}:{index:03d}",
                "sequence_id": sequence,
                "role": role,
                "rgb_exposure_start_timestamp_us": index,
                "raw_counts": {"car": 1},
                "evaluable_counts": {"car": 1},
                "trainable_counts": {
                    "car": 1,
                    **({"other_vehicle": 1} if rare and index == 1 else {}),
                },
                "trainable_identities": {"other_vehicle": [f"id-{sequence}"]}
                if rare and index == 1
                else {},
                "trainer_overflow": False,
            }
        )
    return result


def test_evenly_spaced_is_deterministic_unique_and_has_endpoints():
    values = rows("a", 10, "train")
    selected = evenly_spaced(list(reversed(values)), 4)
    assert [r["sample_token"] for r in selected] == ["a:000", "a:003", "a:006", "a:009"]
    assert evenly_spaced(values, 4) == selected


def test_load_split_rejects_role_reinterpretation(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "schema": "eap_detection_sequence_split_v1",
                "source_split": "train",
                "dataset_revision": "a" * 40,
                "assignments": {"train": ["x"], "selection": ["y"], "calibration": ["z"]},
            }
        )
    )
    with pytest.raises(ValueError, match="24/8/8"):
        load_candidate_split(path)


def test_campaign_intrinsic_matches_pillow_pixel_center_resize():
    intrinsic = np.asarray([[1000.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]])
    resized = _resized_intrinsic(intrinsic)
    assert resized[0].tolist() == [500.0, 0.0, 319.75]
    assert resized[1].tolist() == pytest.approx([0.0, 480.0, 191.76666666666665])


def test_campaign_roles_are_disjoint_and_rare_identity_is_covered():
    train = [f"t{i:02d}" for i in range(24)]
    selection = [f"s{i:02d}" for i in range(8)]
    calibration = [f"c{i:02d}" for i in range(8)]
    records = []
    for sequence in train:
        records.extend(rows(sequence, 4, "train", rare=True))
    for sequence in selection:
        records.extend(rows(sequence, 4, "selection"))
    # Add one trainable sample for every remaining class to satisfy the gate.
    for category in ("bicycle", "truck", "pedestrian", "motorcycle", "bus", "tricycle"):
        records[0]["trainable_counts"][category] = 1
        records[0]["raw_counts"][category] = 1
        records[0]["evaluable_counts"][category] = 1
    split = {"assignments": {"train": train, "selection": selection, "calibration": calibration}}
    report = {
        "dataset_revision": "a" * 40,
        "source_media_sha256": "b" * 64,
        "candidate_split_sha256": "c" * 64,
    }
    pilot = {"selection": {"sample_tokens": ["s00:000"]}}
    manifest, coverage = build_campaign_manifest(
        report=report,
        records=records,
        split=split,
        pilot_manifest=pilot,
        train_budget=72,
        selection_budget=24,
    )
    train_tokens = manifest["roles"]["train"]["sample_tokens"]
    selection_tokens = manifest["roles"]["selection"]["sample_tokens"]
    assert not set(train_tokens) & set(selection_tokens)
    assert all(f"{sequence}:001" in train_tokens for sequence in train)
    assert coverage["train"]["trainable_annotations_legacy_profile"]["other_vehicle"] == 24
    assert manifest["historical_panels"]["pilot_selection256"]["sample_tokens"] == ["s00:000"]


def test_train_fill_spans_long_sequences_instead_of_clustering_at_middle():
    values = rows("a", 1000, "train")
    for category in (
        "bicycle",
        "truck",
        "pedestrian",
        "motorcycle",
        "bus",
        "tricycle",
        "other_vehicle",
    ):
        values[500]["trainable_counts"][category] = 1
    selected = _select_train(values, 100)
    indices = sorted(int(token.split(":")[1]) for token in selected)
    assert indices[0] == 0
    assert indices[-1] == 999
    assert max(b - a for a, b in zip(indices, indices[1:], strict=False)) < 25

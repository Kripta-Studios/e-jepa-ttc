"""CPU contracts for the isolated temporal campaign batching path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample
from e_jepa_ttc.detection.temporal_campaign import (
    HISTORY_MEDIA_RECEIPT_V2_SHA256,
    TEMPORAL_CAMPAIGN_SCHEMA,
    TEMPORAL_MEDIA_ADMISSION_V2_SHA256,
    HistoryMediaAdmission,
    TemporalCampaignConfig,
    TemporalCampaignDataset,
    TemporalCampaignSampler,
    TemporalMediaError,
    load_temporal_admission_ledger,
)
from e_jepa_ttc.detection.temporal_data import TemporalRGBIndex


def rows() -> list[dict[str, object]]:
    result = []
    for sequence, values in {"a": [(0, 10), (20, 30), (40, 50), (60, 70)], "b": [(0, 10)]}.items():
        for index, (start, end) in enumerate(values):
            result.append(
                {
                    "sample_token": f"{sequence}:{index}",
                    "sequence_id": sequence,
                    "rgb_exposure_start_timestamp_us": start,
                    "rgb_exposure_end_timestamp_us": end,
                }
            )
    return result


def raw(token: str, *, labels: bool) -> RawRGBSample:
    sequence, index = token.split(":")
    row = next(item for item in rows() if item["sample_token"] == token)
    transform = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]]
    )
    annotations = (
        [{"category": "car", "bbox_3d_ego": [0, 20, 0, 4, 1.5, 2, 0]}]
        if labels
        else []
    )
    return RawRGBSample(
        token,
        sequence,
        np.full((32, 48, 3), int(index), dtype=np.uint8),
        annotations,
        np.array([[40, 0, 24], [0, 40, 16], [0, 0, 1.0]]),
        transform,
        int(row["rgb_exposure_start_timestamp_us"]),
        int(row["rgb_exposure_end_timestamp_us"]),
        {},
        (int(index) + 1).to_bytes(32, "big").hex(),
    )


class Reader:
    def __init__(self, *, labels_enabled: bool, fail: set[str] | None = None) -> None:
        self.labels_enabled = labels_enabled
        self.fail = fail or set()
        self.reads: list[str] = []

    def read(self, token: str) -> RawRGBSample:
        self.reads.append(token)
        if token in self.fail:
            raise OSError("fixture unreadable")
        return raw(token, labels=self.labels_enabled)


def config(variant: str) -> TemporalCampaignConfig:
    return TemporalCampaignConfig.from_mapping(
        {
            "schema": TEMPORAL_CAMPAIGN_SCHEMA,
            "variant": variant,
            "producer_version": "r0-update512",
            "width": 64,
            "height": 64,
        }
    )


def admissions(*, unreadable: set[str] | None = None) -> dict[str, HistoryMediaAdmission]:
    unreadable = unreadable or set()
    result = {}
    for token in ("a:0", "a:1", "a:2", "a:3"):
        if token in unreadable:
            result[token] = HistoryMediaAdmission(
                token, False, None, "strict_decode_failed:image file is truncated"
            )
        else:
            result[token] = HistoryMediaAdmission(
                token, True, raw(token, labels=False).member_sha256, "strict_decode_pass"
            )
    return result


def dataset(
    variant: str,
    *,
    tokens: tuple[str, ...] = ("a:0", "a:1", "a:2", "a:3"),
    current_labels: bool = True,
    history_admission: dict[str, HistoryMediaAdmission] | None = None,
    generations: dict[str, int] | None = None,
) -> tuple[TemporalCampaignDataset, Reader, Reader]:
    current = Reader(labels_enabled=current_labels)
    history = Reader(labels_enabled=False)
    admission_value = history_admission or admissions()
    index = TemporalRGBIndex(rows())
    admission_rows = {}
    for token in tokens:
        resolved = index.resolve(token)
        final_mask = []
        final_ages = []
        reasons = []
        for history_token, valid, age in zip(
            resolved.history_tokens,
            resolved.history_mask,
            resolved.history_age_us,
            strict=True,
        ):
            record = admission_value.get(history_token) if history_token is not None else None
            admitted = record is not None and record.readable
            final_mask.append(valid and admitted)
            final_ages.append(age if valid and admitted else None)
            reasons.append(
                "readable"
                if valid and admitted
                else record.reason
                if record is not None
                else "sequence_boundary"
            )
        admission_rows[token] = {
            "canonical_history_tokens_before_media": resolved.history_tokens,
            "history_mask": final_mask,
            "history_age_us": final_ages,
            "history_mask_reasons": reasons,
            "sequence_id": resolved.sequence_id,
            "current_exposure_start_us": resolved.current_exposure_start_us,
            "current_exposure_end_us": resolved.current_exposure_end_us,
            "decision_sensor_us": resolved.decision_sensor_us,
        }
    value = TemporalCampaignDataset(
        current_tokens=tokens,
        exposure_rows=rows(),
        current_reader=current,
        history_reader=history,
        history_admission=admission_value,
        config=config(variant),
        current_labels=current_labels,
        role="train" if current_labels else "full_selection",
        manifest_sha256="fixture-v5",
        token_generations=generations,
        admission_rows=admission_rows,
    )
    return value, current, history


def test_strict_config_rejects_unknown_and_changed_max_age() -> None:
    value = {
        "schema": TEMPORAL_CAMPAIGN_SCHEMA,
        "variant": "correct_past_2",
        "producer_version": "p",
    }
    with pytest.raises(ValueError, match="unknown"):
        TemporalCampaignConfig.from_mapping({**value, "typo": 1})
    with pytest.raises(ValueError, match="500000"):
        TemporalCampaignConfig.from_mapping({**value, "maximum_history_age_us": 1})


def test_frozen_v2_admission_receipts_load_exact_populations() -> None:
    root = Path("output/campaign_data/temporal_d4096_v5_admission_v2")
    ledger = load_temporal_admission_ledger(
        root / "temporal_media_admission.json", root / "history_media_receipt.json"
    )
    assert len(ledger.current_rows) == 28_054
    assert len(ledger.history_media) == 32_080
    assert ledger.source_hashes == {
        "temporal_media_admission": TEMPORAL_MEDIA_ADMISSION_V2_SHA256,
        "history_media_receipt": HISTORY_MEDIA_RECEIPT_V2_SHA256,
    }


@pytest.mark.parametrize(
    ("variant", "limit", "mask", "ages"),
    [
        ("current_only", 0, None, None),
        ("current_repeated", 2, [True, True], [0.0, 0.0]),
        ("correct_past_1", 1, [False, True], [0.0, 20.0]),
        ("correct_past_2", 2, [True, True], [40.0, 20.0]),
    ],
)
def test_controls_map_to_adapter_shapes_and_actual_ages(
    variant: str, limit: int, mask: list[bool] | None, ages: list[float] | None
) -> None:
    source, _, history = dataset(variant, tokens=("a:3",))
    batch = source.collate([0])
    assert batch.current_images.shape == (1, 3, 64, 64)
    assert batch.calibs.shape == (1, 3, 4)
    assert batch.image_sizes.tolist() == [[64.0, 64.0]]
    assert batch.history_limit == limit
    assert set(batch.detector_kwargs()) == {
        "temporal_context",
        "history_images",
        "history_ages_us",
        "history_mask",
        "history_limit",
    }
    if mask is None:
        assert batch.history_images is batch.history_mask is batch.history_ages_us is None
    else:
        assert batch.history_images is not None
        assert batch.history_images.shape == (1, 2, 3, 64, 64)
        assert batch.history_mask.tolist() == [mask]
        assert batch.history_ages_us.tolist() == [ages]
    if variant in {"current_only", "current_repeated"}:
        assert history.reads == []


def test_history_is_label_free_and_targets_are_current_only() -> None:
    source, current, history = dataset("correct_past_2", tokens=("a:3",))
    batch = source.collate([0])
    assert current.labels_enabled and not history.labels_enabled
    assert current.reads == ["a:3"]
    assert history.reads == ["a:1", "a:2"]
    assert batch.targets is not None and batch.targets[0]["labels"].tolist() == [0]
    assert batch.provenance["history_labels"] is False
    assert batch.provenance["feature_cache"] is False


def test_selection_current_is_label_free_and_targets_none() -> None:
    source, current, history = dataset(
        "correct_past_2", tokens=("a:3",), current_labels=False
    )
    batch = source.collate([0])
    assert not current.labels_enabled and not history.labels_enabled
    assert batch.targets is None


def test_sequence_start_is_left_padded_without_cross_sequence_history() -> None:
    source, _, history = dataset("correct_past_2", tokens=("b:0",))
    batch = source.collate([0])
    assert batch.history_mask.tolist() == [[False, False]]
    assert batch.history_ages_us.tolist() == [[0.0, 0.0]]
    assert history.reads == []
    assert batch.temporal_context.history == ((None, None),)


def test_generation_boundary_masks_without_reusing_old_generation() -> None:
    source, _, history = dataset(
        "correct_past_2",
        tokens=("a:2",),
        generations={"a:0": 0, "a:1": 0, "a:2": 1},
    )
    batch = source.collate([0])
    assert batch.history_mask.tolist() == [[False, False]]
    assert history.reads == []
    assert batch.temporal_context.current[0].generation == 1


def test_explicit_unreadable_slot_is_masked_without_backfill_or_read() -> None:
    source, _, history = dataset(
        "correct_past_2",
        tokens=("a:3",),
        history_admission=admissions(unreadable={"a:2"}),
    )
    batch = source.collate([0])
    assert batch.history_mask.tolist() == [[True, False]]
    assert batch.history_ages_us.tolist() == [[40.0, 0.0]]
    assert history.reads == ["a:1"]
    assert batch.provenance["canonical_history_tokens"] == [["a:1", "a:2"]]
    assert "strict_decode_failed" in batch.provenance["history_slot_reasons"][0][1]


def test_missing_admission_and_runtime_read_failure_fail_closed() -> None:
    admission = admissions()
    del admission["a:2"]
    source, _, _ = dataset(
        "correct_past_2", tokens=("a:3",), history_admission=admission
    )
    with pytest.raises(TemporalMediaError, match="no explicit media admission"):
        source[0]
    source, _, history = dataset("correct_past_2", tokens=("a:3",))
    history.fail.add("a:2")
    with pytest.raises(TemporalMediaError, match="Required history RGB failed"):
        source[0]


def test_one_history_does_not_shift_unreadable_newest_into_slot() -> None:
    source, _, history = dataset(
        "correct_past_1",
        tokens=("a:3",),
        history_admission=admissions(unreadable={"a:2"}),
    )
    batch = source.collate([0])
    assert batch.history_mask.tolist() == [[False, False]]
    assert history.reads == []
    assert batch.provenance["canonical_history_tokens"] == [["a:1", "a:2"]]


def test_sampler_resume_returns_exact_next_indices_and_refuses_other_population() -> None:
    source, _, _ = dataset("current_only")
    first = TemporalCampaignSampler(source, seed=13)
    first.next_batch(2)
    state = first.state_dict()
    expected_indices, _ = first.next_batch(2)
    resumed = TemporalCampaignSampler(source, seed=13)
    resumed.load_state_dict(state)
    actual_indices, _ = resumed.next_batch(2)
    assert actual_indices == expected_indices
    other = replace(config("current_only"), variant="current_repeated")
    altered, _, _ = dataset("current_repeated")
    assert altered.config == other
    with pytest.raises(ValueError, match="another campaign"):
        TemporalCampaignSampler(altered, seed=13).load_state_dict(state)

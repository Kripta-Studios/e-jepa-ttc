"""Producer-only RGB collation preserves every consumed tensor and metadata."""

from dataclasses import fields
from types import MethodType

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.rgb_port.data import RGBProducerSource
from operational.rgb_port_revision.loss import EffectiveSource
from operational.rgb_port_rgb_loader.collate import installed_collate, producer_batch


def _sample(index, steps):
    rng = np.random.default_rng(index)
    image = rng.integers(0, 256, (steps, 3, 8, 8), dtype=np.uint8)
    return {
        "rgb": image.astype(np.float32) / np.float32(255.0),
        "rgb_uint8": image,
        "teacher_rgb_uint8": image,
        "frame_times_us": np.arange(steps, dtype=np.int64),
        "sensor_frame_times_us": np.arange(steps, dtype=np.int64),
        "annotation_frame_times_us": np.arange(steps, dtype=np.int64),
        "delta_t_s": np.full(steps - 1, 0.1, dtype=np.float32),
        "frame_valid": np.ones(steps, dtype=np.bool_),
        "foreground_mask": np.ones((steps, 1, 8, 8), dtype=np.bool_),
        "boxes_in_crop_xyxy": np.tile([0, 0, 7, 7], (steps, 1)).astype(np.float32),
        "visible_heights_px": np.full(steps, 7, dtype=np.float32),
        "log_visible_heights": np.full(steps, np.log(7), dtype=np.float32),
        "roi_xyxy": np.asarray([0, 0, 8, 8], dtype=np.float32),
        "requested_roi_xyxy": np.asarray([0, 0, 8, 8], dtype=np.float32),
        "dino_teacher_available": np.asarray(False, dtype=np.bool_),
        "target_ttc": np.asarray(index + 0.5, dtype=np.float32),
        "target_phase": np.asarray(index / 100, dtype=np.float32),
        "mass": np.asarray(0.25, dtype=np.float32),
        "query_time_us": np.asarray(index * 100000, dtype=np.int64),
        "rgb_anchor_us": np.asarray(index * 100000, dtype=np.int64),
        "query_delay_us": np.asarray(0, dtype=np.int64),
        "ordinal": index,
        "sample_token": f"sample-{index}",
        "group_id": "group",
        "role": "P",
        "track_id": f"track-{index % 2}",
        "sequence_id": f"sequence-{index % 3}",
    }


class Source:
    batch = RGBProducerSource.batch
    identity = {"unchanged": True}

    def __init__(self, counts):
        self.frame_counts = counts
        self.population_size = len(counts)
        self.input_span_us = [100000 * (count - 1) for count in counts]
        self.dataset = [_sample(index, steps) for index, steps in enumerate(counts)]
        self.teacher_calls = []
        self.fail_teacher = False

    def _dino_for_indices(self, indices):
        self.teacher_calls.append(list(indices))
        if self.fail_teacher:
            raise ValueError("generic-DINO shard row identity mismatch")
        return (
            np.stack([np.full((2, 4, 2, 2), index / 10, np.float32) for index in indices]),
            np.stack([np.ones((2, 4, 2, 2), dtype=np.bool_) for _ in indices]),
        )


def _assert_batch_equal(actual, reference):
    for field in fields(ObjectEventV4Batch):
        left, right = getattr(actual, field.name), getattr(reference, field.name)
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype
            assert left.shape == right.shape
            assert left.stride() == right.stride()
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


@pytest.mark.parametrize("steps", [2, 3])
def test_collate_matches_every_canonical_batch_field_and_keeps_rng(steps):
    source = Source([steps] * 4)
    ids = [3, 1, 1, 0]
    before = torch.get_rng_state().clone()
    reference = source.batch(ids, "rgb")
    actual = producer_batch(source, ids, "rgb")
    _assert_batch_equal(actual, reference)
    assert source.teacher_calls == [ids, ids]
    assert torch.equal(torch.get_rng_state(), before)
    source.dataset[3]["rgb"][:] = 0
    assert torch.equal(actual.events, reference.events)


def test_collate_preserves_effective_t2_t3_partition_and_teacher_order():
    from types import SimpleNamespace

    source = Source([3, 2, 3, 2, 3])
    recipe = SimpleNamespace(microbatch_size=2)
    effective = EffectiveSource(source, recipe, "revision")
    ids = [4, 1, 0, 3, 2]
    reference = effective.batch(ids, "rgb")
    with installed_collate(source):
        actual = effective.batch(ids, "rgb")
    assert len(actual.parts) == len(reference.parts) == 3
    for left, right in zip(actual.parts, reference.parts, strict=True):
        _assert_batch_equal(left, right)
    assert source.teacher_calls == [[1, 3], [4, 0], [2]] * 2


@pytest.mark.parametrize("ids,modality", [([], "rgb"), ([0], "event"), ([0, 1], "rgb")])
def test_collate_rejects_invalid_requests_before_teacher(ids, modality):
    source = Source([2, 3])
    with pytest.raises(ValueError) as canonical:
        source.batch(ids, modality)
    with pytest.raises(ValueError) as optimized:
        producer_batch(source, ids, modality)
    assert str(optimized.value) == str(canonical.value)
    assert source.teacher_calls == []


def test_collate_keeps_teacher_identity_errors_and_restores_inherited_method():
    source = Source([3])
    source.fail_teacher = True
    with pytest.raises(ValueError, match="row identity"):
        with installed_collate(source):
            source.batch([0], "rgb")
    assert "batch" not in vars(source)
    assert source.batch.__func__ is RGBProducerSource.batch


def test_collate_restores_existing_instance_override_on_failure():
    source = Source([3])
    previous = MethodType(lambda self, ids, modality: "sentinel", source)
    source.batch = previous
    with pytest.raises(RuntimeError, match="test failure"):
        with installed_collate(source):
            assert source.batch is not previous
            raise RuntimeError("test failure")
    assert source.batch is previous


def test_collate_keeps_shape_errors_for_consumed_tensors():
    source = Source([3, 3])
    source.dataset[1]["boxes_in_crop_xyxy"] = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="stack"):
        source.batch([0, 1], "rgb")
    with pytest.raises(RuntimeError, match="stack"):
        producer_batch(source, [0, 1], "rgb")
    assert source.teacher_calls == []

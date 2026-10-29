from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.data.foreground_masks import (
    BinaryMaskRLE,
    ForegroundMaskUnavailableError,
    LocalTeacherMaskAdapter,
    OfficialMaskPathResolver,
    decode_binary_mask,
    encode_binary_mask,
    make_sam_automatic_teacher,
    square_crop_resize_mask,
)
from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi


def test_official_path_resolver_is_traced_and_guards_missing_masks(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    material = root / "sequence-a" / "image_blobs" / "frame.npy"
    material.parent.mkdir(parents=True)
    material.write_bytes(b"mask")
    resolver = OfficialMaskPathResolver([root])

    trace = resolver.trace("image_blobs/frame.npy", sequence_id="sequence-a")

    assert trace.status == "resolved"
    assert trace.resolved_path == material.resolve()
    assert resolver.require("image_blobs/frame.npy", sequence_id="sequence-a") == material.resolve()
    missing = resolver.trace("image_blobs/absent.npy", sequence_id="sequence-a")
    assert missing.status == "missing"
    with pytest.raises(ForegroundMaskUnavailableError, match="must remain disabled"):
        resolver.require("image_blobs/absent.npy", sequence_id="sequence-a")


def test_official_path_resolver_rejects_root_escape(tmp_path: Path) -> None:
    resolver = OfficialMaskPathResolver([tmp_path / "masks"])
    with pytest.raises(ValueError, match="escapes"):
        resolver.trace("../outside.npy")


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 5), dtype=bool),
        np.ones((3, 5), dtype=np.uint8),
        np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8),
    ],
)
def test_binary_rle_round_trip(mask: np.ndarray) -> None:
    encoded = encode_binary_mask(mask)
    assert encoded.order == "C"
    np.testing.assert_array_equal(decode_binary_mask(encoded), mask.astype(bool))


def test_binary_rle_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        decode_binary_mask(BinaryMaskRLE(height=2, width=3, counts=(5,)))


def test_square_crop_resize_matches_release_transform() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:7, 3:8] = 1
    square = (2, 1, 8, 7)

    actual = square_crop_resize_mask(mask, square, output_size=(7, 7))
    expected = official_resize_roi(mask[None].astype(np.float32), square, (7, 7))[0]

    np.testing.assert_array_equal(actual, expected.numpy().astype(np.int64).astype(bool))


def test_local_teacher_is_target_only_box_free_and_hashed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"synthetic-teacher")
    calls: list[tuple[int, ...]] = []

    def image_only_backend(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[..., 0] > 0

    teacher = LocalTeacherMaskAdapter(
        checkpoint=checkpoint,
        backend=image_only_backend,
        backend_name="synthetic-image-only",
        model_type="unit-test",
        selection_policy="positive first channel; no annotations",
    )
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[1:3, 2:4, 0] = 255

    target = teacher.generate(image, sample_id="sample-1")

    assert calls == [(4, 5, 3)]
    assert target.teacher.uses_gt_boxes is False
    assert target.teacher.target_only is True
    assert target.teacher.checkpoint_sha256 == hashlib.sha256(b"synthetic-teacher").hexdigest()
    np.testing.assert_array_equal(decode_binary_mask(target.mask), image[..., 0] > 0)


def test_teacher_checkpoint_guard_does_not_call_backend(tmp_path: Path) -> None:
    calls = 0

    def backend(_: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((1, 1), dtype=bool)

    with pytest.raises(ForegroundMaskUnavailableError, match="does not exist"):
        LocalTeacherMaskAdapter(
            checkpoint=tmp_path / "missing.ckpt",
            backend=backend,
            backend_name="never-called",
            model_type="unit-test",
            selection_policy="none",
        )
    assert calls == 0


def test_sam_teacher_factory_is_lazy_and_records_box_free_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam.ckpt"
    checkpoint.write_bytes(b"synthetic-sam")

    teacher = make_sam_automatic_teacher(
        checkpoint=checkpoint,
        proposal_selector=lambda _proposals, image: np.zeros(image.shape[:2], dtype=bool),
        selection_policy="image-only deterministic selector",
        model_type="vit_h",
        device="cpu",
    )

    assert teacher.metadata.backend == "segment_anything.SamAutomaticMaskGenerator"
    assert teacher.metadata.uses_gt_boxes is False
    assert teacher.metadata.target_only is True
    assert teacher.metadata.checkpoint_sha256 == hashlib.sha256(b"synthetic-sam").hexdigest()

"""Bounded persistent TAR reads reproduce the original teacher preprocessing exactly."""

import io
import tarfile

import numpy as np
from PIL import Image

from operational.train40_system.teacher import rgb_crop


def test_persistent_tar_crop_matches_original_reference(tmp_path):
    import torch

    from scripts.materialize_dinov3_relational_teacher import _load_and_crop_rgb

    pixel = np.arange(80 * 90 * 3, dtype=np.uint8).reshape(80, 90, 3)
    stream = io.BytesIO()
    Image.fromarray(pixel).save(stream, format="PNG")
    raw = stream.getvalue()
    path = tmp_path / "rgb.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("rgb/frame.png")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    for square in [
        np.array([-2.7, -1.5, 50.8, 61.9], dtype=np.float32),
        np.array([20.1, 24.7, 87.9, 85.3], dtype=np.float32),
    ]:
        actual, actual_hash = rgb_crop(path, "rgb/frame.png", square)
        expected, expected_hash = _load_and_crop_rgb(
            path,
            "rgb/frame.png",
            square,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            torch.device("cpu"),
        )
        np.testing.assert_array_equal(actual, expected[0].numpy())
        assert actual_hash == expected_hash

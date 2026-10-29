"""Independent geometry and source-pair regression tests for detector preparation."""

from __future__ import annotations

import io
import pickle
import unittest

import numpy as np
import pandas as pd

from scripts.verify_eap_geometry_real import (
    AnnotationUnpickler,
    corners_ego,
    native_dimensions,
    project_box,
    verify_pairs,
)


class GeometryRealTests(unittest.TestCase):
    def fixture(self):
        box = [10, 2, 1, 6, 2, 4, 0]
        intrinsic = np.eye(3)
        transform = np.eye(4)
        source = {
            "sequence_id": "a",
            "file_name": "rgb/f.png",
            "instance_id": "a_1",
            "bbox_3d": box,
            "category": "truck",
            "velocity": [1, 2, 0],
            "K_event": intrinsic,
            "T_event_ego": transform,
            "rgb_exposure_start_timestamp_us": 100,
            "rgb_exposure_end_timestamp_us": 110,
        }
        target = {
            "sample_token": "a:f",
            "instance_id": "a_1",
            "bbox_3d_ego": box,
            "category": "truck",
            "velocity_3d": [1, 2, 0],
            "size": [4, 6, 2],
        }
        media = {
            "sample_token": "a:f",
            "K_event": intrinsic,
            "T_event_ego": transform,
            "rgb_exposure_start_timestamp_us": 100,
            "rgb_exposure_end_timestamp_us": 110,
        }
        return source, target, media

    def test_non_cube_dimensions_center_and_clockwise_rotation(self):
        c = corners_ego([10, 2, 1, 6, 2, 4, 0])
        np.testing.assert_array_equal(c[0], [8, -1, 2])
        np.testing.assert_array_equal(c.mean(0), [10, 2, 1])
        np.testing.assert_array_equal(np.ptp(c, axis=0), [4, 6, 2])
        c = corners_ego([10, 2, 1, 6, 2, 4, np.pi / 2])
        np.testing.assert_allclose(c[0], [7, 4, 2], atol=1e-12)

    def test_native_dimensions_are_not_hf_size_order(self):
        self.assertEqual(native_dimensions([10, 2, 1, 6, 2, 4, 0]), [6, 4, 2])

    def test_projection_applies_ego_to_camera_directly(self):
        transform = np.asarray([[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3], [0, 0, 0, 1]])
        intrinsic = np.asarray([[100, 0, 20], [0, 100, 10], [0, 0, 1]])
        # Corner zero: ego (8,-1,2) -> camera (2,0,11).
        uv = project_box([10, 2, 1, 6, 2, 4, 0], intrinsic, transform)
        np.testing.assert_allclose(uv[0], [20 + 200 / 11, 10], atol=1e-12)

    def test_box_crossing_camera_plane_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "behind"):
            project_box([0, 0, 0, 6, 2, 4, 0], np.eye(3), np.eye(4))

    def test_exact_pair_and_ambiguous_instance(self):
        source, target, media = self.fixture()
        result = verify_pairs([source], pd.DataFrame([target]), pd.DataFrame([media]))
        self.assertTrue(result["all_pairs_exact"])
        with self.assertRaisesRegex(ValueError, "Ambiguous HF"):
            verify_pairs([source], pd.DataFrame([target, target]), pd.DataFrame([media]))

    def test_mismatch_is_not_accepted_as_a_verified_pair(self):
        source, target, media = self.fixture()
        target["bbox_3d_ego"] = [11, 2, 1, 6, 2, 4, 0]
        target["size"] = [6, 4, 2]
        media["rgb_exposure_end_timestamp_us"] = 111
        result = verify_pairs([source], pd.DataFrame([target]), pd.DataFrame([media]))
        self.assertFalse(result["all_pairs_exact"])
        self.assertEqual(result["field_mismatches"]["box"], 1)
        self.assertEqual(result["field_mismatches"]["size"], 1)
        self.assertEqual(result["field_mismatches"]["exposure"], 1)

    def test_pickle_builtin_annotations_allowed_but_executable_globals_refused(self):
        safe = [{"box": [1, 2, 3], "class": "car"}]
        self.assertEqual(AnnotationUnpickler(io.BytesIO(pickle.dumps(safe))).load(), safe)
        with self.assertRaisesRegex(ValueError, "Executable pickle global"):
            AnnotationUnpickler(io.BytesIO(pickle.dumps(eval))).load()

    def test_negative_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            corners_ego([0, 0, 1, -6, 2, 4, 0])


if __name__ == "__main__":
    unittest.main()

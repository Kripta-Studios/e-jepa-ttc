"""Protect correspondence diagnostics from clock and geometry assumptions."""

from __future__ import annotations

import unittest
from itertools import product

import numpy as np

from scripts.audit_eap_garl_alignment import (
    build_media_index,
    compare_hf_frame,
    match_endpoint,
    projection_diagnostic,
)


class AlignmentTests(unittest.TestCase):
    def media_row(self):
        return {
            "split": "train",
            "sequence_id": "a",
            "rgb_member_path": "rgb/f.png",
            "rgb_shard_path": "data/train/a/frames.tar",
            "events_path": "data/train/a/events.h5",
            "rgb_exposure_start_timestamp_us": 1000,
            "rgb_exposure_end_timestamp_us": 1010,
        }

    def query(self):
        return {
            "sequence_id": "a",
            "rgb_member_paths": ["rgb/f.png"],
            "rgb_shard_paths": ["data/train/a/frames.tar"],
            "events_path": "data/train/a/events.h5",
            "frame_timestamps_us": [9000],
            "event_windows_us": [[900, 1000]],
        }

    def test_join_uses_media_identity_and_reports_clock_difference(self):
        result = match_endpoint(build_media_index([self.media_row()]), self.query(), 0)
        self.assertEqual(result["annotation_minus_exposure_start_us"], 8000)
        self.assertEqual(result["event_window_end_minus_exposure_start_us"], 0)

    def test_same_member_from_other_sequence_is_not_a_match(self):
        query = self.query()
        query["sequence_id"] = "b"
        with self.assertRaisesRegex(ValueError, "No train media match"):
            match_endpoint(build_media_index([self.media_row()]), query, 0)

    def test_ambiguous_identity_fails(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            build_media_index([self.media_row(), self.media_row()])

    def test_test_media_rejected(self):
        row = self.media_row()
        row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "public train"):
            build_media_index([row])

    def test_shard_mismatch_fails_even_if_member_matches(self):
        query = self.query()
        query["rgb_shard_paths"] = ["different.tar"]
        with self.assertRaisesRegex(ValueError, "Shard differs"):
            match_endpoint(build_media_index([self.media_row()]), query, 0)

    def test_reversed_exposure_fails(self):
        row = self.media_row()
        row["rgb_exposure_end_timestamp_us"] = 999
        with self.assertRaisesRegex(ValueError, "Exposure end"):
            match_endpoint(build_media_index([row]), self.query(), 0)

    def test_projection_preserves_residual_and_height_discrepancy(self):
        corners = np.array(
            [
                [-2, -1, 9],
                [2, -1, 9],
                [2, 1, 9],
                [-2, 1, 9],
                [-2, -1, 11],
                [2, -1, 11],
                [2, 1, 11],
                [-2, 1, 11],
            ]
        )
        intrinsic = np.array([[90, 0, 10], [0, 90, 10], [0, 0, 1]])
        result = projection_diagnostic(corners, intrinsic, np.eye(4), [0, 0, 30, 20], 3, (100, 80))
        self.assertTrue(result["clipping_applied"])
        self.assertEqual(result["raw_max_abs_roi_error_px"], 10)
        self.assertEqual(result["clipped_max_abs_roi_error_px"], 0)
        self.assertEqual(result["box3d_h_minus_ego_z_extent_m"], 1)

    def test_behind_camera_corner_is_not_silently_dropped(self):
        corners = np.ones((8, 3))
        corners[0, 2] = -1
        with self.assertRaisesRegex(ValueError, "behind"):
            projection_diagnostic(corners, np.eye(3), np.eye(4), [0, 0, 1, 1], 1, (10, 10))

    def test_clipping_distinguishes_extent_from_last_pixel_center(self):
        corners = np.asarray(list(product([1, 11], [1, 7], [1, 2])))
        result = projection_diagnostic(corners, np.eye(3), np.eye(4), [0, 0, 9, 7], 1, (10, 8))
        self.assertEqual(result["clipped_max_abs_roi_error_px"], 1)
        self.assertEqual(result["pixel_center_clipped_max_abs_roi_error_px"], 0.5)

    def test_id_equality_cannot_override_a_geometric_mismatch(self):
        corners = np.asarray(list(product([-3, 3], [-1, 1], [9, 11])))
        transform = np.eye(4)
        transform[:3, 3] = [2, 3, 4]
        rows = [
            {"instance_id": "a", "category": "car", "bbox_3d_ego": [30, 0, 10, 6, 2, 2, np.pi / 2]},
            {"instance_id": "b", "category": "car", "bbox_3d_ego": [0, 0, 10, 6, 2, 2, np.pi / 2]},
        ]
        result = compare_hf_frame(corners + [2, 3, 4], transform, rows, "a")
        self.assertAlmostEqual(result["same_id_hypothesis"]["center_distance_m"], 30)
        nearest = result["nearest_center_diagnostic_not_identity"]
        self.assertEqual(nearest["instance_id"], "b")
        self.assertLess(nearest["corner_set_hausdorff_m"], 1e-12)

    def test_duplicate_instance_within_frame_is_rejected(self):
        row = {"instance_id": "a", "category": "car", "bbox_3d_ego": [0, 0, 10, 6, 2, 2, 0]}
        with self.assertRaisesRegex(ValueError, "Duplicate HF instance_id"):
            compare_hf_frame(np.ones((8, 3)), np.eye(4), [row, row], "a")

    def test_missing_hf_frame_is_reported_without_fake_match(self):
        result = compare_hf_frame(np.ones((8, 3)), np.eye(4), [], "a")
        self.assertIsNone(result["same_id_hypothesis"])
        self.assertIsNone(result["nearest_center_diagnostic_not_identity"])


if __name__ == "__main__":
    unittest.main()

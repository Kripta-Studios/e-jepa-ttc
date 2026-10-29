"""Protect causal association and target-free inputs without loading PHASE/Torch."""

from __future__ import annotations

import argparse
import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.infer_detected_roi_phase import (
    box_iou,
    causal_tracks,
    run,
    suppress_duplicates,
)


class DetectedROIPhaseTests(unittest.TestCase):
    def detection(self, box: list[float], index: int = 0, category: str = "car") -> dict:
        return {
            "class_name": category,
            "score": 0.9,
            "prediction_index": index,
            "roi_xyxy": box,
        }

    def frame(self, timestamp: int, detections: list[dict]) -> dict:
        return {
            "sample_token": f"a:{timestamp}",
            "timestamp_us": timestamp,
            "detections": detections,
        }

    def associate(self, frames: list[dict]) -> list[dict]:
        return causal_tracks(frames, iou_threshold=0.3, ambiguity_margin=0.1, max_gap_us=150_000)

    def test_unambiguous_match_retains_two_real_observations(self):
        frames = [
            self.frame(100, [self.detection([0, 0, 10, 10])]),
            self.frame(100_100, [self.detection([1, 0, 11, 10])]),
        ]
        audit = self.associate(frames)
        self.assertEqual(audit[-1]["reason"], "matched")
        self.assertEqual(
            [item["timestamp_us"] for item in frames[-1]["detections"][0]["history"]],
            [100, 100_100],
        )

    def test_ambiguous_match_starts_new_track_without_history_fill(self):
        frames = [
            self.frame(100, [self.detection([0, 0, 10, 10]), self.detection([2, 0, 12, 10], 1)]),
            self.frame(100_100, [self.detection([1, 0, 11, 10])]),
        ]
        audit = self.associate(frames)
        self.assertEqual(audit[-1]["reason"], "ambiguous")
        self.assertEqual(len(frames[-1]["detections"][0]["history"]), 1)

    def test_large_gap_does_not_reuse_previous_identity(self):
        frames = [
            self.frame(100, [self.detection([0, 0, 10, 10])]),
            self.frame(200_100, [self.detection([0, 0, 10, 10])]),
        ]
        self.associate(frames)
        self.assertEqual(len(frames[-1]["detections"][0]["history"]), 1)
        self.assertNotEqual(
            frames[0]["detections"][0]["track_id"], frames[1]["detections"][0]["track_id"]
        )

    def test_nms_suppresses_same_class_duplicate_and_preserves_other_class(self):
        retained, suppressed = suppress_duplicates(
            [
                self.detection([0, 0, 10, 10]),
                self.detection([0, 0, 10, 10], 1),
                self.detection([0, 0, 10, 10], 2, "truck"),
            ],
            0.5,
        )
        self.assertEqual((len(retained), suppressed), (2, 1))
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_timestamp_rollback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            self.associate([self.frame(200, []), self.frame(100, [])])

    def test_matching_cannot_cross_class(self):
        frames = [
            self.frame(100, [self.detection([0, 0, 10, 10])]),
            self.frame(100_100, [self.detection([0, 0, 10, 10], category="truck")]),
        ]
        self.associate(frames)
        self.assertEqual(len(frames[-1]["detections"][0]["history"]), 1)

    @staticmethod
    def arguments(directory: Path, payload: dict) -> argparse.Namespace:
        predictions = directory / "predictions.json"
        predictions.write_text(json.dumps(payload), encoding="utf-8")
        return argparse.Namespace(
            eap_root=directory,
            predictions=predictions,
            sample_token="a:current",
            minimum_score=0.1,
            nms_iou=0.5,
            association_iou=0.3,
            ambiguity_margin=0.1,
        )

    def test_predictions_json_with_ground_truth_is_refused_before_manifest_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(
                Path(temporary),
                {"predictions": {"a:current": []}, "ground_truth": {}},
            )
            with (
                patch("pyarrow.parquet.read_table") as reader,
                self.assertRaisesRegex(ValueError, "predictions-only JSON"),
            ):
                run(args)
            reader.assert_not_called()

    def test_future_frame_or_unfinished_exposure_is_never_processed(self):
        # A malformed score makes accidental access to the unavailable prediction fail.
        # Run exits before model imports because the current frame has no detection.
        original_import = builtins.__import__

        def forbid_torch(name: str, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("This data-boundary test must not import Torch")
            return original_import(name, *args, **kwargs)

        for start, end in ((101, 120), (95, 111)):
            with self.subTest(start=start, end=end), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "data").mkdir()
                (directory / "data/train.parquet").write_bytes(b"synthetic manifest hash fixture")
                args = self.arguments(
                    directory,
                    {"predictions": {"a:current": [], "a:future": [{"score": "unavailable"}]}},
                )
                rows = [
                    {
                        "sample_token": "a:current",
                        "split": "train",
                        "sequence_id": "a",
                        "rgb_exposure_start_timestamp_us": 100,
                        "rgb_exposure_end_timestamp_us": 110,
                    },
                    {
                        "sample_token": "a:future",
                        "split": "train",
                        "sequence_id": "a",
                        "rgb_exposure_start_timestamp_us": start,
                        "rgb_exposure_end_timestamp_us": end,
                    },
                ]
                with (
                    patch("pyarrow.parquet.read_table") as reader,
                    patch("builtins.__import__", side_effect=forbid_torch),
                ):
                    reader.return_value.to_pylist.return_value = rows
                    report = run(args)
                self.assertEqual(report["status"], "UNAVAILABLE")
                self.assertEqual(report["reason"], "no_current_usable_detection")
                self.assertEqual(report["association"], [])
                self.assertEqual(report["rejected_rois"], [])
                self.assertNotIn("labels_path", reader.call_args.kwargs["columns"])


if __name__ == "__main__":
    unittest.main()

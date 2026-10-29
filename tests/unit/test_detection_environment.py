"""Prevent missing hardware, an import, or a primitive test from passing G1."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import check_deformable_operator as operator
from scripts import probe_detection_environment as probe


class DetectionEnvironmentTests(unittest.TestCase):
    def test_missing_command_is_explicit(self):
        with patch.object(probe.subprocess, "run", side_effect=FileNotFoundError("not found")):
            result = probe.command_record(["nvcc", "--version"])
        self.assertIsNone(result["returncode"])
        self.assertIn("not found", result["error"])

    def test_probe_timeout_is_explicit(self):
        error = subprocess.TimeoutExpired("probe", 30)
        with patch.object(probe.subprocess, "run", side_effect=error):
            result = probe.command_record(["probe"])
        self.assertIsNone(result["returncode"])
        self.assertIn("timed out", result["error"])

    def test_no_gpu_cannot_pass_g1(self):
        result = {"returncode": 0, "stdout": json.dumps({"import": "FAIL"}), "stderr": ""}
        with (
            patch.object(probe, "command_record", return_value=result),
            patch.object(probe.shutil, "which", return_value=None),
        ):
            report = probe.collect_report()
        self.assertEqual(report["g1"]["status"], "BLOCKED")
        self.assertIn("cuda_device_access_unverified", report["g1"]["blockers"])
        self.assertIsNone(report["g1"]["detector_peak_memory_bytes"])

    def test_import_and_primitive_success_are_not_detector_evidence(self):
        evidence = {
            "import": "PASS",
            "cuda_available": True,
            "primitive_status": "PASS",
            "extension_import": "PASS",
            "primitive_peak_allocated_bytes": 100,
        }
        result = {"returncode": 0, "stdout": json.dumps(evidence), "stderr": ""}
        with patch.object(probe, "command_record", return_value=result):
            report = probe.collect_report(run_cuda_primitives=True)
        self.assertEqual(report["g1"]["status"], "BLOCKED")
        self.assertEqual(report["g1"]["detector_forward"], "NOT_RUN")
        self.assertEqual(report["g1"]["detector_loss_and_backward"], "NOT_RUN")
        self.assertEqual(report["g1"]["detector_decode"], "NOT_RUN")
        self.assertIsNone(report["g1"]["detector_peak_memory_bytes"])

    def test_failed_probe_does_not_reuse_output_as_evidence(self):
        result = {"returncode": 1, "stdout": '{"import":"PASS"}', "stderr": "failure"}
        with patch.object(probe, "command_record", return_value=result):
            report = probe.collect_report()
        self.assertEqual(report["torch"]["import"], "FAIL")
        self.assertEqual(report["g1"]["status"], "BLOCKED")

    def test_invalid_json_is_not_a_success(self):
        result = {"returncode": 0, "stdout": "not json", "stderr": ""}
        with patch.object(probe, "command_record", return_value=result):
            report = probe.collect_report()
        self.assertEqual(report["torch"]["import"], "FAIL")

    def test_wrong_json_shape_is_not_a_success(self):
        result = {"returncode": 0, "stdout": "[]", "stderr": ""}
        with patch.object(probe, "command_record", return_value=result):
            report = probe.collect_report()
        self.assertEqual(report["torch"]["import"], "FAIL")

    def test_operator_missing_torch_is_blocked(self):
        with patch.dict(sys.modules, {"torch": None}):
            report = operator.run_operator_check(Path("unavailable"), "monodgp")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["g1_passed"])
        self.assertEqual(report["dtype_results"], [])

    def test_operator_without_cuda_does_not_fallback_to_cpu(self):
        fake = SimpleNamespace(
            __version__="test-only",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(sys.modules, {"torch": fake}):
            report = operator.run_operator_check(Path("unavailable"), "monodgp")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["g1_passed"])
        self.assertEqual(report["dtype_results"], [])

    def test_missing_source_is_not_a_verified_revision(self):
        result = {"returncode": 128, "stdout": "", "stderr": "missing"}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(probe, "command_record", return_value=result):
                record = probe.source_record("monodgp", Path(temporary))
        self.assertFalse(record["revision_matches"])
        self.assertIsNone(record["observed_revision"])
        self.assertTrue(all(value is None for value in record["files_sha256"].values()))

    def test_source_content_and_revision_are_recorded_independently(self):
        head = {"returncode": 0, "stdout": probe.PINNED_SOURCES["monodgp"]}
        diff = {"returncode": 0, "stdout": "modified code"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("fixture", encoding="utf-8")
            with patch.object(probe, "command_record", side_effect=[head, diff]):
                record = probe.source_record("monodgp", root)
        self.assertTrue(record["revision_matches"])
        self.assertTrue(record["tracked_changes_present"])
        self.assertIsNotNone(record["files_sha256"]["README.md"])


if __name__ == "__main__":
    unittest.main()

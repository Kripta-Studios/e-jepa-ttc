from __future__ import annotations

from tests.integration._artifact_helpers import read_artifact


def test_export_artifact_records_verified_onnxruntime_parity() -> None:
    smoke = read_artifact("artifacts/demos/runtime_smoke_current_v1/runtime_smoke_metrics.json")
    export = smoke["export"]
    assert smoke["status"] == "completed"
    assert export["verified_with_onnxruntime_cpu"] is True
    assert max(export["maximum_absolute_error"].values()) < 1e-4

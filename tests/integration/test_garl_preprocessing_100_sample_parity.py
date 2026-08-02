from __future__ import annotations

import pytest

from tests.integration._artifact_helpers import read_artifact


@pytest.mark.external_data
def test_official_preprocessing_parity_covers_100_samples() -> None:
    parity = read_artifact("artifacts/parity/garl_preprocessing_v2/parity.json")
    assert parity["status"] == "pass"
    assert parity["samples"] >= 100
    assert parity["coverage_count"] >= 12
    assert parity["raw_max_abs"] == 0.0
    assert parity["resized_max_abs"] == 0.0

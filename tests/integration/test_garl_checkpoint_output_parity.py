from __future__ import annotations

import pytest

from tests.integration._artifact_helpers import read_artifact


@pytest.mark.external_data
def test_official_checkpoint_output_parity_is_within_signed_tolerance() -> None:
    parity = read_artifact("artifacts/parity/garl_model_v1/parity.json")
    assert parity["status"] == "pass"
    assert parity["raw_height_max_abs"] <= parity["tolerance"]
    assert parity["ttc_max_abs"] <= parity["tolerance"]
    assert parity["mapped_parameter_count"] > 0

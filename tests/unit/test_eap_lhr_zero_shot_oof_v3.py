from __future__ import annotations

import pytest

from scripts.aggregate_eap_lhr_zero_shot import aggregate_payloads
from scripts.compare_eap_lhr_zero_shot import compare


def _payload(prediction: float, token: str = "a") -> dict:
    return {
        "artifact_type": "eap_lhr_object_jepa_ttc_zero_shot_v3",
        "predictions": [
            {
                "sequence_id": "seq",
                "sample_token": token,
                "track_id": "track",
                "timestamp_us": 100,
                "category": "vehicle",
                "sampling_group": "vehicle:longitudinal:visible:intersecting",
                "target_ttc_s": 2.0,
                "predicted_ttc_s": prediction,
                "absolute_error_s": abs(prediction - 2.0),
            }
        ],
    }


def test_oof_rejects_duplicate_samples() -> None:
    with pytest.raises(ValueError, match="Duplicate OOF"):
        aggregate_payloads([_payload(2.1), _payload(2.2)])


def test_paired_comparison_requires_same_samples() -> None:
    with pytest.raises(ValueError, match="sample sets differ"):
        compare(_payload(2.1, "a"), _payload(2.2, "b"), iterations=10, seed=7)

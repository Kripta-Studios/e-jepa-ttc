from __future__ import annotations

import pandas as pd

from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows, signed_ttc_bucket


def test_signed_sampling_buckets_preserve_negative_ttc() -> None:
    assert signed_ttc_bucket(-10.0) == "out_of_protocol"
    assert signed_ttc_bucket(-1.0) == "negative"
    assert signed_ttc_bucket(0.0) == "negative"
    assert signed_ttc_bucket(1.0) == "crucial"
    assert signed_ttc_bucket(4.0) == "small"
    assert signed_ttc_bucket(8.0) == "large"


def test_balanced_sampler_is_deterministic_and_split_aware() -> None:
    rows = pd.DataFrame(
        [
            {"sequence_id": sequence, "track_id": str(index), "ttc": float(index)}
            for sequence in ("train-seq", "validation-seq")
            for index in range(8)
        ]
    )
    assignments = {"train-seq": "train", "validation-seq": "validation"}
    selected_a, report_a = select_balanced_cache_rows(
        rows, assignments, seed=7, max_samples_per_split=4
    )
    selected_b, report_b = select_balanced_cache_rows(
        rows, assignments, seed=7, max_samples_per_split=4
    )
    assert selected_a["track_id"].tolist() == selected_b["track_id"].tolist()
    assert report_a == report_b
    assert selected_a["sequence_id"].value_counts().to_dict() == {
        "train-seq": 4,
        "validation-seq": 4,
    }

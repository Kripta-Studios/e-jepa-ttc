from __future__ import annotations

from e_jepa_ttc.data.eap_pilot import select_eap_pilot_sequences


def _row(index: int, *, gib: float = 1.0) -> dict[str, object]:
    return {
        "sequence_id": f"sequence-{index:02d}",
        "event_file_gib": gib,
        "event_count": 1000 * (index + 1),
        "label_count": 100 + index,
        "track_count": 10 + index,
        "projected_state_count": 100 + index,
        "object_window_count": 20 + index,
        "category_counts": {"car": 80, "pedestrian": index + 1},
        "ttc_proxy_counts": {
            "approaching_0p1_2s": index + 1,
            "approaching_2_4s": 2,
            "approaching_4_8s": 3,
            "approaching_8_20s": 4,
            "receding_0p1_20s": 5,
        },
    }


def test_eap_pilot_selection_is_disjoint_deterministic_and_bounded() -> None:
    rows = [_row(index) for index in range(14)]
    rows[-1] = _row(13, gib=40.0)

    first = select_eap_pilot_sequences(
        rows,
        sequence_count=12,
        validation_count=3,
        anchor_sequence_ids=("sequence-00", "sequence-01"),
        maximum_event_gib=20.0,
    )
    second = select_eap_pilot_sequences(
        rows,
        sequence_count=12,
        validation_count=3,
        anchor_sequence_ids=("sequence-00", "sequence-01"),
        maximum_event_gib=20.0,
    )

    assert first == second
    assert len(first["train"]) == 9
    assert len(first["validation"]) == 3
    assert set(first["train"]).isdisjoint(first["validation"])
    assert "sequence-13" not in first["selected"]
    assert "sequence-13" in first["excluded_large_outliers"]

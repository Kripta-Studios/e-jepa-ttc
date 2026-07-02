from pathlib import Path

from e_jepa_ttc.data.official_protocol import evaluate_official_evttc_coverage
from e_jepa_ttc.data.types import DatasetSequence


def _sequence(root: Path, family: str, speed_dir: str, *, labels: int = 1) -> DatasetSequence:
    sequence_dir = root / family / speed_dir / "overlap-100"
    sequence_dir.mkdir(parents=True)
    event_hdf5 = sequence_dir / "data.hdf5"
    ttc_csv = sequence_dir / "ttc.csv"
    label_dir = sequence_dir / "bbox_segmentation"
    event_hdf5.write_bytes(b"events")
    ttc_csv.write_text("0 0.0 1.0 1.0 1.0\n", encoding="utf-8")
    label_dir.mkdir()
    for index in range(labels):
        (label_dir / f"{index:04d}.json").write_text('{"objects": []}', encoding="utf-8")
    return DatasetSequence(
        dataset="EvTTC",
        sequence_id=f"{family}-{speed_dir}-overlap-100",
        local_path=sequence_dir.as_posix(),
        event_hdf5=event_hdf5.name,
        ttc_csv=ttc_csv.name,
        label_dir=label_dir.name,
        scenario_family=family,
        speed_bucket=speed_dir.split("-")[0],
        target_type="car",
        split_group=f"{family}-{speed_dir}-overlap-100",
        extra={
            "relative_parts": [family, speed_dir, "overlap-100"],
            "label_count": labels,
        },
    )


def test_official_evttc_coverage_blocks_partial_local_assets(tmp_path: Path) -> None:
    sequences = [
        _sequence(tmp_path, "CCRs-1", "low-100"),
        _sequence(tmp_path, "CCRs-1", "medium-100"),
        _sequence(tmp_path, "CCRs-1", "high-100"),
    ]

    report = evaluate_official_evttc_coverage(sequences, include_slider=True)

    assert report["official_real_world_required_sequence_count"] == 8
    assert report["official_real_world_complete_sequence_count"] == 3
    assert report["official_real_world_coverage_percent"] == 37.5
    assert report["official_table_v_required_sequence_count"] == 10
    assert report["official_table_v_complete_sequence_count"] == 3
    assert report["official_table_v_coverage_percent"] == 30.0
    assert report["official_sota_claim_allowed"] is False
    assert "CCRs-2-low-100%" in report["missing_real_world_sequences"]
    assert "Slider-750" in report["missing_table_v_sequences"]


def test_official_evttc_coverage_requires_nonempty_bbox_labels(tmp_path: Path) -> None:
    sequences = [_sequence(tmp_path, "CCRs-1", "low-100", labels=0)]

    report = evaluate_official_evttc_coverage(sequences, include_slider=False)
    row = next(row for row in report["rows"] if row["name"] == "CCRs-1-low-100%")

    assert row["matched_sequence_id"] == "CCRs-1-low-100-overlap-100"
    assert row["assets"]["event_hdf5"] is True
    assert row["assets"]["gt_ttc"] is True
    assert row["assets"]["bbox_roi"] is False
    assert row["missing_assets"] == ["bbox_roi"]
    assert row["status"] == "incomplete_assets"

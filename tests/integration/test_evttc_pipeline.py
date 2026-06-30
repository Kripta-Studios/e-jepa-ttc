from pathlib import Path

from e_jepa_ttc.data.evttc import (
    discover_event_layout,
    read_manifest,
    scan_evttc_root,
    validate_manifest,
    write_manifest,
)
from e_jepa_ttc.data.index import build_temporal_index
from e_jepa_ttc.data.split import create_sequence_splits, validate_split_groups
from e_jepa_ttc.data.synthetic import generate_synthetic_sequence, write_synthetic_hdf5


def _write_sequence(root: Path, family: str, speed: str) -> None:
    sequence_dir = root / family / speed / "overlap-100"
    sequence_dir.mkdir(parents=True)
    sequence = generate_synthetic_sequence(windows=16, seed=len(speed))
    write_synthetic_hdf5(sequence_dir / f"{speed}.hdf5", sequence)
    write_synthetic_hdf5(sequence_dir / "gt.hdf5", sequence)
    rows = []
    for frame_id, timestamp_s, distance, relative_speed, ttc_s in zip(
        sequence.frame_id,
        sequence.timestamp_s,
        sequence.distance,
        sequence.relative_speed,
        sequence.ttc_s,
        strict=True,
    ):
        rows.append(f"{frame_id} {timestamp_s:.6f} {distance:.6f} {relative_speed:.6f} {ttc_s:.6f}")
    (sequence_dir / "ttc.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    label_dir = sequence_dir / "leftlabel"
    label_dir.mkdir()
    (label_dir / "0000.json").write_text('{"objects": []}', encoding="utf-8")


def test_scan_validate_index_and_split(tmp_path: Path) -> None:
    root = tmp_path / "evttc"
    _write_sequence(root, "CCRs-1", "low-100")
    _write_sequence(root, "CCRs-1", "medium-100")
    _write_sequence(root, "CCRs-1", "high-100")

    sequences = scan_evttc_root(root)
    assert [sequence.speed_bucket for sequence in sequences] == ["high", "low", "medium"]

    manifest = tmp_path / "manifest.yaml"
    write_manifest(manifest, sequences)
    loaded = read_manifest(manifest)
    assert len(loaded) == 3

    report = validate_manifest(manifest)
    assert report["sequence_count"] == 3
    assert report["sequences"][0]["event_layout"]["kind"] == "separate"

    entries = build_temporal_index(
        manifest_path=manifest,
        context_ms=20,
        stride_ms=20,
        horizons_ms=(20,),
        clip_ttc_seconds=(0.1, 10.0),
    )
    assert entries

    splits = create_sequence_splits(loaded, seed=42)
    validate_split_groups(loaded, splits)
    assert splits["train"] == ["CCRs-1-low-100-overlap-100"]
    assert splits["validation"] == ["CCRs-1-medium-100-overlap-100"]
    assert splits["test"] == ["CCRs-1-high-100-overlap-100"]


def test_discover_event_layout_on_fixture(tmp_path: Path) -> None:
    sequence = generate_synthetic_sequence(windows=4, seed=1)
    path = tmp_path / "events.hdf5"
    write_synthetic_hdf5(path, sequence)

    layout = discover_event_layout(path)

    assert layout is not None
    assert layout.x == "events/x"
    assert layout.t == "events/t"

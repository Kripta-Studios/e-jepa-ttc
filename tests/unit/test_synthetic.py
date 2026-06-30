from pathlib import Path

from e_jepa_ttc.data.synthetic import (
    generate_synthetic_sequence,
    read_synthetic_hdf5,
    write_synthetic_hdf5,
)
from e_jepa_ttc.data.validation import validate_event_batch


def test_generate_synthetic_sequence_has_known_targets() -> None:
    sequence = generate_synthetic_sequence(windows=16, seed=3)

    validate_event_batch(sequence.events)
    assert sequence.ttc_s.shape[0] == 16
    assert sequence.ttc_s[0] > sequence.ttc_s[-1]
    assert sequence.events.num_events > 0


def test_synthetic_hdf5_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.h5"
    sequence = generate_synthetic_sequence(windows=8, seed=5)

    write_synthetic_hdf5(path, sequence)
    loaded = read_synthetic_hdf5(path)

    assert loaded.events.num_events == sequence.events.num_events
    assert loaded.events.width == sequence.events.width
    assert loaded.ttc_s.tolist() == sequence.ttc_s.tolist()

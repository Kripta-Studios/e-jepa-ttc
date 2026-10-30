from pathlib import Path

import h5py
import numpy as np

from operational.evttc_transfer_audit.identity import match_recording, match_sources


def hdf5(path: Path, times: list[int], pixels: list[int]) -> Path:
    with h5py.File(path, "w") as handle:
        handle["blackflys/right/ts"] = np.array(times, dtype=np.int64)
        handle["blackflys/right/data"] = np.array(pixels, dtype=np.uint8).reshape(-1, 1, 1, 1).repeat(3, axis=3)
    return path


def test_offset_and_all_pixels_are_required(tmp_path: Path) -> None:
    crop = hdf5(tmp_path / "crop.h5", [0, 10], [3, 4])
    source = hdf5(tmp_path / "source.h5", [100, 110, 120, 130], [1, 2, 3, 4])
    result = match_recording(crop, source)
    assert result["status"] == "unique"
    assert result["matches"][0]["original_start_frame"] == 2
    assert result["matches"][0]["timestamp_offset_us"] == 120
    assert len(result["matches"][0]["all_rgb_frames"]) == 2
    with h5py.File(source, "r+") as handle:
        handle["blackflys/right/data"][3, 0, 0, 0] = 5
    assert match_recording(crop, source)["status"] == "no_match"


def test_periodic_timestamps_do_not_prove_identity(tmp_path: Path) -> None:
    crop = hdf5(tmp_path / "crop.h5", [0, 10], [3, 4])
    source = hdf5(tmp_path / "source.h5", [100, 110, 120], [1, 2, 3])
    assert match_recording(crop, source)["status"] == "no_match"


def test_duplicate_sources_and_repeated_segments_are_ambiguous(tmp_path: Path) -> None:
    crop = hdf5(tmp_path / "crop.h5", [0, 10], [3, 4])
    a = hdf5(tmp_path / "a.h5", [100, 110], [3, 4])
    b = hdf5(tmp_path / "b.h5", [200, 210], [3, 4])
    assert match_sources(crop, {"a": a, "b": b})["status"] == "ambiguous"
    repeated = hdf5(tmp_path / "repeat.h5", [100, 110, 120, 130], [3, 4, 3, 4])
    assert match_recording(crop, repeated)["status"] == "ambiguous"

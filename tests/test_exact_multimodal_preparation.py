import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_union, map_roi
from e_jepa_ttc.simplex_t.context_raw_union import _read_crop
from operational.multimodal_research.exact_windows import compact
from operational.multimodal_research.selected_map import map_selected


class BoundedSyntheticReader:
    def __init__(self):
        self.columns = dict(
            x=np.array([-1, 0, 1, 63, 64, 1279, 1280, 10, 32, 63, 64], np.int32),
            y=np.array([0, -1, 1, 63, 64, 719, 720, 10, 32, 63, 64], np.int32),
            t=np.array(
                [0, 99999, 100000, 199999, 200000, 299999, 300000, 399999, 400000, 500000, 600000],
                np.int64,
            ),
            p=np.array([0, -1, 1, 0, 1, -1, 1, -1, 1, 0, 1], np.int8),
        )

    def iter_window_chunks(self, start, end, *, chunk_events):
        keep = (self.columns["t"] >= start) & (self.columns["t"] < end)
        ids = np.flatnonzero(keep)
        for start_index in range(0, len(ids), 3):
            yield {k: v[ids[start_index : start_index + 3]] for k, v in self.columns.items()}


@pytest.mark.parametrize(
    "roi",
    [
        (-10.0, -10.0, 1300.0, 750.0),
        (0.0, 0.0, 64.0, 64.0),
        (0.5, 0.5, 63.1, 63.1),
        (80.0, 80.0, 90.0, 90.0),
    ],
)
def test_selected_map_matches_native_after_boundary_crop(roi):
    raw = _read_crop(BoundedSyntheticReader(), 0, 600000, roi, 0.0, 1024**2)
    assert raw is not None
    native, selected = map_roi(raw, roi, 128, 0.0), map_selected(raw, roi, 128, 0.0)
    for key in native:
        np.testing.assert_array_equal(native[key], selected[key], strict=True)
    with pytest.raises(ValueError):
        map_selected(raw, roi, 128, 0.5)


def test_unique_windows_preserve_half_open_timestamps_and_empty_bins():
    reader = BoundedSyntheticReader()
    windows = np.array([[300000, 400000], [400000, 500000], [500000, 600000]], np.int64)
    row = dict(windows_us=windows, square_xyxy=(0.0, 0.0, 64.0, 64.0), sequence_id="synthetic")
    native = encode_union(
        reader,
        windows,
        np.arange(15, -1, -1) * 100000,
        np.arange(16) >= 12,
        row["square_xyxy"],
        sequence_id="synthetic",
        roi_size=128,
        event_pixel_diff=0.0,
    )[-4:]
    exact, meta = compact(reader, row)
    assert torch.equal(native, exact)
    assert meta == dict(fallback=False, unique_windows=6)

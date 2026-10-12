"""Exact reuse of identical sensor windows inside one frozen four-observation query."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, encode_union, map_roi
from e_jepa_ttc.simplex_t.context_raw_union import _read_crop
from operational.evttc_transfer.inputs import EvTTCEventReader
from operational.multimodal_validation.common import (
    MANIFEST,
    V12,
    admission,
    read,
    write,
)
from operational.multimodal_validation.common import (
    OUT as PREVIOUS,
)
from operational.sota_eval.fcwd_inputs import FCWDEventReader

from .common import OUT


def compact(reader: object, row: dict) -> tuple[torch.Tensor, dict]:
    windows = np.asarray(row["windows_us"], np.int64)
    lags = np.arange(3, -1, -1, dtype=np.int64) * 100000
    intervals = windows[None] - lags[:, None, None]
    roi = tuple(row["square_xyxy"])
    # Native adapters implement the same bounded read_window interface.
    raw = _read_crop(reader, int(intervals.min()), int(intervals.max()), roi, 0.0, 1024**3)  # pyright: ignore[reportArgumentType]
    if raw is None:
        result = encode_union(
            reader,  # pyright: ignore[reportArgumentType] -- admitted native reader adapter
            windows,
            np.arange(15, -1, -1) * 100000,
            np.arange(16) >= 12,
            roi,
            sequence_id=row["sequence_id"],
            roi_size=128,
            event_pixel_diff=0.0,
            retained_bytes_max=1024**3,
        )
        return result[-4:], dict(fallback=True, unique_windows=None)
    mapped = map_roi(raw, roi, 128, 0.0)
    memo = {}
    outputs = []
    for observation in intervals:
        frames = []
        for start, end in observation:
            key = (int(start), int(end))
            if key not in memo:
                memo[key] = encode_mapped(mapped, *key, 128, row["sequence_id"])
            frames.append(memo[key])
        outputs.append(torch.stack(frames))
    return torch.stack(outputs), dict(fallback=False, unique_windows=len(memo))


def main() -> None:
    torch.set_num_threads(1)
    outputs = []
    for cohort, manifest, reader_type in (
        ("DEV32", V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json", EvTTCEventReader),
        ("FCWD", MANIFEST, FCWDEventReader),
    ):
        selected = read(PREVIOUS / f"cpu_profile/{cohort}_COST_SELECTION.json")["query_ids"]
        rows = {r["query_id"]: r for r in read(manifest)["rows"]}
        for index, qid in enumerate(selected):
            admission()
            row = rows[qid]
            values, timings, meta = {}, {}, {}
            for repeat in range(2):
                order = (
                    ("native", "compact") if (index + repeat) % 2 == 0 else ("compact", "native")
                )
                for name in order:
                    start = time.perf_counter()
                    with reader_type(Path(row["raw_path"])) as reader:
                        if name == "native":
                            values[name] = encode_union(
                                reader,  # pyright: ignore[reportArgumentType] -- native adapter
                                np.asarray(row["windows_us"], np.int64),
                                np.arange(15, -1, -1) * 100000,
                                np.arange(16) >= 12,
                                tuple(row["square_xyxy"]),
                                sequence_id=row["sequence_id"],
                                roi_size=128,
                                event_pixel_diff=0.0,
                                retained_bytes_max=1024**3,
                            )[-4:]
                        else:
                            values[name], meta = compact(reader, row)
                    timings[name] = (time.perf_counter() - start) * 1000
                equal = torch.equal(values["native"], values["compact"])
                if not equal:
                    raise AssertionError("Exact window reuse changed sensor input: " + qid)
                outputs.append(
                    dict(
                        cohort=cohort,
                        query_id=qid,
                        repeat=repeat,
                        native_ms=timings["native"],
                        compact_ms=timings["compact"],
                        bitwise_equal=equal,
                        **meta,
                    )
                )
            pd.DataFrame(outputs).to_csv(OUT / "EXACT_WINDOWS.csv", index=False)
            print(cohort, qid, timings, flush=True)
    write(
        OUT / "EXACT_WINDOWS.json",
        dict(
            queries=64,
            paired_repeats=2,
            all_input_tensors_bitwise_equal=True,
            algorithm="canonical map/voxel unchanged; one query ROI; memo exact [start,end)",
            scope="HDF5 read, ROI map, voxel encoding only; CPU 1 thread; no neural inference",
            optimizer_updates=0,
            baseline_production_modified=False,
        ),
    )


if __name__ == "__main__":
    main()

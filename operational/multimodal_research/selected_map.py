"""Audit removal of a duplicate spatial mask after canonical bounded ROI cropping."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.efficient_context.mapped_union import map_roi
from e_jepa_ttc.simplex_t.context_raw_union import _read_crop
from operational.evttc_transfer.inputs import EvTTCEventReader
from operational.multimodal_validation.common import MANIFEST, V12, admission, read, sha, write
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.sota_eval.fcwd_inputs import FCWDEventReader

from .common import OUT


def map_selected(raw: dict, roi: tuple, size: int, offset: float) -> dict:
    """Only accepts columns already filtered by canonical _read_crop with this ROI."""
    if offset != 0.0 or size != 128:
        raise ValueError("Only the audited zero-offset 128-pixel route is admitted")
    x0, y0, x1, y1 = roi
    if not np.isfinite(roi).all() or x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid ROI")
    x, y = np.asarray(raw["x"], np.float64), np.asarray(raw["y"], np.float64)
    return dict(
        x=np.clip(np.floor((x - x0) * size / (x1 - x0)).astype(np.int32), 0, size - 1),
        y=np.clip(np.floor((y - y0) * size / (y1 - y0)).astype(np.int32), 0, size - 1),
        t=raw["t"],
        p=np.where(raw["p"] > 0, 1, -1).astype(np.int8),
    )


def main() -> None:
    torch.set_num_threads(1)
    records, bindings = [], {}
    for cohort, manifest, reader_type in (
        ("DEV32", V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json", EvTTCEventReader),
        ("FCWD", MANIFEST, FCWDEventReader),
    ):
        selection = PREVIOUS / f"cpu_profile/{cohort}_COST_SELECTION.json"
        selected = read(selection)["query_ids"]
        rows = {r["query_id"]: r for r in read(manifest)["rows"]}
        bindings[cohort] = dict(manifest_sha256=sha(manifest), selection_sha256=sha(selection))
        for index, qid in enumerate(selected):
            admission()
            row = rows[qid]
            stat = Path(row["raw_path"]).stat()
            assert row["raw_stat"] == dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            windows = np.asarray(row["windows_us"], np.int64)
            intervals = windows[None] - (np.arange(3, -1, -1) * 100000)[:, None, None]
            roi = tuple(row["square_xyxy"])
            with reader_type(Path(row["raw_path"])) as reader:
                raw = _read_crop(
                    reader,  # pyright: ignore[reportArgumentType] -- native bounded adapter
                    int(intervals.min()),
                    int(intervals.max()),
                    roi,
                    0.0,
                    1024**3,
                )  # pyright: ignore[reportArgumentType]
            if raw is None:
                raise RuntimeError("Capacity fallback must retain the native map")
            values, costs = {}, {}
            for repeat in range(2):
                order = (
                    ("native", "selected") if (index + repeat) % 2 == 0 else ("selected", "native")
                )
                for name in order:
                    start = time.perf_counter()
                    values[name] = (map_roi if name == "native" else map_selected)(
                        raw, roi, 128, 0.0
                    )
                    costs[name] = (time.perf_counter() - start) * 1000
                assert all(
                    np.array_equal(values["native"][k], values["selected"][k])
                    and values["native"][k].dtype == values["selected"][k].dtype
                    for k in raw
                )
                records.append(
                    dict(
                        cohort=cohort,
                        query_id=qid,
                        repeat=repeat,
                        native_ms=costs["native"],
                        selected_ms=costs["selected"],
                        raw_events=len(raw["t"]),
                        exact=True,
                    )
                )
            pd.DataFrame(records).to_csv(OUT / "SELECTED_MAP.csv", index=False)
            print(cohort, qid, costs, flush=True)
    write(
        OUT / "SELECTED_MAP.json",
        dict(
            sources={"selected_map.py": sha(__file__)},
            bindings=bindings,
            all_arrays_equal=True,
            queries=len(records) // 2,
            comparisons=len(records),
            raw_stat_verified=True,
            optimizer_updates=0,
            precondition="canonical _read_crop; same ROI; zero offset; no alternate raw callers",
            scope="CPU map stage only, identical already-read raw arrays; concurrent research load",
            production_modified=False,
        ),
    )


if __name__ == "__main__":
    main()

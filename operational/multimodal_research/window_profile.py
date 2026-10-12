"""Attribute exact sensor preparation costs on the sealed hash-selected queries."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from operational.evttc_transfer.inputs import EvTTCEventReader
from operational.multimodal_validation.common import MANIFEST, V12, admission, read, sha, write
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.sota_eval.fcwd_inputs import FCWDEventReader

from . import exact_windows as engine
from .common import OUT

P = ParamSpec("P")
T = TypeVar("T")


def timed(function: Callable[P, T], key: str, costs: dict[str, float]) -> Callable[P, T]:
    def call(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        value = function(*args, **kwargs)
        costs[key] += (time.perf_counter() - start) * 1000
        return value

    return call


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
        for qid in selected:
            admission()
            row = rows[qid]
            stat = Path(row["raw_path"]).stat()
            assert row["raw_stat"] == dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            costs = dict(read_crop_ms=0.0, map_ms=0.0, voxels_ms=0.0)

            start = time.perf_counter()
            with reader_type(Path(row["raw_path"])) as reader:
                opened = time.perf_counter()
                with (
                    patch.object(
                        engine, "_read_crop", timed(engine._read_crop, "read_crop_ms", costs)
                    ),
                    patch.object(engine, "map_roi", timed(engine.map_roi, "map_ms", costs)),
                    patch.object(
                        engine, "encode_mapped", timed(engine.encode_mapped, "voxels_ms", costs)
                    ),
                ):
                    value, meta = engine.compact(reader, row)
                done = time.perf_counter()
            assert np.isfinite(value.numpy()).all()
            records.append(
                dict(
                    cohort=cohort,
                    query_id=qid,
                    open_ms=(opened - start) * 1000,
                    total_ms=(time.perf_counter() - start) * 1000,
                    other_compute_ms=(done - opened) * 1000 - sum(costs.values()),
                    tensor_bytes=value.numel() * value.element_size(),
                    **costs,
                    **meta,
                )
            )
            pd.DataFrame(records).to_csv(OUT / "WINDOW_STAGE_COST.csv", index=False)
            print(cohort, qid, costs, flush=True)
    write(
        OUT / "WINDOW_STAGE_COST.json",
        dict(
            bindings=bindings,
            sources={
                str(Path(__file__).name): sha(__file__),
                "exact_windows.py": sha(engine.__file__),
            },
            raw_stat_verified=True,
            queries=len(records),
            optimizer_updates=0,
            scope="one CPU thread; concurrent research load; preparation only",
        ),
    )


if __name__ == "__main__":
    main()

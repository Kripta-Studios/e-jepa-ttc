"""Paired complete event-route cost for exact window/map/encoder reuse."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from operational.fcwd_rgb_registration import transfer as fcwd_engine
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import (
    MANIFEST,
    ROOT,
    RUN,
    V12,
    admission,
    read,
    sha,
    write,
)
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.rgb_port import transfer as engine
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_revision import transfer_inputs as dev_inputs

from . import exact_windows
from .common import OUT
from .encoder_reuse_profile import reuse
from .selected_map import map_selected


def compact_union(original: Callable) -> Callable:
    def encode(
        reader: object,
        windows: np.ndarray,
        lags: np.ndarray,
        valid: np.ndarray,
        roi: tuple,
        **kwargs: object,
    ) -> torch.Tensor:
        supported = (
            np.array_equal(lags, np.arange(15, -1, -1) * 100000)
            and np.array_equal(valid, np.arange(16) >= 12)
            and kwargs.get("roi_size") == 128
            and kwargs.get("event_pixel_diff") == 0.0
            and kwargs.get("retained_bytes_max") == 1024**3
        )
        if not supported:
            return original(reader, windows, lags, valid, roi, **kwargs)
        row = dict(windows_us=windows, square_xyxy=roi, sequence_id=kwargs["sequence_id"])
        with patch.object(exact_windows, "map_roi", map_selected):
            tail, _ = exact_windows.compact(reader, row)
        return torch.cat([torch.zeros((12, *tail.shape[1:]), dtype=tail.dtype), tail])

    return encode


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    selection = []
    for cohort, manifest in (
        ("DEV32", V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"),
        ("FCWD", MANIFEST),
    ):
        selected = read(PREVIOUS / f"cpu_profile/{cohort}_COST_SELECTION.json")["query_ids"]
        rows = {r["query_id"]: r for r in read(manifest)["rows"]}
        selection.extend((cohort, rows[qid]) for qid in selected)
    output = []
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cpu"
        )
        mean, scale = engine._normalizer(
            RUN / "features/H/E_NORMALIZER.json",
            "event",
            sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
        )
        for index, (cohort, row) in enumerate(selection):
            admission()
            delta = torch.full((len(engine.EVENT_LAGS_US), 2), 0.1)
            for repeat in range(2):
                values, timings, records = {}, {}, []
                order = ("native", "reuse") if (index + repeat) % 2 == 0 else ("reuse", "native")
                for variant in order:
                    with torch.inference_mode(), ExitStack() as probes:
                        if variant == "reuse":
                            adapter = fcwd_engine if cohort == "FCWD" else dev_inputs
                            probes.enter_context(
                                patch.object(
                                    adapter, "encode_union", compact_union(adapter.encode_union)
                                )
                            )
                            for name in ("E_A5_MATCHED", "E_C2F_MATCHED"):
                                encoder = models[name].get_submodule("encoder")
                                probes.enter_context(
                                    patch.object(
                                        encoder, "forward", reuse(encoder.forward, records, name)
                                    )
                                )
                        start = time.perf_counter()
                        raw = (
                            fcwd_engine.prepare_event if cohort == "FCWD" else engine.prepare_event
                        )(row)
                        sensor = torch.from_numpy(raw["own_events"])
                        raw_features, experts = engine._expert_block(models, "E", sensor, delta)
                        query = int(row["anchor_us"])
                        anchors = query - engine.EVENT_LAGS_US
                        clock = torch.from_numpy(
                            engine._clock(anchors, np.full(len(anchors), query), query)
                        )[None]
                        valid = torch.ones((1, len(anchors)), dtype=torch.bool)
                        head_output = models["E_CTX_MATCHED"](
                            ((raw_features - mean) / scale)[None], clock, valid, experts[-1:]
                        )
                        values[variant] = (
                            sensor,
                            raw_features,
                            experts,
                            head_output["point_phase"],
                            head_output["q10"],
                            head_output["q90"],
                        )
                        timings[variant] = (time.perf_counter() - start) * 1000
                a, b = values["native"], values["reuse"]
                exact = [torch.equal(x, y) for x, y in zip(a, b, strict=True)]
                assert all(exact), (cohort, row["query_id"], exact)
                output.append(
                    dict(
                        cohort=cohort,
                        query_id=row["query_id"],
                        repeat=repeat,
                        native_ms=timings["native"],
                        reuse_ms=timings["reuse"],
                        inputs_exact=exact[0],
                        features_exact=exact[1],
                        experts_exact=exact[2],
                        point_exact=exact[3],
                        quantiles_exact=exact[4] and exact[5],
                        frames=records[0]["frames"],
                        unique=records[0]["unique"],
                    )
                )
                pd.DataFrame(output).to_csv(OUT / "EXACT_EVENT_ROUTE.csv", index=False)
            print(cohort, row["query_id"], timings, flush=True)
    write(
        OUT / "EXACT_EVENT_ROUTE.json",
        dict(
            rows=output,
            source_sha256=sha(__file__),
            optimizer_updates=0,
            targets_read=False,
            scope=(
                "64 hash-selected Dev32/FCWD queries, CPU 1 thread, 2 AB/BA repeats "
                "under concurrent load; raw HDF5 to E_CTX point/quantiles with resident weights; "
                "excludes model loading and query/ROI acquisition; no RGB or registration"
            ),
            promoted=False,
            latency_measured=True,
        ),
    )


if __name__ == "__main__":
    main()

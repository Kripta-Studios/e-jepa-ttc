"""Measure exact encoder reuse with canonical inputs and full expert-output parity."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import torch

from operational.fcwd_rgb_registration.transfer import prepare_event
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

from .common import OUT


def reuse(original: Callable, records: list, name: str) -> Callable:
    def forward(values: torch.Tensor, *, return_dense_features: bool = False) -> tuple:
        first, indices, keys = [], [], {}
        for i, value in enumerate(values):
            key = hashlib.sha256(value.numpy().tobytes()).hexdigest()
            if key not in keys:
                keys[key] = len(first)
                first.append(i)
            else:
                assert torch.equal(value, values[first[keys[key]]])
            indices.append(keys[key])
        dense = original(values[first], return_dense_features=return_dense_features)
        rebuilt = tuple(v[indices] if v is not None else None for v in dense)
        records.append(dict(model=name, frames=len(values), unique=len(first)))
        return rebuilt

    return forward


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
        for index, (cohort, row) in enumerate(selection):
            admission()
            raw = (prepare_event if cohort == "FCWD" else engine.prepare_event)(row)
            sensor = torch.from_numpy(raw["own_events"])
            delta = torch.full((len(engine.EVENT_LAGS_US), 2), 0.1)
            for repeat in range(2):
                values, timings, records = {}, {}, []
                order = ("native", "reuse") if (index + repeat) % 2 == 0 else ("reuse", "native")
                for variant in order:
                    with torch.inference_mode(), ExitStack() as probes:
                        if variant == "reuse":
                            for name in ("E_A5_MATCHED", "E_C2F_MATCHED"):
                                encoder = models[name].get_submodule("encoder")
                                probes.enter_context(
                                    patch.object(
                                        encoder, "forward", reuse(encoder.forward, records, name)
                                    )
                                )
                        start = time.perf_counter()
                        values[variant] = engine._expert_block(models, "E", sensor, delta)
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
                        features_exact=exact[0],
                        experts_exact=exact[1],
                        frames=records[0]["frames"],
                        unique=records[0]["unique"],
                    )
                )
                pd.DataFrame(output).to_csv(OUT / "ENCODER_REUSE_PROFILE.csv", index=False)
            print(cohort, row["query_id"], timings, flush=True)
    write(
        OUT / "ENCODER_REUSE_PROFILE.json",
        dict(
            rows=output,
            source_sha256=sha(__file__),
            optimizer_updates=0,
            targets_read=False,
            scope=(
                "64 hash-selected Dev32/FCWD queries, CPU 1 thread, 2 AB/BA repeats "
                "under concurrent load; prepared inputs to full event expert block; not end-to-end"
            ),
            promoted=False,
            latency_measured=True,
        ),
    )


if __name__ == "__main__":
    main()

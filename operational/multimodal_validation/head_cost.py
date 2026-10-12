"""Measure one canonical V12 H8 head and the actual median of three separately."""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer.models import FrozenModels
from operational.train40_system.heads import head_source
from operational.train40_system.history_features import load_source

from .common import OUT, V12, csvwrite, sha, write


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    campaign = V12 / "artifacts/train40_system_20261005"
    owner = SimpleNamespace(campaign=campaign, device=torch.device("cpu"), bindings={})
    heads = FrozenModels._load_h8_heads(cast(FrozenModels, owner))
    source = head_source(campaign, "H8")
    queries = load_source(campaign, source)
    selected = sorted(
        range(queries.population), key=lambda i: hashlib.sha256(str(i).encode()).hexdigest()
    )[:64]
    rows = []
    for iteration in range(2):
        for index in selected:
            x, timing, valid, experts, _, _ = queries.gather(torch.tensor([index]))
            with torch.inference_mode():
                start = time.perf_counter()
                single = phase_to_ttc(heads[7](x, timing, valid, experts)["point_phase"])
                single_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                predictions = torch.stack(
                    [
                        phase_to_ttc(heads[s](x, timing, valid, experts)["point_phase"])
                        for s in (7, 13, 23)
                    ]
                )
                median = predictions.median(0).values
                median_ms = (time.perf_counter() - start) * 1000
                assert torch.equal(single, predictions[0])
            rows.append(
                dict(
                    query_ordinal=index,
                    iteration=iteration,
                    single_head_ms=single_ms,
                    median3_actual_ms=median_ms,
                    single_ttc=float(single[0]),
                    median3_ttc=float(median[0]),
                )
            )
    csvwrite(OUT / "V12_HEAD_COST.csv", rows)
    summary = {}
    for key in ("single_head_ms", "median3_actual_ms"):
        values = [r[key] for r in rows if r["iteration"] == 1]
        summary[key] = dict(
            zip(("p50", "p95", "p99"), np.quantile(values, [0.5, 0.95, 0.99]).tolist(), strict=True)
        )
    write(
        OUT / "V12_HEAD_COST.json",
        dict(
            scope=(
                "prepared_native_TRAIN40_H8_inputs_only; "
                "no sensor reading, producers or state commit"
            ),
            queries=64,
            iterations=2,
            device="cpu",
            threads=1,
            summary=summary,
            bindings=owner.bindings,
            feature_sha256=sha(campaign / "H8_FEATURES.npz"),
            optimizer_updates=0,
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()

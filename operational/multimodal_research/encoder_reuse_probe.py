"""Check numerical admissibility of removing duplicate frozen encoder frames."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import ExitStack
from unittest.mock import patch

import torch

from operational.fcwd_rgb_registration.transfer import prepare_event
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import (
    MANIFEST,
    ROOT,
    RUN,
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
        reference = original(values, return_dense_features=return_dense_features)
        checks = []
        for a, b in zip(reference, rebuilt, strict=True):
            if a is None:
                assert b is None
                continue
            assert b is not None
            checks.append(dict(exact=torch.equal(a, b), max_abs_delta=float((a - b).abs().max())))
        records.append(dict(model=name, frames=len(values), unique=len(first), checks=checks))
        return rebuilt

    return forward


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    selected = read(PREVIOUS / "cpu_profile/FCWD_COST_SELECTION.json")["query_ids"]
    rows = {r["query_id"]: r for r in read(MANIFEST)["rows"]}
    ids = [
        next(qid for qid in selected if qid.startswith(seq + ":"))
        for seq in ("FCWD1", "FCWD2", "FCWD3")
    ]
    output = []
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cpu"
        )
        for qid in ids:
            admission()
            sensor = torch.from_numpy(prepare_event(rows[qid])["own_events"])
            delta = torch.full((len(engine.EVENT_LAGS_US), 2), 0.1)
            records = []
            with torch.inference_mode():
                before = engine._expert_block(models, "E", sensor, delta)
                with ExitStack() as probes:
                    for name in ("E_A5_MATCHED", "E_C2F_MATCHED"):
                        encoder = models[name].get_submodule("encoder")
                        probes.enter_context(
                            patch.object(encoder, "forward", reuse(encoder.forward, records, name))
                        )
                    after = engine._expert_block(models, "E", sensor, delta)
            output.append(
                dict(
                    query_id=qid,
                    encoder=records,
                    features_exact=torch.equal(before[0], after[0]),
                    experts_exact=torch.equal(before[1], after[1]),
                    feature_max_delta=float((before[0] - after[0]).abs().max()),
                    expert_max_delta=float((before[1] - after[1]).abs().max()),
                )
            )
            print(output[-1], flush=True)
    write(
        OUT / "ENCODER_REUSE_PROBE.json",
        dict(
            rows=output,
            source_sha256=sha(__file__),
            optimizer_updates=0,
            targets_read=False,
            scope="3 hash-selected FCWD queries, CPU 1 thread; batch-size change inspected",
            promoted=False,
            latency_measured=False,
        ),
    )


if __name__ == "__main__":
    main()

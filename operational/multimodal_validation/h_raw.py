"""Trace the same 32 H controls from real HDF5/TAR through frozen endpoints."""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any, cast
from unittest.mock import patch

# ruff: noqa: ANN401 -- retained native model interfaces.
import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.rgb_port.cost_routes import RouteRuntime
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .ancestry import historical_source_binding
from .common import OUT, ROOT, RUN, V12, admission, read, write
from .traces import Recorder, save


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rec = Recorder()
    rec.details = read(OUT / "STAGE_TRACE_DETAILS.json")
    rec.rows = [{k: v for k, v in r.items() if k not in ("inputs", "outputs")} for r in rec.details]
    tokens = read(OUT / "H_CONTROLS_SEALED.json")["tokens"]
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        runtime = RouteRuntime(
            run=RUN,
            config=ROOT / "configs/rgb_port/producers.json",
            p_manifest=RUN / "P_MANIFEST.json",
            manifest=RUN / "H_MANIFEST.json",
            event_source_root=V12 / "artifacts/train40_system_20261005",
            event_history=RUN / "features/H/E_RAW/history.npz",
            normalizers={
                m: FrozenNormalizer.load(RUN / f"features/H/{p}_NORMALIZER.json")
                for m, p in [("event", "E"), ("rgb", "R")]
            },
            device="cpu",
        )
        stack.callback(runtime.close)

        class Wrapped:
            def __init__(self, name: str, model: torch.nn.Module) -> None:
                self.name, self.model = name, model

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                rec.name = self.name
                return rec.call(self.model, *args, **kwargs)

            def __getattr__(self, key: str) -> Any:
                return getattr(self.model, key)

        for group in (runtime.producers, runtime.pairs, runtime.heads):
            for name in list(group):
                cast(Any, group)[name] = Wrapped(name, group[name])
        lookup = {str(r["sample_token"]): i for i, r in enumerate(runtime.rows)}
        for token in tokens:
            if any(
                r.get("query_id") == token
                and r.get("scope") == "H_raw_hash_control"
                and r.get("stage") == "F_TRUE"
                for r in rec.rows
            ):
                continue
            admission()
            index = lookup[token]
            rec.query = token
            rec.scope = "H_raw_hash_control"
            value = runtime.raw_call(index, "full_fusion", "F_TRUE")
            write(
                OUT / "h_raw_queries" / f"{index:06d}.json",
                dict(
                    sample_token=token,
                    prediction=float(value.item()),
                    metadata=json.loads(
                        json.dumps(
                            runtime.rows[index],
                            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
                        )
                    ),
                ),
            )
            save(rec, [])
            print("H raw", tokens.index(token) + 1, "/", len(tokens), flush=True)
    write(
        OUT / "H_RAW_COMPLETE.json",
        dict(queries=len(tokens), all_instrumented_calls_bitwise_equal=True, optimizer_updates=0),
    )


if __name__ == "__main__":
    main()

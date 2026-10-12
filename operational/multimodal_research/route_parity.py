"""Locate historical FCWD route deltas without changing checkpoints or predictions."""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.fcwd_rgb_registration.transfer import prepare_event
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import MANIFEST, REG, ROOT, RUN, read, sha, write
from operational.rgb_port import transfer as engine
from operational.rgb_port_external_fix.runtime import observation_contract

from .common import OUT, guard, setup


def main() -> None:
    setup()
    ids = [
        "FCWD1:0007",
        "FCWD3:0194",
        "FCWD1:0069",
        "FCWD2:0073",
        "FCWD1:0205",
        "FCWD2:0211",
        "FCWD1:0204",
        "FCWD2:0209",
        "FCWD1:0206",
        "FCWD3:0196",
    ]
    rows = {r["query_id"]: r for r in read(MANIFEST)["rows"]}
    historical_path = REG.parent / "predictions/PREDICTIONS.csv"
    historical = pd.read_csv(historical_path)
    historical = historical.set_index("query_id")
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
        result = []
        for qid in ids:
            guard()
            row = rows[qid]
            data = prepare_event(row)
            cpu = torch.from_numpy(data["own_events"])
            values = {}
            for device, matmul_tf32, cudnn_tf32 in (
                ("cpu", False, False),
                ("cuda", False, False),
                ("cuda", False, True),
                ("cuda", True, False),
                ("cuda", True, True),
            ):
                for model in models.values():
                    model.to(device)
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
                sensor = cpu.to(device)
                delta = torch.full((len(engine.EVENT_LAGS_US), 2), 0.1, device=device)
                with torch.inference_mode():
                    raw, experts = engine._expert_block(models, "E", sensor, delta)
                    features = ((raw - mean.to(device)) / scale.to(device))[None]
                    query = int(row["anchor_us"])
                    anchors = query - np.asarray(engine.EVENT_LAGS_US)
                    clock = torch.from_numpy(
                        engine._clock(anchors, np.full(len(anchors), query), query)
                    ).to(device)[None]
                    valid = torch.ones((1, len(anchors)), dtype=torch.bool, device=device)
                    prediction = models["E_CTX_MATCHED"](features, clock, valid, experts[-1:])
                    ttc = float(phase_to_ttc(prediction["point_phase"])[0])
                key = f"{device}_matmul{int(matmul_tf32)}_cudnn{int(cudnn_tf32)}"
                values[key] = dict(
                    ttc=ttc,
                    raw_features=raw.cpu().numpy().tolist(),
                    experts=experts.cpu().numpy().tolist(),
                    raw_location=float(prediction["raw_location"][0]),
                    historical_delta=abs(ttc - float(historical.loc[qid, "E_CTX_MATCHED"])),
                )
            result.append(
                dict(
                    query_id=qid,
                    modes=values,
                    historical=float(historical.loc[qid, "E_CTX_MATCHED"]),
                )
            )
            write(
                OUT / "ROUTE_PARITY.json",
                dict(
                    rows=result,
                    optimizer_updates=0,
                    source_sha256=sha(__file__),
                    historical_predictions_sha256=sha(historical_path),
                    external_targets_read=False,
                    selection="4 existing trace cases plus worst CPU/historical deltas; no GT",
                    interpretation=(
                        "same raw inputs; device and TF32 controlled; no prediction replacement"
                    ),
                ),
            )
            print(qid, {k: v["historical_delta"] for k, v in values.items()}, flush=True)


if __name__ == "__main__":
    main()

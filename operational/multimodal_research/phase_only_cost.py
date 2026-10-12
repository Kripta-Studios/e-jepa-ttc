"""Measure sensor-to-candidate stages and exact cached-state parity on fixed hash controls."""

from __future__ import annotations

import hashlib
import time
from contextlib import ExitStack

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.fcwd_rgb_registration.transfer import prepare_event, rgb_observations
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import (
    MANIFEST,
    ROOT,
    RUN,
    V12,
    admission,
    read,
    registrations,
    sha,
    write,
)
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.rgb_port import transfer as engine
from operational.rgb_port_external_fix.runtime import observation_contract

from .common import OUT, authorize
from .model import frozen_state
from .phase_only_anchor import PhaseOnlyAnchor


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_path = OUT / "fits/R_ANCHOR_PHASE_ONLY_EVENT_S7/checkpoint_last.pt"
    head = PhaseOnlyAnchor("EVENT").eval().requires_grad_(False)
    head.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    selection = []
    for cohort, manifest in (
        ("FCWD", MANIFEST),
        ("DEV32", V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"),
    ):
        rows = {r["query_id"]: r for r in read(manifest)["rows"]}
        ids = read(PREVIOUS / f"cpu_profile/{cohort}_COST_SELECTION.json")["query_ids"]
        selection.extend((cohort, rows[qid]) for qid in ids)
    regs = registrations()
    records = []
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cpu"
        )
        for model in models.values():
            model.eval().requires_grad_(False)
        normalizers = {
            m: engine._normalizer(
                RUN / f"features/H/{m}_NORMALIZER.json",
                modality,
                sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
            )
            for m, modality in (("E", "event"), ("R", "rgb"))
        }
        for index, (cohort, row) in enumerate(selection):
            admission()
            from datetime import UTC, datetime

            auth = authorize()
            assert (datetime.now(UTC) - datetime.fromisoformat(auth["utc"])).total_seconds() < auth[
                "max_elapsed_seconds"
            ]
            key = hashlib.sha256(row["query_id"].encode()).hexdigest()
            with np.load(OUT / f"external_states/{cohort}/{key}.npz", allow_pickle=False) as z:
                expected = {k: z[k] for k in z.files}
            for repeat in range(2):
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    raw = (prepare_event if cohort == "FCWD" else engine.prepare_event)(row)
                    sensor = torch.from_numpy(raw["own_events"])
                    t1 = time.perf_counter()
                    eraw, experts = engine._expert_block(
                        models, "E", sensor, torch.full((4, 2), 0.1)
                    )
                    t2 = time.perf_counter()
                    query = int(row["anchor_us"])
                    mean, scale = normalizers["E"]
                    ef = ((eraw - mean) / scale)[None]
                    anchors = query - np.asarray(engine.EVENT_LAGS_US)
                    et = torch.from_numpy(
                        engine._clock(anchors, np.full(len(anchors), query), query)
                    )[None]
                    ev, ee = torch.ones((1, len(anchors)), dtype=torch.bool), experts[-1:]
                    eb, _, eo = frozen_state(models["E_CTX_MATCHED"], ef, et, ev, ee)
                    t3 = time.perf_counter()
                    observations, reason = (
                        rgb_observations(row, regs[row["query_id"]])
                        if cohort == "FCWD"
                        else engine.raw_rgb_observations(row)
                    )
                    t4 = time.perf_counter()
                    timeline = []
                    mean, scale = normalizers["R"]
                    for pixels, meta in observations:
                        rr, re = engine._expert_block(
                            models,
                            "R",
                            torch.from_numpy(pixels)[None],
                            torch.tensor([meta["delta_t_s"]]),
                        )
                        timeline.append(
                            (
                                meta["first_sensor_ts_us"],
                                meta["anchor_us"],
                                meta["available_us"],
                                ((rr - mean) / scale)[0],
                                re[0],
                            )
                        )
                    t5 = time.perf_counter()
                    if timeline:
                        history = engine._select_rgb_history(timeline, query)
                        rf = torch.stack([v[3] for v in history])[None]
                        rt = torch.from_numpy(
                            engine._clock(
                                np.asarray([v[1] for v in history]),
                                np.asarray([v[2] for v in history]),
                                query,
                            )
                        )[None]
                        rv = torch.ones((1, len(history)), dtype=torch.bool)
                        rb, widths, _ = frozen_state(
                            models["R_CTX"], rf, rt, rv, history[-1][4][None]
                        )
                        features = torch.cat([rb, eb], -1)
                    t6 = time.perf_counter()
                    prediction = (
                        head(features, widths)["ttc"]
                        if timeline
                        else phase_to_ttc(eo["point_phase"])
                    )
                    t7 = time.perf_counter()
                    assert np.array_equal(eb.numpy(), expected["event_block"])
                    if timeline:
                        assert np.array_equal(features.numpy(), expected["features"])
                        assert np.array_equal(widths.numpy(), expected["rgb_width_logits"])
                        reference = head(
                            torch.from_numpy(expected["features"]),
                            torch.from_numpy(expected["rgb_width_logits"]),
                        )["ttc"]
                        assert torch.equal(prediction, reference)
                    record = dict(
                        cohort=cohort,
                        query_id=row["query_id"],
                        repeat=repeat,
                        rgb_available=bool(timeline),
                        rgb_observations=len(timeline),
                        reason=reason,
                        event_read_voxels_ms=1000 * (t1 - t0),
                        event_producers_pair_ms=1000 * (t2 - t1),
                        event_context_state_ms=1000 * (t3 - t2),
                        rgb_read_crop_ms=1000 * (t4 - t3),
                        rgb_producers_pair_ms=1000 * (t5 - t4),
                        rgb_context_state_ms=1000 * (t6 - t5),
                        correction_ms=1000 * (t7 - t6),
                        sensor_to_candidate_ms=1000 * (t7 - t0),
                        ttc=float(prediction[0]),
                        exact_state_parity=True,
                    )
                    records.append(record)
                pd.DataFrame(records).to_csv(OUT / "PHASE_ONLY_ROUTE_COST.csv", index=False)
            print("PHASE_ONLY_COST", index + 1, len(selection), flush=True)
    frame = pd.DataFrame(records)
    summary = {}
    for cohort in ("FCWD", "DEV32"):
        sub = frame[(frame.cohort == cohort) & (frame.repeat == 1)]
        summary[cohort] = {
            name: dict(
                zip(("p50", "p95"), np.quantile(sub[name], [0.5, 0.95]).tolist(), strict=True)
            )
            for name in frame
            if name.endswith("_ms")
        }
    write(
        OUT / "PHASE_ONLY_ROUTE_COST.json",
        dict(
            source_sha256=sha(__file__),
            checkpoint_sha256=sha(model_path),
            summary=summary,
            rows=len(records),
            all_states_and_candidate_points_exact=True,
            optimizer_updates=0,
            scope="CPU FP32 one thread, concurrent system load. Raw sensor read through candidate, "
            "resident weights, public/frozen ROI and already sealed FCWD registration. "
            "Excludes registration search, detector and initialization; no streaming claim.",
            csv_sha256=sha(OUT / "PHASE_ONLY_ROUTE_COST.csv"),
        ),
    )


if __name__ == "__main__":
    main()

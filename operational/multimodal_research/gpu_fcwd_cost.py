"""Frozen FCWD sensor-to-candidate GPU cost including sealed-equivalent registration."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from contextlib import ExitStack
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import psutil
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.fcwd_rgb_registration.prepare import register_query
from operational.fcwd_rgb_registration.transfer import prepare_event, rgb_observations
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import (
    MANIFEST,
    REG,
    ROOT,
    RUN,
    read,
    registrations,
    sha,
    write,
)
from operational.multimodal_validation.common import OUT as PREVIOUS
from operational.rgb_port import transfer as engine
from operational.rgb_port_external_fix.runtime import observation_contract

from . import common
from .guard_scan import wrap
from .model import frozen_state
from .phase_only_anchor import PhaseOnlyAnchor


def exclusive() -> None:
    common.guard()
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    for line in output.splitlines():
        if not line.strip().isdigit() or int(line) == os.getpid():
            continue
        try:
            proc = psutil.Process(int(line))
            assert not proc.name().lower().startswith("python"), (
                "Another Python GPU owner must finish before profiling",
                proc.pid,
            )
        except psutil.NoSuchProcess:
            continue
    assert torch.cuda.mem_get_info()[0] >= 6 * 1024**3
    assert torch.cuda.memory_reserved() <= 1024**3


def mark() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    with patch.object(common, "guard", wrap(common.guard)):
        run()


def run() -> None:
    assert (common.OUT / "test12_phase_only/PACKAGE_VERIFICATION.json").exists(), (
        "Finish and package frozen test inference before occupying the GPU"
    )
    common.setup()
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(
        1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    cv2.setNumThreads(1)
    exclusive()
    selection_path = PREVIOUS / "cpu_profile/FCWD_COST_SELECTION.json"
    ids = read(selection_path)["query_ids"]
    rows = {r["query_id"]: r for r in read(MANIFEST)["rows"]}
    regs = registrations()
    calibration = read(REG / "BINDING.json")["calibration"]
    checkpoint = common.OUT / "fits/R_ANCHOR_PHASE_ONLY_EVENT_S7/checkpoint_last.pt"
    head = PhaseOnlyAnchor("EVENT").eval().requires_grad_(False)
    head.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    scored_path = (
        common.OUT
        / "studies/PHASE_ONLY/S7/external_evaluation_v2/native_gpu/FCWD/SCORED_PREDICTIONS.csv"
    )
    reference = pd.read_csv(scored_path, float_precision="round_trip").set_index("query_id")
    results = []
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cuda"
        )
        for model in models.values():
            model.eval().requires_grad_(False)
        normalizers = {}
        for prefix, modality in (("E", "event"), ("R", "rgb")):
            mean, scale = engine._normalizer(
                RUN / f"features/H/{prefix}_NORMALIZER.json",
                modality,
                sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
            )
            normalizers[prefix] = mean.cuda(), scale.cuda()
        for index, qid in enumerate(ids):
            exclusive()
            row = rows[qid]
            key = hashlib.sha256(qid.encode()).hexdigest()
            with np.load(
                common.OUT / f"external_states_gpu_native/FCWD/{key}.npz", allow_pickle=False
            ) as z:
                expected = {k: z[k] for k in z.files}
            for repeat in range(2):
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    t0 = mark()
                    fresh = register_query(row, calibration)
                    assert all(fresh.get(k) == regs[qid].get(k) for k in fresh)
                    t1 = mark()
                    raw = prepare_event(row)
                    t2 = mark()
                    sensor = torch.from_numpy(raw["own_events"]).cuda()
                    delta = torch.full((4, 2), 0.1, device="cuda")
                    t3 = mark()
                    eraw, experts = engine._expert_block(models, "E", sensor, delta)
                    t4 = mark()
                    query = int(row["anchor_us"])
                    mean, scale = normalizers["E"]
                    anchors = query - np.asarray(engine.EVENT_LAGS_US)
                    et = torch.from_numpy(
                        engine._clock(anchors, np.full(len(anchors), query), query)
                    )[None].cuda()
                    ev = torch.ones((1, len(anchors)), dtype=torch.bool, device="cuda")
                    eb, _, eo = frozen_state(
                        models["E_CTX_MATCHED"], ((eraw - mean) / scale)[None], et, ev, experts[-1:]
                    )
                    t5 = mark()
                    observations, reason = rgb_observations(row, fresh)
                    t6 = mark()
                    timeline = []
                    rgb_transfer_ms = 0.0
                    rgb_producers_ms = 0.0
                    mean, scale = normalizers["R"]
                    for pixels, meta in observations:
                        a = mark()
                        rs = torch.from_numpy(pixels)[None].cuda()
                        rd = torch.tensor([meta["delta_t_s"]], device="cuda")
                        b = mark()
                        rr, re = engine._expert_block(models, "R", rs, rd)
                        normalized = ((rr - mean) / scale)[0]
                        c = mark()
                        rgb_transfer_ms += 1000 * (b - a)
                        rgb_producers_ms += 1000 * (c - b)
                        timeline.append(
                            (
                                meta["first_sensor_ts_us"],
                                meta["anchor_us"],
                                meta["available_us"],
                                normalized,
                                re[0],
                            )
                        )
                    t7 = mark()
                    if timeline:
                        history = engine._select_rgb_history(timeline, query)
                        rf = torch.stack([v[3] for v in history])[None]
                        rt = torch.from_numpy(
                            engine._clock(
                                np.asarray([v[1] for v in history]),
                                np.asarray([v[2] for v in history]),
                                query,
                            )
                        )[None].cuda()
                        rv = torch.ones((1, len(history)), dtype=torch.bool, device="cuda")
                        rb, widths, _ = frozen_state(
                            models["R_CTX"], rf, rt, rv, history[-1][4][None]
                        )
                    t8 = mark()
                    e_cpu = eb.cpu()
                    if timeline:
                        features = torch.cat([rb, eb], -1).cpu()
                        widths_cpu = widths.cpu()
                    else:
                        phase_cpu = eo["point_phase"].cpu()
                    t9 = mark()
                    prediction = (
                        phase_to_ttc(head(features, widths_cpu)["point_phase"])
                        if timeline
                        else phase_to_ttc(phase_cpu)
                    )
                    t10 = time.perf_counter()
                    assert np.array_equal(e_cpu.numpy(), expected["event_block"])
                    if timeline:
                        assert np.array_equal(features.numpy(), expected["features"])
                        assert np.array_equal(widths_cpu.numpy(), expected["rgb_width_logits"])
                    assert float(prediction[0]) == float(reference.loc[qid, "R_ANCHOR_EVENT"])
                    results.append(
                        dict(
                            query_id=qid,
                            repeat=repeat,
                            rgb_available=bool(timeline),
                            reason=reason,
                            registration_ms=1000 * (t1 - t0),
                            event_read_voxels_ms=1000 * (t2 - t1),
                            event_h2d_ms=1000 * (t3 - t2),
                            event_producers_pair_ms=1000 * (t4 - t3),
                            event_context_ms=1000 * (t5 - t4),
                            rgb_read_crop_ms=1000 * (t6 - t5),
                            rgb_h2d_ms=rgb_transfer_ms,
                            rgb_producers_pair_ms=rgb_producers_ms,
                            rgb_dispatch_and_sync_ms=1000 * (t7 - t6)
                            - rgb_transfer_ms
                            - rgb_producers_ms,
                            rgb_context_ms=1000 * (t8 - t7),
                            states_d2h_ms=1000 * (t9 - t8),
                            cpu_correction_ms=1000 * (t10 - t9),
                            sensor_to_candidate_ms=1000 * (t10 - t0),
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                            exact_registration_state_and_point=True,
                            ttc=float(prediction[0]),
                        )
                    )
                pd.DataFrame(results).to_csv(common.OUT / "FCWD_GPU_ROUTE_COST.csv", index=False)
            print("FCWD_GPU_COST", index + 1, len(ids), flush=True)
    frame = pd.DataFrame(results)
    measured = frame.query("repeat == 1")
    summary = {
        k: dict(zip(("p50", "p95"), np.quantile(measured[k], [0.5, 0.95]).tolist(), strict=True))
        for k in frame
        if k.endswith("_ms")
    }
    write(
        common.OUT / "FCWD_GPU_ROUTE_COST.json",
        dict(
            source_sha256=sha(__file__),
            checkpoint_sha256=sha(checkpoint),
            selection_sha256=sha(selection_path),
            reference_predictions_sha256=sha(scored_path),
            rows=len(results),
            summary=summary,
            csv_sha256=sha(common.OUT / "FCWD_GPU_ROUTE_COST.csv"),
            optimizer_updates=0,
            all_registration_states_and_predictions_exact=True,
            scope="FCWD raw sensors, fresh sealed-equivalent registration and frozen GPU parents "
            "and CPU readout. One CPU thread; batch4 event observations, batch1 RGB observations. "
            "Stage synchronization; no other Python GPU owner, normal desktop GPU load allowed. "
            "Resident weights, public query ROI. Excludes detector, startup and persistent "
            "cross-query service/state management. Repeat1 quantiles are warm-access observations, "
            "not isolated cold-I/O or streaming latency.",
        ),
    )


if __name__ == "__main__":
    main()

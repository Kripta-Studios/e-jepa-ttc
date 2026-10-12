"""Same 64 query identities: Garl native networks, exact optimized inputs, CPU."""

from __future__ import annotations

import time

import cv2
import h5py
import numpy as np
import torch

from e_jepa_ttc.data.garl_official_preprocessing import official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature
from operational.evttc_rgb_transfer.inputs import prepare as dev_prepare
from operational.evttc_rgb_transfer.model import release_ttc
from operational.fcwd_rgb_registration.prepare import register_query
from operational.sota_eval.fcwd_inputs import FCWDEventReader
from operational.train40_system.garl_predictions import native_model

from .common import (
    CODE,
    MANIFEST,
    OUT,
    REG,
    V12,
    admission,
    csvwrite,
    read,
    registrations,
    sha,
    write,
)
from .garl import load, prepare, timing


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    full, binding = load("full")
    campaign = V12 / "artifacts/train40_system_20261005"
    event_hash = sha(campaign / "public_garl/paper_event_only_lhr.pth")
    assert event_hash == read(campaign / "DELIVERY_FREEZE.json")["public_checkpoint_sha256"]
    event = native_model(campaign, CODE)
    load_ms = (time.perf_counter() - started) * 1000
    regs = registrations()
    calibration = read(REG / "BINDING.json")["calibration"]
    rows_out = []
    for cohort, manifest in [
        ("DEV32", V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"),
        ("FCWD", MANIFEST),
    ]:
        wanted = read(OUT / f"cpu_profile/{cohort}_COST_SELECTION.json")["query_ids"]
        lookup = {r["query_id"]: r for r in read(manifest)["rows"]}
        for iteration in range(2):
            for qid in wanted:
                admission()
                row = lookup[qid]
                t0 = time.perf_counter()
                if cohort == "FCWD":
                    fresh = register_query(row, calibration)
                    assert all(fresh.get(k) == regs[qid].get(k) for k in fresh)
                    registration_ms = (time.perf_counter() - t0) * 1000
                    with h5py.File(row["raw_path"], "r") as h:
                        clock = np.asarray(h["blackflys/right/ts"], np.int64)
                    temporal = timing(row, clock)
                    available = regs[qid]["accepted"] and temporal["available"]
                    if available:
                        sensor, details = prepare(row, regs[qid], temporal, True)
                    else:
                        parts = []
                        with FCWDEventReader(row["raw_path"]) as reader:
                            for i, w in enumerate(row["windows_us"][-2:]):
                                raw = reader.read_window(*w)
                                parts.append(
                                    native_feature(
                                        raw, official_square_box(row["boxes_xyxy3"][-2:], i)
                                    ).numpy()
                                )
                        sensor = np.concatenate(parts)
                else:
                    registration_ms = 0.0
                    prepared = dev_prepare(row)
                    available = prepared["sensor"] is not None
                    sensor = prepared["sensor"] if available else prepared["garl_events"]
                pre = time.perf_counter()
                assert sensor is not None, "A declared available native input must exist"
                tensor = torch.from_numpy(sensor[None])
                model = full if available else event
                with torch.inference_mode():
                    heights, _ = model.forward_test(tensor)
                prediction = float(release_ttc(heights.numpy(), 0.1)[0])
                model_ms = (time.perf_counter() - pre) * 1000
                total_ms = (time.perf_counter() - t0) * 1000
                # Same input, output must be exact; prepared resident perimeter is separate.
                t1 = time.perf_counter()
                with torch.inference_mode():
                    resident, _ = model.forward_test(tensor)
                resident_ms = (time.perf_counter() - t1) * 1000
                assert torch.equal(heights, resident)
                rows_out.append(
                    dict(
                        query_id=qid,
                        cohort=cohort,
                        iteration=iteration,
                        full_available=available,
                        registration_ms=registration_ms,
                        preprocess_including_registration_ms=(pre - t0) * 1000,
                        native_network_ms=model_ms,
                        total_raw_system_ms=total_ms,
                        resident_network_ms=resident_ms,
                        prediction_ttc=prediction,
                        bitwise_parity=True,
                    )
                )
                csvwrite(OUT / "GARL_CPU_COST.csv", rows_out)
                print("Garl cost", cohort, iteration, qid, flush=True)
    write(
        OUT / "GARL_CPU_COST_BINDING.json",
        dict(
            queries=64,
            iterations=2,
            model_load_ms=load_ms,
            scope="same sensor queries; full with exact event fallback; FP32 CPU 1 thread",
            input_optimization=(
                "existing native_feature vectorized time-volume and ROI sampling, no A5 voxels"
            ),
            GPU_graphs="not_applicable_CPU; deferred_while_V15_owns_GPU",
            binding=binding,
            event_checkpoint_sha256=event_hash,
            all_bitwise_parity=True,
            optimizer_updates=0,
        ),
    )


if __name__ == "__main__":
    main()

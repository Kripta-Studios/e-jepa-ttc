"""Target-blind startup-router replay using native road or Slider sensor contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch

from operational.evttc_competition.geometry import load_calibration
from operational.evttc_competition.prepare import prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import load_sensor_runtime, verified_upstream_paths
from operational.phase_robust.cache import SensorCache
from operational.phase_robust.data import histories
from operational.phase_startup.runtime import (
    describe_prediction,
    load_bundle_router,
    predict_history,
)
from scripts.final_train40_transfer import FinalTransferPredictor, reject_targets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--contract", choices=("native_road", "slider"), required=True)
    p.add_argument("--variant", choices=("evaluated", "all_data_refit"), default="evaluated")
    p.add_argument("--manifest", type=Path, help="native_road sensor-only JSON population")
    p.add_argument("--sensors", type=Path, help="slider sensor root")
    p.add_argument("--sequence", help="Slider sequence ID")
    p.add_argument("--profile", choices=("competition", "event_aided_ttc"), default="competition")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fusion = load_bundle_router(args.bundle, args.variant)
    calibration = (
        read(args.bundle / "evaluated/CONFIRMATION.json")["calibration"]
        if args.variant == "evaluated"
        else None
    )
    output = []
    if args.contract == "native_road":
        if args.manifest is None:
            p.error("native_road requires --manifest")
        rows = read(args.manifest)["rows"]
        reject_targets(rows)
        if any("ttc_csv_path" in row for row in rows):
            raise ValueError("Target file paths are forbidden in inference")
        tracks = histories(rows)
        runtime = FinalTransferPredictor(
            verified_upstream_paths(args.bundle / "upstream"), args.device
        )
        for row in rows[: args.limit or len(rows)]:
            estimates = runtime.predict(row)["predictions"]
            native = np.array(
                [estimates[k] for k in ("PHASE_ONLY_TRAIN40_S7", "R_CTX_TRAIN40", "E_CTX_TRAIN40")],
                np.float32,
            )
            boxes, clocks = tracks[row["sequence_id"]]
            pred, physical = predict_history(fusion, native, boxes, clocks, row["anchor_us"])
            output.append(
                {
                    "query_id": row["query_id"],
                    **describe_prediction(pred, native, physical, calibration),
                }
            )
            print(f"native {len(output)}", flush=True)
    else:
        if args.sensors is None or args.sequence is None or not args.sequence.startswith("Slider-"):
            p.error("slider requires --sensors and --sequence Slider-...")
        slider_runtime = load_sensor_runtime(args.bundle / "upstream", args.device)
        camera = load_calibration(args.sensors / args.sequence, "slider_candidate")
        with SensorSequence(args.sensors, args.sequence, slider_profile=args.profile) as source:
            rows = source.bbox_rows()
            boxes = np.array([r["box_xyxy"] for r in rows])
            clocks = np.array([r["anchor_us"] for r in rows])
            cached = SensorCache(source)
            for row in rows[: args.limit or len(rows)]:
                prepared = prepare_query(cast(SensorSequence, cached), rows, row, camera)
                result = slider_runtime.predict(prepared.events, prepared.rgb, row["anchor_us"])
                native = np.array([result.model_ttc, result.rgb_ttc, result.event_ttc], np.float32)
                pred, physical = predict_history(fusion, native, boxes, clocks, row["anchor_us"])
                output.append(
                    {
                        "frame_index": row["frame_index"],
                        "anchor_us": row["anchor_us"],
                        **describe_prediction(pred, native, physical, calibration),
                    }
                )
                print(f"slider {len(output)}", flush=True)
    atomic_receipt(
        args.output,
        {
            "labels_read": False,
            "contract": args.contract,
            "variant": args.variant,
            "binding_sha256": file_sha(args.bundle / "ROUTER_BINDING.json"),
            "rows": output,
        },
    )


if __name__ == "__main__":
    main()

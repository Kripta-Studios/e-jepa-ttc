"""Target-blind offline sensor replay with frozen fusion and explicit quality flags."""

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
from operational.phase_robust.bundle import load_bundle_fusion, load_sensor_runtime
from operational.phase_robust.cache import SensorCache
from operational.phase_robust.runtime import predict_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--sensors", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--variant", choices=("evaluated", "all_data_refit"), default="evaluated")
    parser.add_argument(
        "--profile", choices=("competition", "event_aided_ttc"), default="competition"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="0 runs every provided bbox query")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.sequence.startswith("Slider-"):
        raise ValueError("Use predict_phase_robust_native for the native road RGB/history contract")
    configure_cuda()
    torch.set_num_threads(2)
    runtime = load_sensor_runtime(args.bundle, args.device)
    fusion = load_bundle_fusion(args.bundle, args.variant)
    radius = None
    if args.variant == "evaluated":
        directory = args.bundle / args.variant
        seal = read(directory / "CALIBRATION_SEAL.json")
        if (
            file_sha(directory / "CALIBRATION.json") != seal["calibration_sha256"]
            or file_sha(directory / "FINAL_FREEZE.json") != seal["fusion_freeze_sha256"]
        ):
            raise ValueError("Interval calibration is not bound to this frozen model")
        radius = float(read(directory / "CALIBRATION.json")["radius_seconds_80"])
    calibration = load_calibration(
        args.sensors / args.sequence,
        "slider_candidate" if args.sequence.startswith("Slider") else "road_right",
    )
    result_rows = []
    with SensorSequence(args.sensors, args.sequence, slider_profile=args.profile) as reader:
        rows = reader.bbox_rows()
        boxes, clocks = (
            np.array([r["box_xyxy"] for r in rows]),
            np.array([r["anchor_us"] for r in rows]),
        )
        cached = SensorCache(reader)
        for row in rows[: args.limit or len(rows)]:
            prepared = prepare_query(cast(SensorSequence, cached), rows, row, calibration)
            result = runtime.predict(prepared.events, prepared.rgb, row["anchor_us"])
            native = np.array([result.model_ttc, result.rgb_ttc, result.event_ttc], np.float32)
            pred, physical = predict_history(fusion, native, boxes, clocks, row["anchor_us"])
            flags = []
            if not physical["valid"]:
                flags.append(str(physical["reason"]))
            elif np.isfinite(native[0]) and np.sign(native[0]) != np.sign(physical["ttc"]):
                flags.append("neural_geometry_sign_conflict")
            if not np.isfinite(pred):
                flags.append("prediction_unavailable")
            if radius is None:
                flags.append("uncertainty_unvalidated_after_all_data_refit")
            result_rows.append(
                {
                    "frame_index": row["frame_index"],
                    "anchor_us": row["anchor_us"],
                    "ttc_seconds": pred if np.isfinite(pred) else None,
                    "interval_80_seconds": (
                        [pred - radius, pred + radius]
                        if radius is not None and np.isfinite(pred)
                        else None
                    ),
                    "quality_flags": flags,
                    "geometry_count": physical["count"],
                    "variant": args.variant,
                }
            )
            print(f"predicted {len(result_rows)}/{args.limit or len(rows)}", flush=True)
    atomic_receipt(
        args.output, {"sequence": args.sequence, "labels_read": False, "rows": result_rows}
    )


if __name__ == "__main__":
    main()

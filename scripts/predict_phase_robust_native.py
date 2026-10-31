"""Target-blind native road replay using the exact audited left-RGB history contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from operational.phase_adapt_cache import atomic_receipt
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import load_bundle_fusion, verified_upstream_paths
from operational.phase_robust.data import histories
from operational.phase_robust.runtime import predict_history
from scripts.final_train40_transfer import FinalTransferPredictor, reject_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Sensor-only native population; rejects ground-truth fields",
    )
    parser.add_argument("--variant", choices=("evaluated", "all_data_refit"), default="evaluated")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read(args.manifest)["rows"]
    reject_targets(rows)
    if any("ttc_csv_path" in r for r in rows):
        raise ValueError("Use a sensor-only manifest without target file paths")
    tracks = histories(rows)
    runtime = FinalTransferPredictor(verified_upstream_paths(args.bundle), args.device)
    fusion = load_bundle_fusion(args.bundle, args.variant)
    radius = None
    if args.variant == "evaluated":
        radius = read(args.bundle / "evaluated/CALIBRATION.json")["radius_seconds_80"]
    output = []
    for row in rows[: args.limit or len(rows)]:
        result = runtime.predict(row)["predictions"]
        native = np.array(
            [result[k] for k in ("PHASE_ONLY_TRAIN40_S7", "R_CTX_TRAIN40", "E_CTX_TRAIN40")],
            np.float32,
        )
        boxes, clocks = tracks[row["sequence_id"]]
        pred, physical = predict_history(fusion, native, boxes, clocks, row["anchor_us"])
        flags = [physical["reason"]] if not physical["valid"] else []
        if physical["valid"] and np.sign(native[0]) != np.sign(physical["ttc"]):
            flags.append("neural_geometry_sign_conflict")
        if not np.isfinite(pred):
            flags.append("prediction_unavailable")
        if radius is None:
            flags.append("uncertainty_unvalidated_after_all_data_refit")
        output.append(
            {
                "query_id": row["query_id"],
                "ttc_seconds": pred if np.isfinite(pred) else None,
                "native_ttc": [float(x) if np.isfinite(x) else None for x in native],
                "quality_flags": flags,
                "interval_80_seconds": [pred - radius, pred + radius]
                if radius is not None and np.isfinite(pred)
                else None,
            }
        )
        print(f"native prediction {len(output)}/{args.limit or len(rows)}", flush=True)
    atomic_receipt(
        args.output,
        {
            "labels_read": False,
            "contract": "native_left_rgb_history",
            "variant": args.variant,
            "rows": output,
        },
    )


if __name__ == "__main__":
    main()

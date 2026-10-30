"""Fixed raw-event geometric comparator pilot, retaining unsupported estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.geometry.cmax import maximize_radial_event_contrast
from e_jepa_ttc.geometry.strttc import inverse_ttc_at_endpoint
from e_jepa_ttc.geometry.strttc_frontend import run_strttc_linear_frontend
from operational.evttc_competition.geometry import load_calibration
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.evaluation import metrics
from operational.phase_campaign.review_controls import endpoint_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with SensorSequence(
        args.external / "inputs", "Slider-500", slider_profile="event_aided_ttc"
    ) as reader:
        rows = reader.bbox_rows()
        selected = np.unique(np.linspace(0, len(rows) - 1, 12).astype(int))
        calibration = load_calibration(reader.sequence_dir, "slider_candidate")
        atomic_receipt(
            args.output / "FREEZE.json",
            {
                "queries": selected.tolist(),
                "history_us": 200000,
                "maximum_events": 50000,
                "roi": "current event box with 25 percent extent margin",
                "strttc": "existing linear frontend defaults; no nonlinear refinement",
                "cmax": "existing exponential radial warp defaults; current box center",
                "comparison": "local bounded implementations, not official reproductions",
                "script_sha256": file_sha(Path(__file__)),
                "external_freeze_sha256": file_sha(args.external / "FREEZE.json"),
            },
        )
        predictions = {k: np.full(len(selected), np.nan) for k in ("CMax", "STRTTC_linear")}
        details = []
        for j, index in enumerate(selected):
            row = rows[index]
            end = int(row["anchor_us"])
            raw = reader.read_events(max(reader.event_start_us, end - 200000), end)
            box = np.array(row["box_xyxy"], dtype=float)
            center = (box[:2] + box[2:]) / 2
            half = 0.75 * (box[2:] - box[:2])
            lo = np.maximum(0, np.floor(center - half)).astype(int)
            hi = np.minimum(
                [reader.event_width, reader.event_height], np.ceil(center + half)
            ).astype(int)
            raw = raw[np.all((raw[:, :2] >= lo) & (raw[:, :2] < hi), axis=1)].copy()
            if len(raw) > 50000:
                raw = raw[np.linspace(0, len(raw) - 1, 50000).astype(int)]
            detail = {"query": int(index), "events": len(raw), "failures": {}}
            if len(raw) < 100:
                detail["failures"]["all"] = "insufficient_events"
                details.append(detail)
                continue
            xy = raw[:, :2] - lo
            clocks = raw[:, 2] / 1e6
            width, height = map(int, hi - lo)
            cmax = maximize_radial_event_contrast(
                xy, clocks, raw[:, 3], image_shape=(height, width), center_xy=tuple(center - lo)
            )
            predictions["CMax"][j] = cmax.ttc_seconds - (end / 1e6 - float(clocks.max()))
            detail["failures"]["CMax"] = cmax.reason
            try:
                intrinsics = calibration.event_intrinsics.copy()
                intrinsics[2:] -= lo
                txyp = np.column_stack((clocks, xy, np.where(raw[:, 3] > 0, 1, -1)))
                fitted = run_strttc_linear_frontend(
                    txyp, width=width, height=height, intrinsics=tuple(intrinsics)
                )
                q = inverse_ttc_at_endpoint(
                    fitted.linear.inverse_ttc, end / 1e6 - fitted.absolute_reference_time_s
                )
                predictions["STRTTC_linear"][j], detail["failures"]["STRTTC_linear"] = (
                    endpoint_result(q)
                )
            except (ValueError, RuntimeError) as exc:
                detail["failures"]["STRTTC_linear"] = str(exc)
            details.append(detail)
            print(f"geometry {j + 1}/{len(selected)}", flush=True)
        save_npz(args.output / "predictions.npz", {**predictions, "indices": selected})
        # Predictions locked before opening previously sealed labels or PHASE errors.
        with np.load(args.external / "targets.npz") as f:
            target = f["target"][selected]
        with np.load(args.external / "predictions_sealed.npz") as f:
            predictions.update(F0=f["F0"][selected], F1=f["F1_7"][selected])
        shared = {}
        for method in ("CMax", "STRTTC_linear"):
            mask = (
                np.isfinite(target)
                & np.isfinite(predictions[method])
                & np.isfinite(predictions["F0"])
                & np.isfinite(predictions["F1"])
            )
            shared[method] = {
                k: metrics(target[mask], predictions[k][mask]) for k in (method, "F0", "F1")
            }
        atomic_receipt(
            args.output / "REPORT.json",
            {
                "metrics": {k: metrics(target, v) for k, v in predictions.items()},
                "paired_shared": shared,
                "details": details,
                "official_reproduction": False,
                "population": len(selected),
            },
        )


if __name__ == "__main__":
    main()

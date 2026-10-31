"""Reproduce score provenance and synthetic parallax without changing inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from operational.evttc_competition.geometry import load_calibration
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.evaluation import metrics


def pinhole_residual(
    rays: np.ndarray, rotation: np.ndarray, translation: np.ndarray, focal: np.ndarray, depth: float
) -> np.ndarray:
    """Pixel displacement relative to rotation-only rays at known synthetic depth."""
    rotated = rays @ rotation.T
    actual = depth * rotated + translation
    if depth <= 0 or np.any(actual[:, 2] <= 0) or np.any(rotated[:, 2] <= 0):
        raise ValueError("Synthetic points must be in front of both cameras")
    return focal * (actual[:, :2] / actual[:, 2:] - rotated[:, :2] / rotated[:, 2:])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--submitted", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    submitted = {
        r["query_id"]: r["prediction"]
        for r in json.loads(args.submitted.read_text(encoding="utf-8"))
    }
    scores = {}
    for speed in (750, 1000):
        with np.load(args.previous / f"external/Slider-{speed}/PREDICTIONS.npz") as z:
            fresh = z["native"][:, 0]
            clocks = z["clocks"]
        gtpath = args.previous / f"external_contract/{speed}_gt.csv"
        bboxpath = args.previous / f"external_contract/{speed}_bbox.csv"
        gt, bbox = [np.loadtxt(path, delimiter=",") for path in (gtpath, bboxpath)]
        target = []
        for row, clock in zip(bbox, clocks, strict=True):
            match = gt[gt[:, 0] == row[0]]
            if len(match) != 1 or abs(match[0, 1] * 1e6 - clock) > 2:
                raise ValueError("Public target/frame/sensor clock join failed")
            target.append(match[0, -1])
        original = np.array([submitted[f"Slider-{speed}:{i}"] for i in range(len(bbox))])
        scores[str(speed)] = {
            "rows": len(bbox),
            "official_pdf_relative_error_pct": 132.0 if speed == 750 else 156.88,
            "submitted_on_public_targets": metrics(np.array(target), original),
            "fresh_on_public_targets": metrics(np.array(target), fresh),
            "prediction_max_abs_difference_seconds": float(np.max(np.abs(fresh - original))),
            "public_bbox_sha256": file_sha(bboxpath),
            "public_gt_sha256": file_sha(gtpath),
        }
    calibration = load_calibration(args.calibration, "slider")
    # Candidate is RGB->event in mm; invert the full rigid transform for this audit.
    rotation = calibration.rotation_event_to_rgb
    translation = (
        -rotation @ np.array(calibration.metadata["translation_candidate_mm_median"]) / 1000
    )
    # Restrict the synthetic pinhole rays to the event camera's nominal image field.
    fx, fy, cx, cy = calibration.event_intrinsics
    width, height = calibration.event_resolution
    rays = np.array(
        [
            [(x * width - cx) / fx, (y * height - cy) / fy, 1.0]
            for x in (0.1, 0.5, 0.9)
            for y in (0.1, 0.5, 0.9)
        ]
    )
    residuals = []
    for depth in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        delta = pinhole_residual(rays, rotation, translation, calibration.rgb_intrinsics[:2], depth)
        residuals.append(
            {
                "synthetic_depth_m": depth,
                "median_pixels": float(np.median(np.linalg.norm(delta, axis=1))),
                "max_pixels": float(np.max(np.linalg.norm(delta, axis=1))),
            }
        )
    atomic_receipt(
        args.output,
        {
            "source_sha256": file_sha(Path(__file__)),
            "pdf_sha256": file_sha(args.pdf),
            "submitted_sha256": file_sha(args.submitted),
            "scores": scores,
            "score_status": (
                "unreconciled Slider750: PDF has aggregate/formula only, no aligned "
                "per-query targets/evaluator; no fitted shift, clipping or denominator change"
            ),
            "parallax": {
                "calibration_sha256": file_sha(args.calibration),
                "metadata": calibration.metadata,
                "synthetic_pinhole_residuals": residuals,
                "normalized_event_rays": rays.tolist(),
                "grid": "10/50/90 percent of each nominal event pinhole image axis",
                "scope": (
                    "candidate extrinsic, synthetic depths/rays, undistorted pinhole; "
                    "not measured real correspondence error; no GT depth used in "
                    "inference; no adapter change"
                ),
            },
        },
    )


if __name__ == "__main__":
    main()

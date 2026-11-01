"""Local reference profile aligned with eAP's returned 2026-11-01 configuration.

This is not the private scorer. Matching follows the public nuScenes reference;
raw public labels are retained without the historical camera-depth domain.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
ERRORS = ("trans_err", "scale_err", "orient_err")
PROFILE = {
    "schema": "phase_ttc_eap_observed_reference_v1",
    "classes": CLASSES,
    "distance_thresholds_m": THRESHOLDS,
    "tp_threshold_m": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 500,
    "mean_ap_weight": 5.0,
    "tp_metrics": ERRORS,
    "velocity_evaluated": False,
    "population": "all_raw_public_labels_no_domain_filter_no_ignored_objects",
    "distance": "ego_xy_euclidean_strict_less_than_threshold",
    "matching": "score_descending_greedy_nearest_unmatched_same_class",
    "score_ties": "reverse_input_index",
    "distance_ties": "first_gt_in_input_order",
    "interpolation": "numpy_interp_101_recall_points_right_zero_no_precision_envelope",
    "unsupported_class": "AP_zero_TP_error_one_included_in_macro",
    "private_scorer_equivalence": "unverified",
}
PROFILE_SHA256 = hashlib.sha256(json.dumps(PROFILE, sort_keys=True).encode()).hexdigest()
Box = Mapping[str, Any]
Array = NDArray[np.float64]


def box_array(boxes: Sequence[Box], *, prediction: bool) -> Array:
    """Validate native geometry and return XYZ/LWH/yaw/score rows."""
    rows = []
    for box in boxes:
        if box["class_name"] not in CLASSES:
            raise ValueError("Unknown detection class")
        xyz, size = box["translation_m"], box["size_lwh_m"]
        if len(xyz) != 3 or len(size) != 3:
            raise ValueError("Expected three coordinates and dimensions")
        rows.append([*xyz, *size, box["yaw_rad"], box["score"] if prediction else 0.0])
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    if not np.isfinite(values).all() or np.any(values[:, 3:6] <= 0):
        raise ValueError("Nonfinite box or nonpositive size")
    if prediction and (np.any(values[:, 7] < 0) or np.any(values[:, 7] > 1)):
        raise ValueError("Confidence must be in [0,1]")
    return values


def match_frame(pred: Array, gt: Array, offset: int) -> Array:
    """Match one class/frame independently, preserving global score-tie semantics.

    Columns: confidence, global input index, four TP flags, three errors at 2 m.
    Frame-local matching is equivalent to global matching because frames share
    no targets. Global ordering is restored before metric accumulation.
    """
    count = len(pred)
    rows = np.zeros((count, 9), dtype=np.float64)
    rows[:, 0] = pred[:, 7]
    rows[:, 1] = np.arange(offset, offset + count)
    if not count or not len(gt):
        return rows
    distances = np.linalg.norm(pred[:, None, :2] - gt[None, :, :2], axis=2)
    order = np.lexsort((-rows[:, 1], -rows[:, 0]))
    for t, threshold in enumerate(THRESHOLDS):
        available = np.ones(len(gt), dtype=bool)
        for i in order:
            candidates = np.where(available, distances[i], np.inf)
            j = int(np.argmin(candidates))
            if candidates[j] >= threshold:
                continue
            available[j] = False
            rows[i, 2 + t] = 1
            if threshold == 2.0:
                intersection = np.minimum(pred[i, 3:6], gt[j, 3:6]).prod()
                union = pred[i, 3:6].prod() + gt[j, 3:6].prod() - intersection
                rows[i, 6:] = (
                    distances[i, j],
                    1 - intersection / union,
                    abs((pred[i, 6] - gt[j, 6] + np.pi) % (2 * np.pi) - np.pi),
                )
    return rows


def summarize_class(rows: Array, gt_count: int) -> dict[str, Any]:
    """Reduce compact match records to the observed AP/TP arithmetic and curves."""
    rows = rows[np.lexsort((-rows[:, 1], -rows[:, 0]))]
    grid = np.linspace(0, 1, 101)
    curves: dict[str, Any] = {}
    for t, threshold in enumerate(THRESHOLDS):
        flags = rows[:, t + 2]
        tp = np.cumsum(flags)
        precision = np.zeros(101)
        confidence = np.zeros(101)
        envelope_ap = 0.0
        if gt_count and len(tp) and tp[-1]:
            recall = tp / gt_count
            raw_precision = tp / np.arange(1, len(tp) + 1)
            precision = np.interp(grid, recall, raw_precision, right=0)
            confidence = np.interp(grid, recall, rows[:, 0], right=0)
            envelope = np.maximum.accumulate(raw_precision[::-1])[::-1]
            indices = np.searchsorted(recall, grid, side="left")
            valid = indices < len(envelope)
            envelope_ap = float(envelope[indices[valid]].sum() / 101)
        curve: dict[str, Any] = {
            "AP": float(np.maximum(precision[11:] - 0.1, 0).mean() / 0.9),
            "TP": int(flags.sum()),
            "FP": int(len(rows) - flags.sum()),
            "recall": float(flags.sum() / gt_count) if gt_count else None,
            "diagnostic_unclipped_envelope_AP101_same_matches": envelope_ap,
            "precision": precision.tolist(),
            "confidence": confidence.tolist(),
        }
        if threshold == 2.0:
            positive = np.flatnonzero(confidence > 0)
            last = int(positive[-1]) if len(positive) else 0
            matched = rows[flags == 1]
            error_curves = np.ones((101, 3))
            if len(matched) and gt_count:
                means = np.cumsum(matched[:, 6:], axis=0) / np.arange(1, len(matched) + 1)[:, None]
                for e in range(3):
                    error_curves[:, e] = np.interp(
                        confidence[::-1], matched[::-1, 0], means[::-1, e]
                    )[::-1]
            curve["tp_errors"] = {
                name: float(error_curves[11 : last + 1, e].mean()) if last >= 11 else 1.0
                for e, name in enumerate(ERRORS)
            }
            curve["tp_error_curves"] = {
                name: error_curves[:, e].tolist() for e, name in enumerate(ERRORS)
            }
            curve["tp_last_valid_recall_index"] = last
        curves[str(threshold)] = curve
    return {
        "gt_count": gt_count,
        "pred_count": len(rows),
        "thresholds": curves,
        "AP": float(np.mean([v["AP"] for v in curves.values()])),
    }


class BenchmarkEvaluator:
    """Accumulate bounded per-frame matching records, without retaining images/boxes."""

    def __init__(self) -> None:
        self.records: dict[str, list[Array]] = {c: [] for c in CLASSES}
        self.counts: dict[str, int] = dict.fromkeys(CLASSES, 0)
        self.pred_counts: dict[str, int] = dict.fromkeys(CLASSES, 0)
        self.tokens: set[str] = set()

    def add_frame(self, token: str, predictions: Sequence[Box], targets: Sequence[Box]) -> None:
        """Add a unique sample; reject over-cap inputs instead of silently truncating."""
        if not token or token in self.tokens:
            raise ValueError("Empty or repeated sample token")
        if len(predictions) > 500:
            raise ValueError("More than 500 predictions in a sample")
        pred = box_array(predictions, prediction=True)
        gt = box_array(targets, prediction=False)
        for name in CLASSES:
            p = pred[[i for i, b in enumerate(predictions) if b["class_name"] == name]]
            g = gt[[i for i, b in enumerate(targets) if b["class_name"] == name]]
            self.counts[name] += len(g)
            if len(p):
                self.records[name].append(match_frame(p, g, self.pred_counts[name]))
            self.pred_counts[name] += len(p)
        self.tokens.add(token)

    def summarize(self) -> dict[str, Any]:
        """Return a small, serializable report with all eight classes in macro means."""
        if not self.tokens:
            raise ValueError("Cannot evaluate an empty population")
        classes = {
            name: summarize_class(
                np.concatenate(parts) if parts else np.empty((0, 9)), self.counts[name]
            )
            for name, parts in self.records.items()
        }
        mean_ap = float(np.mean([v["AP"] for v in classes.values()]))
        errors = {
            e: float(np.mean([v["thresholds"]["2.0"]["tp_errors"][e] for v in classes.values()]))
            for e in ERRORS
        }
        return {
            "profile": PROFILE,
            "profile_sha256": PROFILE_SHA256,
            "samples": len(self.tokens),
            "gt_boxes": sum(self.counts.values()),
            "pred_boxes": sum(self.pred_counts.values()),
            "classes": classes,
            "mAP": mean_ap,
            "tp_errors": errors,
            "eAP_score": (5 * mean_ap + sum(max(0, 1 - v) for v in errors.values())) / 8,
            "diagnostic_unclipped_envelope_AP101_same_population": float(
                np.mean(
                    [
                        t["diagnostic_unclipped_envelope_AP101_same_matches"]
                        for v in classes.values()
                        for t in v["thresholds"].values()
                    ]
                )
            ),
        }

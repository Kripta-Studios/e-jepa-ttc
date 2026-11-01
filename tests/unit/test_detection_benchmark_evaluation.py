"""Tests of the new additive profile; the historical evaluator is untouched."""

import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from e_jepa_ttc.detection.benchmark_evaluation import (
    ERRORS,
    BenchmarkEvaluator,
    box_array,
    match_frame,
    summarize_class,
)


def box(x=0.0, score=0.9, yaw=0.0, size=(4.0, 2.0, 1.0), category="car") -> dict[str, Any]:
    return dict(
        class_name=category, translation_m=[x, 0, 0], size_lwh_m=size, yaw_rad=yaw, score=score
    )


def records(pred, gt):
    return match_frame(box_array(pred, prediction=True), box_array(gt, prediction=False), 0)


def test_strict_boundary_duplicates_and_xy_only():
    rows = records([box(0.5), box(0, 0.8), box(0, 0.7)], [box()])
    assert rows[:, 2].tolist() == [0, 1, 0]
    assert rows[:, 3].tolist() == [1, 0, 0]
    high_z = box()
    high_z["translation_m"][2] = 1000
    assert records([high_z], [box()])[0, 2] == 1


def test_score_ties_reverse_input_and_gt_ties_first():
    rows = records([box(0), box(0.1)], [box()])
    assert rows[:, 2].tolist() == [0, 1]
    rows = records([box()], [box(-0.1, yaw=1), box(0.1, yaw=2)])
    assert rows[0, 8] == pytest.approx(1)


def test_dimension_and_wrapped_orientation_errors():
    rows = records([box(yaw=2 * math.pi - 0.1, size=(2, 2, 1))], [box(yaw=0.1)])
    assert rows[0, 6:].tolist() == pytest.approx([0, 0.5, 0.2])


def test_interpolation_is_not_envelope_and_minima_apply():
    rows = np.zeros((3, 9))
    rows[:, 0] = [0.9, 0.8, 0.7]
    rows[:, 1] = [0, 1, 2]
    rows[:, 2:6] = [[0] * 4, [1] * 4, [0] * 4]
    result = summarize_class(rows, 1)["thresholds"]["2.0"]
    grid = np.linspace(0, 1, 101)
    expected = np.interp(grid, [0, 1, 1], [0, 0.5, 1 / 3], right=0)
    assert result["precision"] == pytest.approx(expected)
    assert result["AP"] == pytest.approx(np.maximum(expected[11:] - 0.1, 0).mean() / 0.9)
    assert result["AP"] != result["diagnostic_unclipped_envelope_AP101_same_matches"]


@pytest.mark.parametrize("gt_count", [0, 10, 100])
def test_low_recall_no_gt_and_no_predictions(gt_count):
    rows = records([box()], [box()])
    result = summarize_class(rows, gt_count)["thresholds"]["2.0"]
    assert result["AP"] == 0
    assert result["tp_errors"] == dict.fromkeys(ERRORS, 1)
    empty = summarize_class(np.empty((0, 9)), gt_count)
    assert empty["AP"] == 0


def test_all_eight_classes_and_raw_far_population():
    evaluator = BenchmarkEvaluator()
    evaluator.add_frame("one", [box(100)], [box(100)])
    result = evaluator.summarize()
    assert result["mAP"] == pytest.approx(1 / 8)
    assert result["tp_errors"]["trans_err"] == pytest.approx(7 / 8)
    assert result["gt_boxes"] == 1
    with pytest.raises(ValueError, match="repeated"):
        evaluator.add_frame("one", [], [])
    with pytest.raises(ValueError, match="500"):
        evaluator.add_frame("two", [box()] * 501, [])
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluator.add_frame("three", [box(float("nan"))], [])


def test_per_frame_matching_equals_slow_global_reference():
    # Independent, deliberately slow global greedy matcher exercises competing
    # proposals, empty frames, exact score ties and shuffled frame order.
    rng = np.random.default_rng(713)
    predictions, targets, compact = [], [], []
    offset = 0
    gt_count = 0
    for frame in range(17):
        p = [
            box(float(x), float(s))
            for x, s in zip(rng.normal(0, 3, 13), rng.choice([0.1, 0.5, 0.9], 13), strict=True)
        ]
        g = [box(float(x)) for x in rng.normal(0, 3, frame % 7)]
        compact.append(
            match_frame(box_array(p, prediction=True), box_array(g, prediction=False), offset)
        )
        predictions.extend((frame, x) for x in p)
        targets.append(g)
        offset += len(p)
        gt_count += len(g)
    rows = np.concatenate(compact)
    from e_jepa_ttc.detection.benchmark_evaluation import THRESHOLDS

    for t, threshold in enumerate(THRESHOLDS):
        taken = set()
        flags = np.zeros(len(predictions))
        order = sorted(
            range(len(predictions)), key=lambda i: (predictions[i][1]["score"], i), reverse=True
        )
        for i in order:
            frame, p = predictions[i]
            candidates = [
                (abs(p["translation_m"][0] - g["translation_m"][0]), j)
                for j, g in enumerate(targets[frame])
                if (frame, j) not in taken
            ]
            if candidates:
                distance, j = min(candidates)
                if distance < threshold:
                    flags[i] = 1
                    taken.add((frame, j))
        np.testing.assert_array_equal(rows[:, t + 2], flags)
        recall = np.cumsum(flags[order]) / gt_count
        precision = np.cumsum(flags[order]) / np.arange(1, len(flags) + 1)
        curve = np.interp(np.linspace(0, 1, 101), recall, precision, right=0)
        summary = summarize_class(rows, gt_count)["thresholds"][str(threshold)]
        assert summary["precision"] == pytest.approx(curve)


def test_returned_official_ap_and_tp_arithmetic_all_classes():
    root = Path(__file__).resolve().parents[2]
    evidence = root / "docs/detection_sota_20261029/evidence/official_benchmark_20261101"
    summary = json.loads((evidence / "metrics_summary.json").read_text(encoding="utf-8"))
    details = json.loads((evidence / "metrics_details.json").read_text(encoding="utf-8"))
    for category, aps in summary["label_aps"].items():
        for threshold, ap in aps.items():
            d = details[f"{category}:{threshold}"]
            calculated = np.maximum(np.array(d["precision"])[11:] - 0.1, 0).mean() / 0.9
            assert calculated == pytest.approx(ap, abs=1e-14)
        d = details[f"{category}:2.0"]
        valid = np.flatnonzero(np.array(d["confidence"]) > 0)
        last = int(valid[-1]) if len(valid) else 0
        for name in ERRORS:
            calculated = np.mean(d[name][11 : last + 1]) if last >= 11 else 1
            assert calculated == pytest.approx(
                summary["label_tp_errors"][category][name], abs=1e-14
            )


def test_cli_rejects_incomplete_full_selection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    verify = runpy.run_path(str(root / "scripts/evaluate_detection_benchmark_local.py"))[
        "verify_shard_coverage"
    ]
    (tmp_path / "COMPARISON.json").write_text(
        json.dumps({"schema": "dfull_ranking_full_confirmation_v1", "frames": 2}), encoding="utf-8"
    )
    shard = tmp_path / "shards/000000"
    shard.mkdir(parents=True)
    (shard / "COMPLETE.json").write_text(json.dumps({"frames": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        verify(shard.parent)
    (shard / "COMPLETE.json").write_text(json.dumps({"frames": 2}), encoding="utf-8")
    assert verify(shard.parent)[1]["frames"] == 2

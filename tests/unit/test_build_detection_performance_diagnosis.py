from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "build_detection_performance_diagnosis.py"
    spec = importlib.util.spec_from_file_location("performance_diagnosis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load performance diagnosis script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_folded_samples_counts_stack_shares_not_lines(tmp_path: Path) -> None:
    folded = tmp_path / "raw.txt"
    folded.write_text("root;tarfile.py:10(read) 3\nroot;other.py:2(run) 1\n")
    result = cast(Any, _module()).parse_folded_samples(folded)
    assert result == {
        "total_samples": 4,
        "tarfile_stack_samples": 3,
        "tarfile_sample_fraction": 0.75,
        "unknown_empty_stack_samples": 0,
        "unknown_empty_stack_fraction": 0.0,
    }


def test_parse_folded_samples_keeps_empty_stack_unknown(tmp_path: Path) -> None:
    folded = tmp_path / "raw.txt"
    folded.write_text(" 3\nroot;tarfile.py:10(read) 1\n")
    result = cast(Any, _module()).parse_folded_samples(folded)
    assert result["unknown_empty_stack_samples"] == 3
    assert result["unknown_empty_stack_fraction"] == 0.75


def test_parse_folded_samples_rejects_empty_profile(tmp_path: Path) -> None:
    folded = tmp_path / "raw.txt"
    folded.write_text("not folded\n")
    with pytest.raises(ValueError, match="no folded samples"):
        cast(Any, _module()).parse_folded_samples(folded)


def test_first_float_after_uses_requested_section(tmp_path: Path) -> None:
    payload = tmp_path / "large.json"
    payload.write_text('{"old":{"metric":9.0},\n"metrics_v2": {\n"metric": 0.25\n}}\n')
    value = cast(Any, _module())._first_float_after(payload, "metrics_v2", "metric")
    assert value == 0.25


def test_metrics_prefix_is_cutoff_bound_and_contiguous(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            f'{{"update":{update},"preprocessing_seconds":1.0,'
            f'"forward_backward_update_seconds":2.0}}\n'
            for update in range(1025, 1030)
        )
    )
    rows, digest, byte_count = cast(Any, _module())._read_metrics_prefix(metrics, 1028)
    assert [row["update"] for row in rows] == [1025, 1026, 1027, 1028]
    assert len(digest) == 64
    assert byte_count < metrics.stat().st_size


def test_timing_window_reports_component_rate() -> None:
    rows = [
        {
            "update": 7,
            "preprocessing_seconds": 1.0,
            "forward_backward_update_seconds": 2.0,
        },
        {
            "update": 8,
            "preprocessing_seconds": 3.0,
            "forward_backward_update_seconds": 4.0,
        },
    ]
    timing = cast(Any, _module())._timing_window(rows)
    assert timing["median_component_sum_seconds_per_update"] == 5.0
    assert timing["updates_per_component_second"] == 0.2


def test_projection_uses_aggregate_mean_not_median() -> None:
    timing = {
        "update_count": 100,
        "observed_component_sum_seconds": 346.12597160000587,
        "median_component_sum_seconds_per_update": 2.73634489995311,
    }
    projection = cast(Any, _module())._conditional_projections(
        timing,
        basis_update=3420,
        target_update=8192,
        monitor_inference_seconds=406.85253029857995,
        full_selection_frame_median_seconds=0.39416095000342466,
    )
    expected_mean = 346.12597160000587 / 100
    assert projection["last100_component_mean_seconds_per_update"] == expected_mean
    assert projection["training_component_seconds_using_last100_aggregate_mean"] == (
        4772 * expected_mean
    )
    assert projection["training_component_seconds_using_last100_aggregate_mean"] != (
        4772 * 2.73634489995311
    )


def test_projection_counts_both_remaining_full_selection_passes() -> None:
    timing = {"update_count": 1, "observed_component_sum_seconds": 1.0}
    projection = cast(Any, _module())._conditional_projections(
        timing,
        basis_update=3420,
        target_update=8192,
        monitor_inference_seconds=1.0,
        full_selection_frame_median_seconds=0.5,
    )
    assert projection["remaining_full_selection_updates"] == [4096, 8192]
    assert projection[
        "full_selection_inference_seconds_per_pass_using_update3072_frame_median"
    ] == (23958 * 0.5)
    assert projection["remaining_full_selection_inference_seconds_all_passes"] == (2 * 23958 * 0.5)

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_runpod_windows.py"
SPEC = importlib.util.spec_from_file_location("analyze_runpod_windows", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sample(elapsed: float, updates: tuple[int, int], pending: tuple[int, int] = (0, 0)) -> dict:
    return {
        "elapsed_s": elapsed,
        "cuda_compute_pids": [101, 202],
        "fits": [
            {
                "directory": "/run/r2",
                "progress": {"pid": 101},
                "journal": {
                    "completed_updates": updates[0],
                    "pending_update_upper": pending[0],
                },
            },
            {
                "directory": "/run/r3",
                "progress": {"pid": 202},
                "journal": {
                    "completed_updates": updates[1],
                    "pending_update_upper": pending[1],
                },
            },
        ],
    }


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_concurrent_rate_uses_one_shared_wall_interval(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    _write_jsonl(samples, [_sample(10.0, (16, 16)), _sample(370.0, (616, 416))])

    result = MODULE.analyze_window(
        "parallel", samples, expected_seconds=360.0, duration_tolerance=0.01
    )

    assert result["status"] == "PASS"
    assert result["total_updates"] == 1000
    assert result["pod_aggregate_updates_per_min"] == pytest.approx(1000 / 6)
    assert result["average_worker_updates_per_min"] == pytest.approx(500 / 6)
    assert result["cuda_pid_validation"] == "PASS"


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (
            [_sample(0.0, (16, 16)), {**_sample(360.0, (15, 20))}],
            "completed_updates decreased",
        ),
        ([_sample(0.0, (16, 16)), _sample(360.0, (20, 20), (0, 1))], "terminal pending"),
    ],
)
def test_invalid_update_state_fails(tmp_path: Path, samples: list[dict], message: str) -> None:
    path = tmp_path / "samples.jsonl"
    _write_jsonl(path, samples)

    result = MODULE.analyze_window(
        "parallel", path, expected_seconds=360.0, duration_tolerance=0.01
    )

    assert result["status"] == "FAIL"
    assert any(message in violation for violation in result["violations"])


def test_cuda_pid_set_must_equal_all_expected_workers(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    first = _sample(0.0, (16, 16))
    last = _sample(360.0, (20, 20))
    last["cuda_compute_pids"] = [101]
    _write_jsonl(path, [first, last])

    result = MODULE.analyze_window(
        "parallel", path, expected_seconds=360.0, duration_tolerance=0.01
    )

    assert result["status"] == "FAIL"
    assert result["cuda_pid_validation"] == "FAIL"


def test_missing_cuda_pid_inventory_is_not_a_validated_window(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    first = _sample(0.0, (16, 16))
    last = _sample(360.0, (20, 20))
    first.pop("cuda_compute_pids")
    last.pop("cuda_compute_pids")
    _write_jsonl(path, [first, last])

    result = MODULE.analyze_window(
        "parallel", path, expected_seconds=360.0, duration_tolerance=0.01
    )

    assert result["status"] == "FAIL"
    assert result["cuda_pid_validation"] == "NOT_OBSERVED"


def test_baseline_v1_result_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"actual_seconds": 360.0, "fits": [{"updates": 1140}]}), encoding="utf-8"
    )

    result = MODULE.read_baseline(path)

    assert result["pod_aggregate_updates_per_min"] == pytest.approx(190.0)

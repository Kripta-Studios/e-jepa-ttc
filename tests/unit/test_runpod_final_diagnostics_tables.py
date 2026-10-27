"""Offline diagnostics preserve physical-query weighting and refuse stale final evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import runpod_final_diagnostics_tables as tables


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    directory = root / "ejepa_final_runs/final/test12/staged_sequences/diagnostics"
    directory.mkdir(parents=True)
    for index, duration in enumerate((0.1, 0.3)):
        token = f"physical_{index}"
        npz = directory / f"{index}.npz"
        np.savez_compressed(npz, **{"HEAD.output_point_phase": np.asarray([0.1])})
        write(
            npz.with_suffix(".json"),
            {
                "schema": "test12_staged_query_diagnostic_v1",
                "sample_token": token,
                "sequence_id": "sequence",
                "track_id": "track",
                "query_time_us": index,
                "prediction": 1.0,
                "run_binding_sha256": "a" * 64,
                "fallback_source": "PHASE_ONLY",
                "instrumented_eager_wall_seconds": duration,
                "timing_scope": "PREDICT_WITH_CACHE_AND_READ_ONLY_HOOKS_EXCLUDES_DIAGNOSTIC_IO",
                "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                "cache": {key: {"history_length": 3, "hits": 2} for key in ("event", "rgb")},
                "private_targets_read": False,
                "latency_benchmark": False,
                "gpu_kernel_time_measured": False,
            },
        )
    return root


def test_partial_is_deterministic_and_weights_physical_queries(
    workspace: Path, tmp_path: Path
) -> None:
    output = tmp_path / "report"
    result = tables.build_report(
        archive_root=None, live_root=workspace, output=output, mode="partial"
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    tables.build_report(archive_root=None, live_root=workspace, output=output, mode="partial")
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    assert result["physical_queries"] == 2
    assert result["instrumented_wall"]["p50_seconds"] == pytest.approx(0.2)
    assert result["instrumented_wall"]["p95_seconds"] == pytest.approx(0.29)
    assert result["status"] == "PARTIAL"


def test_npz_tamper_rejected(workspace: Path, tmp_path: Path) -> None:
    path = next(workspace.rglob("*.npz"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="NPZ differs"):
        tables.build_report(
            archive_root=None, live_root=workspace, output=tmp_path / "out", mode="partial"
        )


def test_final_refuses_missing_package_and_closed_telemetry(
    workspace: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="FINAL requires"):
        tables.build_report(
            archive_root=None, live_root=workspace, output=tmp_path / "out", mode="final"
        )


def test_final_alias_count_and_closed_sample_sha(
    workspace: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tables, "EXPECTED_PHYSICAL", 2)
    monkeypatch.setattr(tables, "EXPECTED_TOKENS", 3)
    final = workspace / "ejepa_final_runs/final/test12"
    write(
        final / "PREDICTION_RECEIPT.json",
        {
            "status": "COMPLETE",
            "physical_queries": 2,
            "population": 3,
            "private_targets_read": False,
        },
    )
    write(
        final / "PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json",
        {"status": "PACKAGED_NOT_SUBMITTED", "external_submission": False},
    )
    diag = workspace / "ejepa_final_diagnostics"
    diag.mkdir()
    sample = diag / "a_samples.jsonl"
    sample.write_text('{"utc":"2026-10-27T16:00:00+00:00"}\n')
    closed = {
        "status": "STOPPED",
        "sample_files": [
            {
                "path": "/workspace/ejepa_final_diagnostics/a_samples.jsonl",
                "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                "bytes": sample.stat().st_size,
            }
        ],
    }
    write(diag / "COLLECTOR_STOPPED.json", closed)
    result = tables.build_report(
        archive_root=None, live_root=workspace, output=tmp_path / "out", mode="final"
    )
    assert result["status"] == "FINAL" and result["physical_queries"] == 2
    closed["sample_files"][0]["sha256"] = "0" * 64
    write(diag / "COLLECTOR_STOPPED.json", closed)
    with pytest.raises(ValueError, match="Telemetry changed"):
        tables.build_report(
            archive_root=None, live_root=workspace, output=tmp_path / "out", mode="final"
        )


def test_resource_windows_keep_observed_span_and_counter_scope() -> None:
    rows = [
        {
            "utc": "first",
            "unix_seconds": 100,
            "gpu0.utilization.gpu": "10",
            "cgroup_cpu_usage_ns": "1000000000",
            "network.bytes_recv": 1000,
        },
        {
            "utc": "second",
            "unix_seconds": 120,
            "gpu0.utilization.gpu": "30",
            "cgroup_cpu_usage_ns": "4000000000",
            "network.bytes_recv": 1100,
        },
        {"utc": "third", "unix_seconds": 400, "gpu0.utilization.gpu": "0"},
    ]
    result = tables.resource_windows(rows)
    assert result[0]["observed_span_seconds"] == 20
    assert result[0]["gpu0.utilization.gpu.sample_mean"] == 20
    assert result[0]["cgroup_cpu_usage_ns.delta_per_second"] == pytest.approx(0.15)
    assert result[0]["network.bytes_recv.delta_per_second"] == 5
    assert result[1]["samples"] == 1
    assert "network.bytes_recv.delta_per_second" not in result[1]


def test_mixed_timing_scope_rejected(workspace: Path, tmp_path: Path) -> None:
    path = next(workspace.rglob("0.json"))
    value = json.loads(path.read_text())
    value["timing_scope"] = "ISOLATED_KERNEL_BENCHMARK"
    write(path, value)
    with pytest.raises(ValueError, match="scope differs"):
        tables.build_report(
            archive_root=None, live_root=workspace, output=tmp_path / "out", mode="partial"
        )

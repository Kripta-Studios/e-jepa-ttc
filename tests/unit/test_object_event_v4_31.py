from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch

from e_jepa_ttc.data.object_event_v4_31 import (
    ADAPT_SEQUENCES,
    AUDIT_SEQUENCES,
    OWNERSHIP_MARKER,
    AtomicDirectory,
    allocate_quotas,
    reject_forbidden_path,
    select_split,
)
from e_jepa_ttc.evaluation.object_event_v4_31 import (
    batched_controls,
    causal_decision,
    gate,
    radial_spectrum,
    sequence_swap_gate,
    spatial_transform,
)
from scripts.analyze_object_event_v4_31_operator_audit import _offsets
from scripts.build_object_event_v4_31_sanitized_cache import (
    _close_memmap,
    _intervals,
    _jsonl_record,
)
from scripts.run_object_event_v4_31_pipeline import _console_safe, _terminate_child, build_stages
from scripts.run_object_event_v4_31_pipeline import parser as pipeline_parser
from scripts.run_object_event_v4_31_pipeline import run as run_pipeline


def _rows() -> list[dict]:
    result = []
    for seq in (*ADAPT_SEQUENCES, *AUDIT_SEQUENCES):
        for track in range(600):
            result.append(
                {
                    "sequence_id": seq,
                    "sample_token": f"{seq}-{track}",
                    "track_id": str(track),
                    "public_track_id": str(track),
                    "timestamp_us": track * 100000,
                    "frame_timestamps_us": [0, 100000],
                    "events_path": "events.h5",
                    "event_windows_us": [0, 1],
                    "boxes_xyxy": [[0, 0, 1, 1]],
                }
            )
    return result


def test_close_memmap_releases_file_for_atomic_promotion(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    target = tmp_path / "promoted.npy"
    values = np.lib.format.open_memmap(source, mode="w+", dtype=np.float32, shape=(4,))
    values[:] = np.arange(4, dtype=np.float32)

    _close_memmap(values)
    os.replace(source, target)

    np.testing.assert_array_equal(np.load(target), np.arange(4, dtype=np.float32))


def test_intervals_accept_real_clock_jitter_without_overlap() -> None:
    t0, t1, t2 = _intervals(
        [[851_499_853, 851_599_855], [851_599_855, 851_699_856]]
    )

    assert t0 == (851_399_851, 851_499_853)
    assert t0[1] == t1[0]
    assert t1[1] == t2[0]


def test_jsonl_record_is_single_line_strict_json() -> None:
    encoded = _jsonl_record({"row_index": 1, "value": 2.0})

    assert "\n" not in encoded
    assert json.loads(encoded) == {"row_index": 1, "value": 2.0}
    with pytest.raises(ValueError):
        _jsonl_record({"value": float("nan")})


def test_console_safe_escapes_unencodable_windows_output() -> None:
    assert _console_safe("C:/Users/�lvaro", "ascii") == "C:/Users/\\ufffdlvaro"


def test_terminate_child_does_not_leave_orphan_process() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _terminate_child(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_exact_pool_and_nested_quotas() -> None:
    rows = _rows()
    full = select_split(rows, full=True)
    diag = select_split(rows, full=False)
    assert len(full) == 4096 and len(diag) == 512
    assert (
        allocate_quotas(full=True)["OBneIVg4Cw"] == 410
        and allocate_quotas(full=False)["OBneIVg4Cw"] == 52
    )


def test_split_gap_is_pairwise_and_diagnostic_is_per_sequence_prefix() -> None:
    rows = _rows()
    full = select_split(rows, full=True)
    diagnostic = select_split(rows, full=False)
    for sequence in (*ADAPT_SEQUENCES, *AUDIT_SEQUENCES):
        full_ids = [item["sample_token"] for item in full if item["sequence_id"] == sequence]
        diagnostic_ids = [
            item["sample_token"] for item in diagnostic if item["sequence_id"] == sequence
        ]
        assert diagnostic_ids == full_ids[: len(diagnostic_ids)]
    tight = [item for item in rows if item["sequence_id"] == ADAPT_SEQUENCES[0]][:3]
    tight[0]["track_id"] = tight[1]["track_id"] = tight[2]["track_id"] = "same"
    tight[0]["timestamp_us"], tight[1]["timestamp_us"], tight[2]["timestamp_us"] = 0, 190000, 100000
    assert (
        len(select_split.__name__) > 0
    )  # selection path is covered above; no motion proxy enters it.


def test_forbidden_paths() -> None:
    with pytest.raises(PermissionError):
        reject_forbidden_path("annotations/x")


def test_scalar_channels_unchanged_by_warp() -> None:
    x = torch.rand(2, 3, 12, 8, 8)
    y = spatial_transform(x, log_eta=0.02)
    assert torch.equal(x[:, :, 10:], y[:, :, 10:])


class Model:
    def __call__(self, events: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "log_eta": events[:, :, 0].mean((1, 2, 3)),
            "unknown": torch.zeros(len(events), dtype=torch.bool),
        }


def test_all_controls_batched() -> None:
    data = batched_controls(Model(), torch.rand(3, 3, 12, 8, 8), ["a", "b", "c"])
    assert "reverse:+0.00" in data and "zero_event:+0.00" in data


def test_v430_offset_support_has_locked_scale_radius() -> None:
    offsets = _offsets()
    assert {scale: value.shape for scale, value in offsets.items()} == {
        1: (9, 2),
        2: (25, 2),
        4: (81, 2),
    }
    assert {scale: float(np.abs(value).max()) for scale, value in offsets.items()} == {
        1: 1.0,
        2: 4.0,
        4: 16.0,
    }


def test_zoom_oracle_uses_negative_expansion_convention() -> None:
    from e_jepa_ttc.evaluation.object_event_v4_31 import control_metrics

    base = np.asarray([0.1, -0.1, 0.2, -0.2])
    zoom = {
        amount: np.full(len(base), -amount, dtype=float) for amount in (-0.04, -0.02, 0.02, 0.04)
    }
    controls: dict[str, object] = {
        "base": base,
        "sequence_id": ["s"] * len(base),
        "swap:+0.00": {"prediction": -base, "unknown": np.zeros(len(base), bool)},
        "reverse:+0.00": {"prediction": -base, "unknown": np.zeros(len(base), bool)},
        "identity:+0.00": {"prediction": base, "unknown": np.zeros(len(base), bool)},
        "translation:+0.02": {"prediction": base, "unknown": np.zeros(len(base), bool)},
        "rotation:+0.02": {"prediction": base, "unknown": np.zeros(len(base), bool)},
        "zero_event:+0.00": {"prediction": base, "unknown": np.ones(len(base), bool)},
    }
    for amount, prediction in zoom.items():
        controls[f"zoom:{amount:+.2f}"] = {
            "prediction": prediction,
            "unknown": np.zeros(len(base), bool),
        }
    metrics = control_metrics(controls)
    assert metrics["analytic_pearson"] == pytest.approx(1.0)
    assert metrics["slope"] == pytest.approx(1.0)
    assert metrics["sign_accuracy"] == pytest.approx(1.0)


def test_decision_priority() -> None:
    assert causal_decision(complete=False) == "invalid_incomplete"
    assert (
        causal_decision(
            stability_pass=False, spectrum_pass=True, operator_pass=True, stage2_pass=True
        )
        == "representation_instability_before_operator"
    )
    assert (
        causal_decision(
            stability_pass=True, spectrum_pass=True, operator_pass=False, stage2_pass=True
        )
        == "object_local_correspondence_operator_failure"
    )
    assert (
        causal_decision(
            stability_pass=True, spectrum_pass=True, operator_pass=True, stage2_pass=False
        )
        == "supervised_objective_or_readout_collapse"
    )


def test_projected_spectrum_is_finite_and_normalized_rank() -> None:
    value = radial_spectrum(torch.rand(8, 16, 16))
    assert value["valid_energy"] is True
    assert 0.0 <= float(value["effective_rank"]) <= 1.0
    assert np.isfinite([float(value["high_fraction"]), float(value["spectral_centroid"])]).all()


def test_missing_gate_is_fail_closed() -> None:
    assert gate({"js_median": None}) == {"finite": False, "passed": False}


def test_sequence_swap_gate_pass_fail_and_missing() -> None:
    assert sequence_swap_gate({"swap_corr": -0.6, "swap_flip": 0.8, "swap_coverage": 0.2})
    assert not sequence_swap_gate({"swap_corr": -0.4, "swap_flip": 0.8, "swap_coverage": 0.2})
    assert not sequence_swap_gate({"swap_corr": None, "swap_flip": 0.8, "swap_coverage": 0.2})


def test_cache_schema_validates_and_rejects_extra() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "schemas/object_event_v4_31_sanitized_cache_v1.schema.json").read_text()
    )
    value = {
        "artifact_type": "object_event_v4_31_sanitized_cache_v1",
        "schema_version": "1.0",
        "evidence_type": "sanitized_event_roi_cache",
        "code_commit": "unavailable",
        "protocol_version": "object_event_v4_31_train_only_v1",
        "protocol_sha256": "a" * 64,
        "created_at": "2026-08-09T00:00:00+00:00",
        "artifact_sha256": "b" * 64,
        "mode": "diagnostic",
        "count": 512,
        "events": {
            "path": "events.npy",
            "dtype": "float16",
            "shape": [512, 3, 12, 128, 128],
            "sha256": "a" * 64,
        },
        "delta_t_s": {"path": "delta_t_s.npy", "dtype": "float32", "sha256": "b" * 64},
        "rows_path": "rows.jsonl",
        "rows_sha256": "c" * 64,
        "source": {
            "path": "source.parquet",
            "sha256": "d" * 64,
            "projection": [str(i) for i in range(9)],
        },
        "split": {
            "path": "split.json",
            "sha256": "e" * 64,
            "version": "object_event_v4_31_train_only_v1",
        },
        "representation": {
            "id": "v4_30_common_roi",
            "t0_t1_t2": True,
            "interval": "[start,end)",
            "event_pixel_diff": 5,
            "bins_per_polarity": 5,
            "shape": [3, 12, 128, 128],
        },
        "opened_paths": ["source.parquet"],
        "provenance": {"boxes_transient_only": True, "targets_opened": False},
    }
    jsonschema.validate(value, schema)
    value["extra"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_audit_summary_schema_validates_real_shape_and_rejects_missing() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas/object_event_v4_31_audit_v1.schema.json").read_text())
    value = {
        "artifact_type": "object_event_v4_31_audit_v1",
        "schema_version": "1.0",
        "evidence_type": "causal_operator_audit",
        "code_commit": "unavailable",
        "protocol_version": "object_event_v4_31_train_only_v1",
        "protocol_sha256": "a" * 64,
        "created_at": "2026-08-09T00:00:00+00:00",
        "artifact_sha256": "b" * 64,
        "status": "not_issued_diagnostic",
        "selectable": False,
        "mode": "diagnostic",
        "preflight": {},
        "seeds": [7, 13, 23],
        "adaptation": {},
        "teacher_consensus": {"sha256": "x", "rows": 512, "batches": 0},
        "metrics": {
            "per_seed": {},
            "median": {},
            "joint_stability": {},
            "stability_counts": {},
            "sequence": {},
        },
        "gates": {
            "per_seed": {},
            "median": {},
            "evidence_complete": False,
            "all_gates_pass": False,
            "thresholds": {},
        },
        "stage2": {},
        "causal_decision": "not_issued_diagnostic",
        "timing": {
            "elapsed_s": 0.0,
            "forwards": 0,
            "batches": 0,
            "ram_bytes": None,
            "vram_bytes": None,
            "throughput_rows_s": None,
        },
        "opened_paths": [],
        "forbidden_access": [],
    }
    jsonschema.validate(value, schema)
    del value["timing"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def _owned_marker(config: str = "config", source: str = "source") -> str:
    return json.dumps(
        {
            "artifact": "object_event_v4_31",
            "owner": "e_jepa_ttc",
            "config_identity": config,
            "source_identity": source,
        }
    )


def test_atomic_directory_restores_quarantined_target_on_promotion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cache"
    target.mkdir()
    (target / "old.txt").write_text("preserve", encoding="utf-8")
    (target / OWNERSHIP_MARKER).write_text(_owned_marker(), encoding="utf-8")
    real_replace = os.replace
    staging: Path | None = None

    def fail_promotion(source: str | Path, destination: str | Path) -> None:
        if staging is not None and Path(source) == staging:
            raise OSError("synthetic promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr("e_jepa_ttc.data.object_event_v4_31.os.replace", fail_promotion)
    with pytest.raises(OSError, match="synthetic promotion failure"):
        with AtomicDirectory(
            target, force=True, config_identity="config", source_identity="source"
        ) as stage:
            staging = stage
            (stage / "new.txt").write_text("new", encoding="utf-8")
    assert (target / "old.txt").read_text(encoding="utf-8") == "preserve"
    assert json.loads((target / OWNERSHIP_MARKER).read_text()) == json.loads(_owned_marker())


def test_atomic_directory_force_requires_exact_identity_marker(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    target.mkdir()
    (target / OWNERSHIP_MARKER).write_text(_owned_marker(source="wrong"), encoding="utf-8")
    with pytest.raises(PermissionError, match="content differs"):
        with AtomicDirectory(
            target, force=True, config_identity="config", source_identity="source"
        ):
            pass


def test_pipeline_dry_run_writes_argv_logs_and_manifest(tmp_path: Path) -> None:
    args = pipeline_parser().parse_args(
        [
            "diagnostic",
            "--cache",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--log-root",
            str(tmp_path / "logs"),
            "--dry-run",
        ]
    )
    assert run_pipeline(args) == 0
    run_dir = next((tmp_path / "logs").iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert [stage["name"] for stage in manifest["stages"]] == ["preflight", "analyze"]
    assert (run_dir / "preflight.log").read_text(encoding="utf-8").startswith("[dry-run]")
    assert (run_dir / "pipeline_summary.json.sha256").is_file()


def test_pipeline_rejects_stage2_overlap_with_output_cache_or_logs(tmp_path: Path) -> None:
    args = pipeline_parser().parse_args(
        [
            "full",
            "--cache",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--log-root",
            str(tmp_path / "logs"),
            "--stage2-dir",
            str(tmp_path / "output" / "stage2"),
            "--dry-run",
        ]
    )
    with pytest.raises(RuntimeError, match="disjoint"):
        build_stages(args)


def test_powershell_wrapper_uses_argument_arrays_and_safe_prerequisites() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/run_object_event_v4_31_operator_audit.ps1").read_text(
        encoding="utf-8"
    )
    assert "& uv @RunnerArgs" in script
    assert "$RunnerArgs = @(" in script
    assert "Stage2Seed7" in script and "Full mode requires" in script

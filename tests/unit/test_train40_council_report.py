"""Receipt-derived reporting never promotes last-only parity or inferred cadence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import report_train40_resume as report


def write_receipt(root: Path, relative: str, value: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def timing(mean: float = 3.0) -> dict:
    return {
        "count": 2,
        "mean_ms": mean,
        "p50_ms": mean,
        "p95_ms": mean + 0.1,
        "p99_ms": mean + 0.2,
        "max_ms": mean + 0.3,
        "inverse_mean_capacity_hz_not_sustained_throughput": 1000 / mean,
    }


def compiled_record() -> dict:
    return {
        "index": 488,
        "objects": 1,
        "latency_ms": {"compiled_transport_graph": timing()},
        "parity": {"compiled_transport_graph": {"passed": True}},
        "replay_parity": [
            {
                "variant": "compiled_transport_graph",
                "stage": "initial_call",
                "iteration": 0,
                "parity": {"passed": False},
            },
            {
                "variant": "compiled_transport_graph",
                "stage": "measured",
                "iteration": 1,
                "parity": {"passed": True},
            },
        ],
    }


def test_compiled_scope_classifies_arbitrary_folder_and_embedded_binding(tmp_path):
    write_receipt(
        tmp_path,
        "native_log_sensitive_v1/summary.json",
        {
            "scope": report.COMPILED_PIPELINE_SCOPE,
            "status": "COMPLETE_WITH_PARITY_FAILURES",
            "binding": {"gpu": "recorded GPU", "native_entropy_log": True},
            "records": [compiled_record()],
        },
    )
    source = report.Sources(tmp_path)
    receipt = source.discover()[0]
    assert receipt.family == "compiled_pipeline"
    assert receipt.device == "recorded GPU"
    assert receipt.binding["native_entropy_log"] is True
    assert set(source.files) == {"native_log_sensitive_v1/summary.json"}


def test_first_replay_failure_cannot_be_hidden_by_last_success():
    record = compiled_record()
    assert report.compiled_route_parity(record, "compiled_transport_graph") == (
        "no [all-replays; 1/2]"
    )
    receipt = report.Receipt("run/summary.json", {"records": [record]}, {}, "compiled_pipeline")
    text = "\n".join(report.run_section(receipt))
    assert "| índice=488 | 1 | compiled_transport_graph |" in text
    assert "no [all-replays; 1/2]" in text
    assert "sí [last-only]" not in text


def test_replay_variants_have_independent_gates_and_all_stages_count():
    record = compiled_record()
    record["replay_parity"].append(
        {"variant": "original_graph", "stage": "warmup", "parity": {"passed": True}}
    )
    assert report.compiled_route_parity(record, "original_graph") == "sí [all-replays; 1/1]"
    assert report.compiled_route_parity(record, "compiled_transport_graph") == (
        "no [all-replays; 1/2]"
    )


def test_legacy_receipt_retains_explicit_last_only_label():
    record = compiled_record()
    del record["replay_parity"]
    assert report.compiled_route_parity(record, "compiled_transport_graph") == "sí [last-only]"
    assert report.compiled_route_parity(record, "unknown") == "— [last-only]"


def test_empty_or_incomplete_replay_population_never_falls_back_to_last_pass():
    record = compiled_record()
    record["replay_parity"] = []
    assert report.compiled_route_parity(record, "compiled_transport_graph") == (
        "no [all-replays; 0/0]"
    )
    record["replay_parity"] = [{"variant": "compiled_transport_graph", "parity": {"passed": True}}]
    record["replay_parity_coverage"] = {"schedule_exact": False}
    assert report.compiled_route_parity(record, "compiled_transport_graph") == (
        "no [all-replays; 1/1]"
    )
    record["replay_parity_coverage"] = {"expected_calls": 2, "observed_calls": 1}
    assert report.compiled_route_parity(record, "compiled_transport_graph") == (
        "no [all-replays; 1/1]"
    )


@pytest.mark.parametrize("value", [None, {}, ["invalid"]])
def test_malformed_replay_population_fails_explicitly(value):
    record = compiled_record()
    record["replay_parity"] = value
    with pytest.raises(ValueError, match="Malformed"):
        report.compiled_route_parity(record, "compiled_transport_graph")


def paced_receipt() -> dict:
    return {
        "schema": "train40_paced_service_summary_v1",
        "scope": "PACED_PERSISTENT_RAW_REPLAY_ARTIFICIAL_DISTINCT_ROIS_ALL_HOST_TTC",
        "status": "FAILED",
        "originals": {"passed": True},
        "separate_parity": {"performed": True, "passed": False},
        "milestone": {
            "target_object_count": 1,
            "milestone_with_parity_passed": False,
            "milestone_checks": {"offered_50hz_for_at_least_60s": False},
        },
        "capacity_with_parity_passed": False,
        "paced": {
            "offered_updates": 2,
            "grouped": {"passed": False, "expected_updates": 2, "expected_object_results": 2},
            "dropped_updates": 1,
            "source_errors": 0,
            "deadline_response_ms": timing(3),
            "raw_availability_response_ms": timing(7),
            "raw_availability_lateness_ms": timing(11),
            "capacity_checks": {"no_drops_or_source_errors": False},
        },
        "history_failure": "RAW_GAP",
    }


def test_paced_schema_separates_clocks_and_never_infers_missing_rate(tmp_path):
    source = tmp_path / "evidence"
    path = write_receipt(source, "named_anything/summary.json", paced_receipt())
    output = tmp_path / "report/RESULTS_GENERATED.md"
    manifest = report.generate(source, output)
    text = output.read_text(encoding="utf-8")
    assert report.Sources(source).discover()[0].family == "paced"
    assert "| rate_hz declarado | — |" in text
    assert "| descartes | 1 |" in text
    assert "| cobertura completa | no |" in text
    assert "| preservación de originales | sí |" in text
    assert "| hito con paridad declarado | no |" in text
    for name, mean in (
        ("deadline_response_ms", 3),
        ("raw_availability_response_ms", 7),
        ("raw_availability_lateness_ms", 11),
    ):
        assert f"| {name} | 2 | {mean} |" in text
    assert "RAW_GAP" in text
    assert "333.333" not in text
    assert manifest["sustained_50hz_claim"] is False
    assert manifest["sources"][0]["sha256"] == report.sha(path.read_bytes())


def test_paced_initialization_failure_is_preserved_without_fabricated_timings():
    receipt = report.Receipt(
        "failure/summary.json",
        {"status": "FAILED", "paced": None, "error": "SOURCE_ERROR"},
        {},
        "paced",
    )
    text = "\n".join(report.paced_section(receipt))
    assert "No hay distribuciones" in text and "SOURCE_ERROR" in text
    assert "| rate_hz declarado | — |" in text


def test_compact_limit_accepts_new_summary_size_and_still_excludes_binaries(tmp_path):
    source = tmp_path / "source"
    payload = {
        "scope": report.COMPILED_PIPELINE_SCOPE,
        "status": "FAILED",
        "error": "preserved failure",
        "padding": "x" * 2_100_000,
    }
    path = write_receipt(source, "run/summary.json", payload)
    (source / "model.pt").write_bytes(b"not evidence")
    write_receipt(source, "weights/summary.json", {"not": "a receipt"})
    sources = report.Sources(source)
    assert len(sources.discover()) == 1
    assert 2_000_000 < path.stat().st_size < report.MAX_RECEIPT_BYTES == 8_000_000
    destination = tmp_path / "published"
    manifest = report.copy_evidence(source, destination)
    assert [row["published"] for row in manifest["files"]] == ["run/summary.json"]
    assert not (destination / "model.pt").exists()
    assert not (destination / "weights").exists()


def test_compact_limit_remains_enforced(tmp_path, monkeypatch):
    path = write_receipt(tmp_path, "run/summary.json", {"padding": "x" * 200})
    monkeypatch.setattr(report, "MAX_RECEIPT_BYTES", 100)
    with pytest.raises(ValueError, match="compact limit"):
        report.Sources(tmp_path).read(path)


def test_exact_rgb_preserves_cold_failure_and_never_labels_it_ttc(tmp_path):
    result = {
        "schema": "train40_rgb_pil_exact_probe_v1",
        "scope": "CPU_RAW_READY_TO_EXACT_RGB_DEVICE_READY_NO_MODELS_NO_TTC_NO_DISK",
        "status": "PARITY_FAILED",
        "weights_or_normalizers_loaded": False,
        "all_exact": False,
        "coverage": {"passed": True, "expected": 2, "measured": 2},
        "cold_records": [
            {
                "index": 553,
                "objects": 1,
                "mode": "full_history_miss",
                "wall_ms": 900,
                "parity": {"passed": False},
            }
        ],
        "records": [
            {
                "index": 553,
                "objects": 1,
                "mode": "causal_prefix_rebuilt",
                "wall_ms": 8,
                "parity": {"passed": True},
            }
        ],
    }
    write_receipt(tmp_path, "arbitrary_name/summary.json", result)
    receipt = report.Sources(tmp_path).discover()[0]
    assert receipt.family == "rgb_exact"
    text = "\n".join(report.rgb_exact_section(receipt))
    assert "| 553 | 1 | primera preparación | full_history_miss | 900 | — | no |" in text
    assert "| 553 | 1 | tras primación | causal_prefix_rebuilt | 8 | — | sí |" in text
    assert "No acredita paridad TTC" in text
    assert "| modelos o normalizadores cargados | no |" in text

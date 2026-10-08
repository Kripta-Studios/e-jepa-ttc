from __future__ import annotations

import ast
import csv
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.sota_eval import cost


def _manifest() -> dict[str, object]:
    rows = []
    for family in range(8):
        for ordinal in range(2):
            rows.append(
                {
                    "query_id": f"sequence-{family}:{ordinal:02d}",
                    "sequence_id": f"sequence-{family}",
                    "scenario_family": f"family-{family}",
                    "anchor_us": ordinal,
                    "metadata_sha256": f"sha-{family}-{ordinal}",
                }
            )
    return {
        "status": "LABEL_FREE_MANIFEST",
        "forbidden_assets": sorted(cost.FORBIDDEN),
        "rows": rows,
    }


def test_stats_known_milliseconds() -> None:
    result = cost.summarize_ms([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "count": 4,
        "mean_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
        "minimum_ms": 1.0,
        "maximum_ms": 4.0,
        "samples_per_second_from_mean": 400.0,
    }


@pytest.mark.parametrize("values", [[], [-1.0], [math.nan], [math.inf]])
def test_stats_reject_empty_negative_or_nonfinite(values: list[float]) -> None:
    with pytest.raises(ValueError):
        cost.summarize_ms(values)


def test_selection_is_first_query_of_exactly_eight_families() -> None:
    rows = cost.select_fixed_queries(_manifest())
    assert [row["query_id"] for row in rows] == [f"sequence-{index}:00" for index in range(8)]


def test_selection_rejects_targets_and_wrong_family_count() -> None:
    document = _manifest()
    document["forbidden_assets"] = ["ttc.csv"]
    with pytest.raises(ValueError, match="target-bearing"):
        cost.select_fixed_queries(document)
    document = _manifest()
    document["rows"] = list(document["rows"])[:-2]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly eight"):
        cost.select_fixed_queries(document)


def test_resume_requires_exact_freeze_and_csv_bytes(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("a,b\n1,2\n", encoding="utf-8")
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    fragment = {
        "status": "COMPLETE",
        "query_id": "q",
        "execution_freeze_sha256": "freeze",
        "model_freeze_sha256": "models",
        "raw_csv_sha256": sha,
    }
    cost.validate_resume(fragment, raw, "freeze", "models", "q")
    raw.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV"):
        cost.validate_resume(fragment, raw, "freeze", "models", "q")


def test_query_failure_is_preserved_and_success_can_resume(tmp_path: Path, monkeypatch) -> None:
    # Exercise fragment-level primitives without importing models or touching CUDA.
    raw = tmp_path / "RAW.csv"
    rows = [
        {
            "query_id": "q",
            "sequence_id": "s",
            "scenario_family": "f",
            "system": cost.SYSTEMS[0],
            "stage": "gpu_inference",
            "iteration": 0,
            "warmup": "false",
            "milliseconds": "1.0",
            "status": "OK",
        }
    ]
    cost._write_csv(raw, rows)
    assert cost._read_csv(raw) == [{key: str(value) for key, value in rows[0].items()}]
    fragment = {
        "status": "FAILED_PRESERVED",
        "query_id": "q",
        "execution_freeze_sha256": "f",
        "model_freeze_sha256": "m",
        "raw_csv_sha256": cost._sha256(raw),
    }
    with pytest.raises(ValueError, match="identity/status"):
        cost.validate_resume(fragment, raw, "f", "m", "q")
    fragment["status"] = "COMPLETE"
    cost.validate_resume(fragment, raw, "f", "m", "q")


def test_import_surface_is_stdlib_only_until_run() -> None:
    source = Path(cost.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module.split(".")[0])
    assert "torch" not in top_imports
    assert "numpy" not in top_imports
    assert "operational" not in top_imports
    assert "e_jepa_ttc" not in top_imports


def test_gpu_guard_uses_real_entrypoint_not_ancestor_text() -> None:
    assert (
        cost._python_entrypoint(
            "python.exe", ["python.exe", "-m", "operational.sota_eval.full_prefetch", "--run"]
        )
        == "operational.sota_eval.full_prefetch"
    )
    assert (
        cost._python_entrypoint(
            "python.exe",
            [
                "python.exe",
                "-m",
                "operational.sota_eval.followups",
                "--note",
                "operational.sota_eval.fcwd_run",
            ],
        )
        == "operational.sota_eval.followups"
    )
    script = cost._python_entrypoint(
        "python.exe", ["python.exe", "C:/repo/operational/sota_eval/prefetch.py"]
    )
    assert script is not None and script.endswith("operational/sota_eval/prefetch.py")
    assert cost._python_entrypoint("powershell.exe", ["operational.sota_eval.fcwd_run"]) is None


def test_wddm_desktop_clients_are_recorded_without_claiming_exclusivity() -> None:
    rows = [
        "2764, Insufficient Permissions, [N/A]",
        "4000, C:/Program Files/Firefox/firefox.exe, [N/A]",
    ]
    clients, blockers = cost._classify_nvidia_clients(
        rows, {2764: "dwm.exe", 4000: "firefox.exe"}, own_pid=999
    )
    assert blockers == []
    assert [item["classification"] for item in clients] == [
        "wddm_desktop_environment",
        "wddm_desktop_environment",
    ]
    assert clients[0]["reported_process"] == "Insufficient Permissions"
    assert clients[0]["observed_process_name"] == "dwm.exe"


def test_wddm_unknown_or_python_gpu_client_remains_fail_closed() -> None:
    clients, blockers = cost._classify_nvidia_clients(
        [
            "7000, python.exe, 2048",
            "8000, mystery.exe, [N/A]",
            "9000, native_training.exe, 4096",
        ],
        {7000: "python.exe", 9000: "native_training.exe"},
        own_pid=999,
    )
    assert [item["classification"] for item in blockers] == [
        "other_python_gpu_client",
        "unresolved_nvidia_client",
        "other_native_gpu_client",
    ]
    assert blockers == clients


def test_nvidia_decodes_native_windows_bytes_without_text_pipe(monkeypatch) -> None:
    def fake_run(*_args, **kwargs):
        assert kwargs["capture_output"] is True
        assert "text" not in kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="C:/Users/Álvaro/python.exe, 1024\r\n".encode("cp1252"),
            stderr=b"",
        )

    monkeypatch.setattr(cost.locale, "getencoding", lambda: "cp1252")
    monkeypatch.setattr(cost.subprocess, "run", fake_run)
    assert cost._nvidia(("--query-compute-apps=pid",)) == {
        "available": True,
        "returncode": 0,
        "rows": ["C:/Users/Álvaro/python.exe, 1024"],
    }


def test_nvidia_nonzero_failure_remains_fail_closed(monkeypatch) -> None:
    response = SimpleNamespace(returncode=9, stdout=b"", stderr=b"driver failure")
    monkeypatch.setattr(cost.subprocess, "run", lambda *_args, **_kwargs: response)
    nonzero = cost._nvidia(())
    assert nonzero["available"] is False
    assert nonzero["returncode"] == 9


def test_aggregate_retains_all_measured_observations() -> None:
    rows = []
    for system in cost.SYSTEMS:
        for stage in ("cpu_prepare", "gpu_inference", "sequential_end_to_end"):
            for index, milliseconds in enumerate((1.0, 3.0)):
                rows.append(
                    {
                        "query_id": "q",
                        "sequence_id": "s",
                        "scenario_family": "f",
                        "system": system,
                        "stage": stage,
                        "iteration": str(index),
                        "warmup": "false",
                        "milliseconds": str(milliseconds),
                        "status": "OK",
                    }
                )
    summary = cost._aggregate(rows)
    assert summary[cost.SYSTEMS[0]]["gpu_inference"]["count"] == 2
    assert summary[cost.SYSTEMS[1]]["sequential_end_to_end"]["mean_ms"] == 2.0
    assert summary[cost.SYSTEMS[0]]["by_sequence"]["s"]["count"] == 2


def test_csv_contract_has_no_target_column(tmp_path: Path) -> None:
    assert not any("ttc" in field.lower() or "target" in field.lower() for field in cost.CSV_FIELDS)
    destination = tmp_path / "rows.csv"
    row = {field: "0" for field in cost.CSV_FIELDS}
    cost._write_csv(destination, [row])
    with destination.open(newline="", encoding="utf-8") as handle:
        assert csv.DictReader(handle).fieldnames == list(cost.CSV_FIELDS)


def test_freeze_serialization_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cost._atomic_json(tmp_path / "bad.json", {"x": math.nan})
    assert not (tmp_path / "bad.json").exists()


def test_prediction_digest_preserves_nonfinite_outcome_identity() -> None:
    assert cost._output_digest({"ttc": math.inf}) == cost._output_digest({"ttc": math.inf})
    assert cost._output_digest({"ttc": math.inf}) != cost._output_digest({"ttc": -math.inf})
    assert cost._output_digest({"ttc": math.nan}) != cost._output_digest({"ttc": 0.0})


def test_source_closure_detects_transitive_input_helper_mutation(tmp_path: Path) -> None:
    for index, relative in enumerate(cost.COST_SOURCE_FILES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}", encoding="utf-8")
    frozen = cost.source_closure(tmp_path)
    cost.validate_source_closure(tmp_path, frozen)
    changed = tmp_path / "src/e_jepa_ttc/efficient_context/mapped_union.py"
    changed.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="source closure"):
        cost.validate_source_closure(tmp_path, frozen)


def test_python_cpu_identity_binds_preparation_environment() -> None:
    identity = cost.python_cpu_identity()
    assert identity["python_version"]
    assert set(identity["packages"]) == {"numpy", "h5py", "PyYAML"}
    assert all(identity["packages"].values())
    assert set(identity["platform"]) == {"system", "release", "machine", "processor", "cpu_count"}
    assert identity["platform"]["system"]
    assert identity["platform"]["cpu_count"] is None or identity["platform"]["cpu_count"] > 0


def test_measurement_covers_three_systems_and_shares_cpu_bundle(monkeypatch) -> None:
    monkeypatch.setattr(cost, "CPU_WARMUPS", 1)
    monkeypatch.setattr(cost, "CPU_MEASUREMENTS", 1)
    monkeypatch.setattr(cost, "GPU_WARMUPS", 0)
    monkeypatch.setattr(cost, "GPU_MEASUREMENTS", 1)
    monkeypatch.setattr(cost, "E2E_MEASUREMENTS", 1)
    calls = {"h8_prepare": 0, "full_prepare": 0}

    def prepare_h8(_row):
        calls["h8_prepare"] += 1
        return {"own_events": "own", "garl_events": "eo", "delta_t_s": 0.1, "valid": "v"}

    def prepare_full(_row):
        calls["full_prepare"] += 1
        return {"sensor": "full", "unavailable_reason": None}

    class Own:
        bindings = {}

        def predict(self, own_events, delta_t_s, valid):
            return {"H8_seed7": 1.0, "H8_seed13": 2.0, "H8_seed23": 3.0}

        def garl_predict(self, sensor):
            return {"ttc": 4.0, "heights": [1.0, 2.0]}

    class Full:
        bindings = {}

        def predict_sensor(self, sensor):
            return {"ttc": 5.0, "heights": [1.0, 2.0]}

    row = {"query_id": "q", "sequence_id": "s", "scenario_family": "f"}
    raw, audit = cost._measure_query(row, prepare_h8, prepare_full, Own(), Full(), lambda: None)
    assert {item["system"] for item in raw} == set(cost.SYSTEMS)
    # Two shared CPU-prep observations, then one independent realistic E2E run per branch.
    assert calls == {"h8_prepare": 4, "full_prepare": 3}
    h8_cpu = [
        item["milliseconds"]
        for item in raw
        if item["system"] == cost.SYSTEMS[0] and item["stage"] == "cpu_prepare"
    ]
    eo_cpu = [
        item["milliseconds"]
        for item in raw
        if item["system"] == cost.SYSTEMS[1] and item["stage"] == "cpu_prepare"
    ]
    assert eo_cpu == h8_cpu
    assert set(audit["prediction_sha256"]) == set(cost.SYSTEMS)

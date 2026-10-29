"""CPU-only recovery and label-separation tests with synthetic FCWD-shaped receipts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from operational.efficient_context.common import Lease, atomic_json, digest
from operational.sota_eval import fcwd_run, fcwd_score


class FakeOwn:
    bindings = {"synthetic": True}

    def __init__(self):
        self.calls = 0

    def predict(self, own_events, delta_t_s, valid, *, availability_lag_s=0.0):
        self.calls += 1
        return {"H8_seed7": 1.0, "H8_seed13": 2.0, "H8_seed23": 3.0}

    def garl_predict(self, sensor):
        return {"ttc": math.inf, "heights": [1.0, 1.0]}


def prepared(row):
    return {
        "own_events": np.zeros((1,), dtype=np.float32),
        "garl_events": np.ones((1,), dtype=np.float32),
        "garl_full_sensor": None,
        "garl_full_unavailable_reason": "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING",
        "valid": np.ones(8, dtype=bool),
        "delta_t_s": 0.1,
        "metadata": {"targets_read": False},
    }


def population(tmp_path, count=3):
    raw = tmp_path / "sensor.hdf5"
    raw.write_bytes(b"synthetic sensor container; not a real HDF5")
    stat = raw.stat()
    rows = [
        {
            "query_id": f"q{i}",
            "sequence_id": f"FCWD{i % 3 + 1}",
            "anchor_us": 2_000_000,
            "anchor_relative_seconds": 2.0,
            "raw_path": str(raw),
            "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        }
        for i in range(count)
    ]
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        {"schema": "fcwd_label_free_population_v1", "status": "FROZEN_LABEL_FREE", "rows": rows},
    )
    output = tmp_path / "run"
    output.mkdir()
    atomic_json(output / "SOURCE_FREEZE.json", {"sources": {}})
    atomic_json(output / "INFERENCE_FREEZE.json", {"synthetic": True})
    return manifest, rows, output, digest(output / "INFERENCE_FREEZE.json")


def test_resume_skips_committed_fragment_and_retains_nonfinite_and_dependency(tmp_path):
    manifest, rows, output, binding = population(tmp_path)
    model = FakeOwn()
    assert not fcwd_run.execute_queries(rows, output, binding, model, None, prepared, limit=1)
    original = (output / "predictions/query_00000.json").read_bytes()
    assert fcwd_run.execute_queries(rows, output, binding, model, None, prepared)
    assert model.calls == 3
    assert (output / "predictions/query_00000.json").read_bytes() == original
    fcwd_run.seal_predictions(output, manifest, rows)
    _, predictions = fcwd_run.verify_seal(output, manifest)
    assert len(predictions) == 3
    assert predictions[0]["ttc"]["public_Garl_event_lhr"] is None
    assert predictions[0]["unavailable_reasons"]["public_Garl_rgb_event_full"] == (
        "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING"
    )
    assert predictions[0]["input_tensors"]["own_events"]["shape"] == [1]


def test_failure_preserves_completed_work_and_leases_clear(tmp_path):
    _, rows, output, binding = population(tmp_path)
    model = FakeOwn()

    def broken(row):
        if row["query_id"] == "q1":
            raise OSError("simulated disconnected drive")
        return prepared(row)

    with pytest.raises(OSError, match="disconnected"):
        with Lease(output):
            fcwd_run.execute_queries(rows, output, binding, model, None, broken)
    assert not (output / "WRITER.lock").exists()
    first = (output / "predictions/query_00000.json").read_bytes()
    assert fcwd_run.execute_queries(rows, output, binding, model, None, prepared)
    assert (output / "predictions/query_00000.json").read_bytes() == first
    assert len(list((output / "failures").glob("*.json"))) == 1


def test_orphan_is_preserved_recomputed_and_tampering_rejected(tmp_path):
    manifest, rows, output, binding = population(tmp_path, 1)
    model = FakeOwn()
    assert fcwd_run.execute_queries(rows, output, binding, model, None, prepared)
    path = output / "predictions/query_00000.json"
    orphan_bytes = path.read_bytes()
    path.with_suffix(".sha256").unlink()
    assert fcwd_run.execute_queries(rows, output, binding, model, None, prepared)
    assert model.calls == 2
    assert next((output / "orphaned").glob("*.json")).read_bytes() == orphan_bytes
    fcwd_run.seal_predictions(output, manifest, rows)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        fcwd_run.verify_seal(output, manifest)


def test_no_seal_no_target_manifest_read(tmp_path, monkeypatch):
    manifest, _, output, _ = population(tmp_path)

    def forbidden_read(path):
        pytest.fail(f"Scorer accessed a target dependency before the prediction seal: {path}")

    monkeypatch.setattr(fcwd_score, "read", forbidden_read)
    with pytest.raises(FileNotFoundError):
        fcwd_score.score_sealed(manifest, output, tmp_path / "never_read.json")
    assert not (output / "WRITER.lock").exists()


def test_manifest_rejects_duplicate_ids_targets_and_changed_raw(tmp_path):
    manifest, rows, _, _ = population(tmp_path)
    value = fcwd_run.read(manifest)
    value["rows"].append(rows[0])
    with pytest.raises(ValueError, match="Duplicate"):
        fcwd_run.validate_manifest(value)
    with pytest.raises(ValueError, match="forbidden"):
        fcwd_run.reject_targets({"metadata": {"truth_ttc_seconds": 5}})
    Path(rows[0]["raw_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="stat changed"):
        fcwd_run.validate_manifest(fcwd_run.read(manifest))


def test_gt_join_never_extrapolates_or_bridges_nonfinite_endpoints():
    times = np.asarray([1.0, 3.0])
    targets = np.asarray([4.0, 2.0])
    assert fcwd_score.interpolate_truth(times, targets, 2.0) == (3.0, "INTERPOLATED_LINEAR")
    assert fcwd_score.interpolate_truth(times, targets, 1.0) == (4.0, "EXACT")
    value, status = fcwd_score.interpolate_truth(times, targets, 0.5)
    assert math.isnan(value) and status == "OUTSIDE_MEASURED_GT_SUPPORT"
    value, status = fcwd_score.interpolate_truth(times, np.array([math.nan, 2]), 2.0)
    assert math.isnan(value) and status == "NONFINITE_GT_ENDPOINT"
    with pytest.raises(ValueError, match="strictly increasing"):
        fcwd_score.interpolate_truth(np.array([1, 1]), targets, 1.0)


def test_synthetic_sealed_scoring_retains_outside_support_and_full_dependency(tmp_path):
    manifest, rows, output, binding = population(tmp_path, 6)
    for row in rows[3:]:
        row["anchor_us"] = 500_000
        row["anchor_relative_seconds"] = 0.5
    assets = []
    for sequence in range(1, 4):
        path = tmp_path / "FCWD_dataset" / "sealed_labels" / f"sequence_{sequence}" / "gt_ttc.csv"
        path.parent.mkdir(parents=True)
        path.write_text("0,1,999,888,4\n1,3,999,888,2\n", encoding="utf-8")
        assets.append(
            {
                "name": f"fcwd_{sequence}_ttc",
                "role": "ttc_target",
                "path": str(path),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
        )
    asset_manifest = tmp_path / "assets.json"
    atomic_json(asset_manifest, {"assets": assets})
    reference = tmp_path / "reference.json"
    atomic_json(reference, {"synthetic": True})
    value = fcwd_run.read(manifest)
    value.update(
        rows=rows,
        asset_manifest_sha256=digest(asset_manifest),
        reference_contract_path=str(reference),
        reference_contract_sha256=digest(reference),
    )
    atomic_json(manifest, value)
    fcwd_run.execute_queries(rows, output, binding, FakeOwn(), None, prepared)
    fcwd_run.seal_predictions(output, manifest, rows)
    receipt = fcwd_score.score_sealed(manifest, output, asset_manifest, bootstrap_draws=30)
    assert receipt["status"] == "COMPLETE"
    assert receipt["cohort"]["gt_nonfinite"] == 3
    assert receipt["cohort"]["gt_eligible"] == 3
    assert len(receipt["methods"]) == 4
    coverage = json.loads((output / "scoring/MODEL_COVERAGE.json").read_text())
    assert coverage["full_model"]["status"] == "DEPENDENCY_UNAVAILABLE"
    with (output / "scoring/SCORED_PREDICTIONS.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 6
    assert saved[0]["truth_ttc_seconds"] == "3.0"
    assert saved[-1]["truth_ttc_seconds"] == ""


def test_source_change_stops_before_model_loading_and_releases_both_guards(tmp_path, monkeypatch):
    manifest, _, output, _ = population(tmp_path)
    monkeypatch.setattr(fcwd_run, "source_binding", lambda *args: {"new_source": True})
    guard = tmp_path / "guard"
    with pytest.raises(ValueError, match="Source/config freeze changed"):
        fcwd_run.run(
            manifest,
            output,
            tmp_path / "old_campaign",
            tmp_path / "full",
            tmp_path / "code",
            "cpu",
            gpu_guard_root=guard,
        )
    assert not (output / "WRITER.lock").exists()
    assert not (guard / "WRITER.lock").exists()
    assert fcwd_run.read(output / "STATE.json")["status"] == "FAILED_PRESERVED"


def test_source_closure_covers_reviewed_numerical_helpers():
    closure = set(fcwd_run.local_source_closure())
    assert {
        "src/e_jepa_ttc/simplex_t/context_raw_union.py",
        "src/e_jepa_ttc/simplex_t/query_context_voxel.py",
        "src/e_jepa_ttc/data/eap_representation.py",
        "src/e_jepa_ttc/representations/voxel_grid.py",
        "src/e_jepa_ttc/data/types.py",
        "src/e_jepa_ttc/data/eap.py",
        "src/e_jepa_ttc/models/local_transport.py",
        "src/e_jepa_ttc/models/collision_clock_math.py",
        "src/e_jepa_ttc/training/incremental_residual.py",
        "src/e_jepa_ttc/simplex_t/expert_phase.py",
        "src/e_jepa_ttc/evaluation/stage61_nested_pair_router.py",
        "operational/simplex_t_cost_context/model.py",
        "operational/simplex_t_post_campaign/route_policy.py",
    } <= closure


@pytest.mark.parametrize("mutation", ["helper", "initializer"])
def test_transitive_helper_byte_mutation_rejects_resume_before_models(
    tmp_path, monkeypatch, mutation
):
    manifest, _, output, _ = population(tmp_path)
    package = tmp_path / "src/e_jepa_ttc"
    package.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_text("# package\n", encoding="utf-8")
    entry = package / "entry.py"
    entry.write_text("from . import helper\n", encoding="utf-8")
    helper = package / "helper.py"
    helper.write_text("def transform(x): return x * 2\n", encoding="utf-8")
    monkeypatch.setattr(fcwd_run, "ROOT", tmp_path)
    monkeypatch.setattr(fcwd_run, "SOURCE_FILES", ("src/e_jepa_ttc/entry.py",))
    campaign = tmp_path / "old_campaign"
    for name in ("TRAINING_PROTOCOL.json", "DELIVERY_FREEZE.json", "H8_FEATURE_MANIFEST.json"):
        atomic_json(campaign / name, {"synthetic": True})
    for fit in ("a5_seed7", "c2f_seed7", "pair_seed7", "h8_seed7", "h8_seed13", "h8_seed23"):
        atomic_json(campaign / "fits" / fit / "CHECKPOINT_RECEIPT.json", {"synthetic": True})
    args = (manifest, campaign, tmp_path / "full", tmp_path / "code", "cpu")
    binding = fcwd_run.source_binding(*args)
    assert set(binding["sources"]) == {
        "src/e_jepa_ttc/__init__.py",
        "src/e_jepa_ttc/entry.py",
        "src/e_jepa_ttc/helper.py",
    }
    atomic_json(output / "SOURCE_FREEZE.json", binding)
    changed_path = helper if mutation == "helper" else initializer
    changed_path.write_text("# changed scientific helper\n", encoding="utf-8")
    guard = tmp_path / "guard"
    with pytest.raises(ValueError, match="Source/config freeze changed"):
        fcwd_run.run(manifest, output, *args[1:], gpu_guard_root=guard)
    assert not (output / "WRITER.lock").exists()
    assert not (guard / "WRITER.lock").exists()
    assert fcwd_run.read(output / "STATE.json")["status"] == "FAILED_PRESERVED"


def test_gpu_guard_rejects_cuda_but_not_cpu_or_score_processes(monkeypatch):
    import psutil

    class FakeProcess:
        def __init__(self, pid, command):
            self.pid = pid
            self.info = {"pid": pid, "name": "python.exe", "cmdline": command}

    processes = [
        FakeProcess(981234, ["python", "-m", "operational.sota_eval.fcwd_run", "--score"]),
        FakeProcess(981235, ["python", "-m", "operational.evttc_transfer.run", "--device", "cpu"]),
    ]
    parents = psutil.Process().parents()
    if parents:
        processes.append(
            FakeProcess(parents[0].pid, ["python", "-m", "operational.sota_eval.fcwd_run"])
        )
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: iter(processes))
    fcwd_run.assert_no_competing_gpu_processes()
    processes.append(
        FakeProcess(981236, ["python", "-m", "operational.evttc_transfer.run", "--device", "cuda"])
    )
    with pytest.raises(RuntimeError, match="981236"):
        fcwd_run.assert_no_competing_gpu_processes()


@pytest.mark.parametrize(
    "command,expected",
    [
        (["-m", "operational.sota_eval.prefetch", "run", "--device", "cuda"], True),
        (["-m", "operational.sota_eval.prefetch", "pilot"], False),
        (["-m", "operational.sota_eval.full_prefetch", "run"], True),
        (["-m", "operational.sota_eval.full_prefetch", "pilot"], False),
        (["-m", "operational.sota_eval.full_prefetch", "run", "--device=cpu"], False),
        (["-m", "operational.sota_eval.cost"], True),
        (["-m", "operational.sota_eval.r1_resume", "--execute"], True),
        (["-m", "operational.sota_eval.r1_resume"], False),
        (["-m", "operational.sota_eval.r1_resume_supervisor", "--run"], False),
        (["-m", "operational.sota_eval.campaign", "--device", "cuda"], False),
        (["C:/repo/operational/sota_eval/prefetch.py", "run"], True),
        (["C:/repo/operational/sota_eval/r1_resume_supervisor.py", "--run"], False),
    ],
)
def test_gpu_entrypoints_are_exact_and_distinguish_planning_modes(command, expected):
    assert fcwd_run.gpu_worker_command(["python", *command]) is expected

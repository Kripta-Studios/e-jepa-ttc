"""Exercise preservation and interruption handling without touching a CUDA device."""

import argparse
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_train40_runtime as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    component = candidate / "weight.pt"
    candidate.mkdir()
    component.write_bytes(b"frozen")
    official = candidate / "official.zip"
    official.write_bytes(b"official predictions")
    source = tmp_path / "runtime.py"
    source.write_text("frozen source", encoding="utf-8")
    write(
        candidate / "PORTABLE_PATHS.json",
        {
            "components": {
                name: {"local_path": str(path), "sha256": runner.sha(path)}
                for name, path in (("weight", component), ("official", official))
            }
        },
    )
    binding = {"model_seal_sha256": "seal", "components": {"weight": runner.sha(component)}}
    manifest = tmp_path / "inputs" / "manifest.json"
    rows = [{"query_id": f"q{i}", "index": i} for i in range(3)]
    write(manifest, {"rows": rows})
    ref = tmp_path / "reference" / "own"
    freeze = ref.parent / "INFERENCE_FREEZE.json"
    common = {
        "manifest_sha256": runner.sha(manifest),
        "targets_read": False,
        "optimizer_updates": 0,
    }
    write(
        freeze,
        {
            **common,
            "model_seal_sha256": "seal",
            "model_hashes": binding["components"],
            "sources": {},
        },
    )
    results, fragments = [], {}
    for index, row in enumerate(rows):
        result = {
            **common,
            "binding_sha256": runner.sha(freeze),
            "query_id": row["query_id"],
            "predictions": {
                key: 2.0 for key in ("PHASE_ONLY_TRAIN40_S7", "E_CTX_TRAIN40", "R_CTX_TRAIN40")
            },
            "rgb_available": True,
            "rgb_unavailable_reason": "",
            "history_lengths": {"event": 4, "rgb": 5},
            "seconds": {"preprocess": 0.0, "total": 0.001},
        }
        path = ref / f"query_{index:05d}.json"
        write(path, result)
        fragments[f"own/{path.name}"] = runner.sha(path)
        results.append(result)
    write(
        ref.parent / "PREDICTIONS_SEALED.json",
        {
            **common,
            "binding_sha256": runner.sha(freeze),
            "fragments": fragments,
            "schema": "final_train40_transfer_predictions_v1",
            "queries": len(rows),
            "status": "COMPLETE",
        },
    )
    args = argparse.Namespace(
        candidate=candidate,
        manifest=manifest,
        reference=ref,
        output=tmp_path / "new_run",
        modes=["eager"],
        stride=1,
        indices=None,
        repeats=1,
        warmup=0,
        synchronize_stages=False,
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "verify_candidate", lambda path: ({"weight": component}, dict(binding))
    )
    monkeypatch.setattr(runner, "verify_frozen_runtime", lambda path: {"source_sha256": {}})
    monkeypatch.setattr(runner, "_source_inventory", lambda *unused: [source])
    monkeypatch.setattr(runner, "load_models", lambda *unused: ({}, {}))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "commit\n")
    fake_torch = SimpleNamespace(
        set_num_threads=lambda n: None,
        manual_seed=lambda n: None,
        device=lambda d: d,
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "CPU test double",
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 0,
        ),
    )
    monkeypatch.setattr(runner, "torch", fake_torch)
    state = {"calls": 0, "closed": False, "behavior": None, "close_error": False}

    class FakePredictor:
        def __init__(self, *unused, **kwargs):
            self.inputs = SimpleNamespace(close=lambda: None)

        def predict(self, row):
            state["calls"] += 1
            if state["behavior"] is not None:
                state["behavior"](row)
            return json.loads(json.dumps(results[row["index"]]))

        def close(self):
            state["closed"] = True
            if state["close_error"]:
                raise RuntimeError("close failure")

    monkeypatch.setattr(runner, "PersistentTrain40Predictor", FakePredictor)
    return args, state, source, binding, rows, official


def test_success_binds_official_files_and_sources(fixture):
    args, state, source, _, _, official = fixture
    runner.run_benchmark(args)
    summary = runner.read(args.output / "summary.json")
    inventory = runner.read(args.output / "original_snapshot.json")
    assert summary["status"] == "COMPLETE" and summary["passed"]
    assert not summary["target_50hz_demonstrated"]
    assert str(official.resolve()) in inventory and str(source.resolve()) in inventory
    assert state["closed"]


def test_model_load_failure_still_verifies_originals(fixture, monkeypatch):
    args, state, *_ = fixture
    monkeypatch.setattr(
        runner, "load_models", lambda *unused: (_ for _ in ()).throw(RuntimeError("load failure"))
    )
    with pytest.raises(RuntimeError, match="load failure"):
        runner.run_benchmark(args)
    summary = runner.read(args.output / "summary.json")
    assert summary["status"] == "FAILED" and summary["originals"]["passed"]
    assert state["calls"] == 0


def test_interrupt_keeps_partial_population_and_survives_cleanup_failure(fixture):
    args, state, *_ = fixture

    def interrupt(row):
        if row["index"] == 1:
            raise KeyboardInterrupt("requested stop")

    state.update(behavior=interrupt, close_error=True)
    with pytest.raises(KeyboardInterrupt, match="requested stop"):
        runner.run_benchmark(args)
    summary = runner.read(args.output / "summary.json")
    assert summary["status"] == "INTERRUPTED" and summary["originals"]["passed"]
    assert any("close failure" in message for message in summary["errors"])
    assert summary["routes"]["eager"][0]["parity"]["candidate_coverage"]["missing_ids"] == [
        "q1",
        "q2",
    ]


def test_mid_run_source_mutation_invalidates_evidence(fixture):
    args, state, source, *_ = fixture
    state["behavior"] = lambda row: source.write_text("changed while running", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Benchmark failed"):
        runner.run_benchmark(args)
    summary = runner.read(args.output / "summary.json")
    assert not summary["passed"] and str(source.resolve()) in summary["originals"]["changed"]


def test_changed_reference_bytes_rejected_before_gpu(fixture):
    args, state, *_ = fixture
    path = args.reference / "query_00001.json"
    data = runner.read(path)
    data["predictions"]["PHASE_ONLY_TRAIN40_S7"] = 2.5
    write(path, data)
    with pytest.raises(ValueError, match="prediction bytes differ"):
        runner.run_benchmark(args)
    assert state["calls"] == 0
    assert runner.read(args.output / "summary.json")["status"] == "FAILED"


def test_reference_candidate_binding_rejected(fixture):
    args, _, _, binding, rows, _ = fixture
    with pytest.raises(ValueError, match="does not bind"):
        runner._validate_reference(
            args.reference, args.manifest, rows, {**binding, "components": {}}
        )


def test_output_ancestor_and_descendant_rejected(tmp_path):
    protected = tmp_path / "original" / "candidate"
    for output in (protected, protected / "new", tmp_path / "original"):
        with pytest.raises(ValueError, match="frozen source"):
            runner._validate_output(output, [protected])


@pytest.mark.parametrize(
    "options", [["--modes", "eager", "eager"], ["--warmup", "-1"], ["--repeats", "0"]]
)
def test_invalid_options_rejected_before_output(tmp_path, options):
    output = tmp_path / "never_created"
    args = [
        "--candidate",
        "candidate",
        "--manifest",
        "manifest",
        "--reference",
        "reference",
        "--output",
        str(output),
        *options,
    ]
    with pytest.raises(SystemExit):
        runner.main(args)
    assert not output.exists()

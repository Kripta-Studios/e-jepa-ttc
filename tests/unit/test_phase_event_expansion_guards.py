"""Reject inconsistent confirmation and causal-cache evidence before inference."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from operational.phase_event_expansion.model import ExpansionResult, SolverConfig
from operational.phase_event_expansion.synthetic import make_fixture
from operational.phase_event_expansion.validation import (
    bind_confirmation,
    require_confirmation_binding,
    validate_cache_manifest,
    validate_cached_window,
    validate_population_tokens,
)
from scripts.benchmark_phase_event_expansion_synthetic import KINDS, score_fixture
from scripts.cache_phase_event_expansion_development import require_new_cache


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def confirmation(tmp_path):
    sources = {}
    for name in (
        "operational/phase_event_expansion/model.py",
        "operational/phase_event_expansion/synthetic.py",
        "scripts/benchmark_phase_event_expansion_synthetic.py",
    ):
        payload = Path(name).read_bytes()
        destination = tmp_path / "executed_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        sources[name] = hashlib.sha256(payload).hexdigest()
    config = {
        "cohort": "confirmation",
        "seeds": [31, 47],
        "kinds": list(KINDS),
        "events_per_fixture": 4096,
        "solver": asdict(SolverConfig(objective="temporal_ncc", reference="endpoint")),
        "source_sha256": sources,
    }
    rows = []
    for seed in (31, 47):
        for kind in KINDS:
            fixture = make_fixture(seed, kind)
            motion = kind in ("expansion", "receding")
            result = ExpansionResult(
                fixture.q if motion else None,
                tuple(fixture.translation) if motion else None,
                1 / fixture.q if motion else None,
                motion,
                "available" if motion else "flat_objective",
                0.0,
            )
            row = {
                "seed": seed,
                "kind": kind,
                "split": "evaluation",
                "truth": {"q": fixture.q, "translation": fixture.translation.tolist()},
                "events_sha256": hashlib.sha256(fixture.events.tobytes()).hexdigest(),
                "result": asdict(result),
                "assessment": score_fixture(fixture, result),
            }
            rows.append(row)
            write(tmp_path / f"{seed}_{kind}.json", row)
    summary = {
        "rows": rows,
        "all_synthetic_gates_pass": True,
        "evaluation_total": 12,
        "evaluation_passed": 12,
    }
    write(tmp_path / "CONFIG.json", config)
    write(tmp_path / "SUMMARY.json", summary)
    return tmp_path


@pytest.mark.parametrize(
    "corruption",
    ["cohort", "missing", "duplicate", "gate", "truth", "events", "assessment", "archive"],
)
def test_binding_refuses_unverified_confirmation(confirmation, corruption):
    config = json.loads((confirmation / "CONFIG.json").read_text())
    summary = json.loads((confirmation / "SUMMARY.json").read_text())
    if corruption == "cohort":
        config["seeds"] = [7, 13]
    elif corruption == "missing":
        summary["rows"].pop()
    elif corruption == "duplicate":
        summary["rows"][-1] = summary["rows"][0]
    elif corruption == "gate":
        config["solver"]["minimum_temporal_agreement"] = 0.1
    elif corruption == "truth":
        summary["rows"][0]["truth"]["q"] += 1
    elif corruption == "events":
        summary["rows"][0]["events_sha256"] = "wrong"
    elif corruption == "assessment":
        summary["rows"][0]["assessment"]["passed"] = False
    else:
        (confirmation / "executed_source/operational/phase_event_expansion/model.py").write_text(
            "changed"
        )
    write(confirmation / "CONFIG.json", config)
    write(confirmation / "SUMMARY.json", summary)
    with pytest.raises(ValueError):
        bind_confirmation(confirmation)
    assert not (confirmation / "CONFIRMATION_BINDING.json").exists()


def test_binding_is_explicit_immutable_and_detects_config_summary_changes(confirmation):
    with pytest.raises(ValueError, match="binding"):
        require_confirmation_binding(confirmation)
    bind_confirmation(confirmation)
    require_confirmation_binding(confirmation)
    with pytest.raises(FileExistsError):
        bind_confirmation(confirmation)
    for filename in ("CONFIG.json", "SUMMARY.json"):
        path = confirmation / filename
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="binding"):
            require_confirmation_binding(confirmation)
        path.write_bytes(original)


def cache_pair():
    rows = [
        {
            "query_id": f"Slider-500:{i}",
            "row_index": i,
            "frame_index": i,
            "query_us": i * 100,
            "file": f"query_{i:05d}.npz",
        }
        for i in range(77)
    ]
    calibration = {"event_intrinsics": [1, 1, 0, 0]}
    manifest = {
        "sequence_id": "Slider-500",
        "queries": 77,
        "rows": rows,
        "calibration": calibration,
        "status": "complete",
        "labels_in_cache": False,
    }
    freeze = {
        "sequence_id": "Slider-500",
        "calibration": copy.deepcopy(calibration),
        "queries": [{"frame_index": r["frame_index"], "anchor_us": r["query_us"]} for r in rows],
    }
    return manifest, freeze


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "clock", "frame", "calibration", "path"]
)
def test_cache_identity_guards(corruption):
    manifest, freeze = cache_pair()
    validate_cache_manifest(manifest, freeze)
    if corruption == "missing":
        manifest["rows"].pop()
    elif corruption == "duplicate":
        manifest["rows"][-1] = manifest["rows"][0]
    elif corruption == "clock":
        manifest["rows"][0]["query_us"] += 1
    elif corruption == "frame":
        manifest["rows"][0]["frame_index"] += 1
    elif corruption == "calibration":
        manifest["calibration"]["event_intrinsics"][0] = 2
    else:
        manifest["rows"][0]["file"] = "../query_00000.npz"
    with pytest.raises(ValueError):
        validate_cache_manifest(manifest, freeze)


def test_window_clock_and_population_duplicates_rejected():
    row = {
        "query_us": 100,
        "window_start_us": 0,
        "window_end_exclusive_us": 101,
        "roi_xyxy": [0, 0, 10, 10],
        "selected_count": 1,
    }
    arrays = {
        "x": np.array([1]),
        "y": np.array([1]),
        "t_us": np.array([100]),
        "p": np.array([0]),
        "query_us": np.array(100),
        "roi_xyxy": np.array([0, 0, 10, 10]),
    }
    validate_cached_window(arrays, row)
    arrays["query_us"] = np.array(99)
    with pytest.raises(ValueError):
        validate_cached_window(arrays, row)
    validate_population_tokens(np.array(["a", "b"]))
    with pytest.raises(ValueError):
        validate_population_tokens(np.array(["a", "a"]))


def test_existing_cache_is_refused_without_writes(tmp_path):
    sentinel = tmp_path / "CACHE_FREEZE.json"
    sentinel.write_text("frozen")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        require_new_cache(tmp_path)
    assert sentinel.read_text() == "frozen"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    require_new_cache(tmp_path / "new")
    assert not (tmp_path / "new").exists()

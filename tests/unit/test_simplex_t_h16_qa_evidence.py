"""Synthetic 64-file evidence validates the reader, not real GPU parity."""

import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import h16_qa_evidence as module


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "artifacts/simplex_t/T0/qa"
    cache = tmp_path / "artifacts/simplex_t/T1/context_features_fp32"
    root.mkdir(parents=True)
    cache.mkdir(parents=True)
    arrays = {
        key: np.ones(2, dtype=np.float32)
        for key in (
            "features145",
            "expert_ttc",
            "known",
            "pair_features",
            "observation_ids",
            "anchor_us",
            "available_us",
        )
    }
    references, results = [], []
    for query in range(64):
        name = f"family00_query{query:05d}"
        item = {"reference_stem": name, "query": query}
        references.append(item)
        np.savez(cache / f"{name}.npz", **arrays)
        np.savez(root / f"{name}.npz", **arrays)
        result = {
            "reference": item,
            "status": "EXACT_PASS",
            "error": None,
            "sha256": sha256(root / f"{name}.npz"),
        }
        (root / f"{name}.json").write_text(json.dumps(result), encoding="utf-8")
        results.append(result)
    bound = {"missing": [], "references": references}
    (root / "CONTRACT.json").write_text(
        json.dumps({"references": bound, "execution_identity": {"fixture": True}}), encoding="utf-8"
    )
    report = {
        "status": "SAME_LAYOUT_H16_EXACT_PASS",
        "completed": 64,
        "optimizer_updates": 0,
        "scientific_admission": False,
        "contract_sha256": sha256(root / "CONTRACT.json"),
        "results": results,
    }
    monkeypatch.setattr(module, "bind_h16_reference_receipts", lambda work: bound)
    return tmp_path, root, report


@pytest.mark.parametrize(
    "corruption",
    ["", "count", "duplicate", "payload", "rehash_changed_array", "identity", "resource"],
)
def test_reader_requires_complete_exact_arrays_and_identity(evidence, corruption):
    work, root, report = evidence
    if corruption == "count":
        report["completed"] = 63
    elif corruption == "duplicate":
        report["results"][1] = report["results"][0]
    elif corruption in {"payload", "rehash_changed_array"}:
        path = root / "family00_query00000.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["expert_ttc"][0] += 1
        np.savez(path, **arrays)
        if corruption == "rehash_changed_array":
            report["results"][0]["sha256"] = sha256(path)
            path.with_suffix(".json").write_text(json.dumps(report["results"][0]), encoding="utf-8")
    (root / "QA.json").write_text(json.dumps(report), encoding="utf-8")
    calls = []

    def identity(record):
        calls.append(record)
        if corruption == "identity":
            raise ValueError("invalid executable identity")

    def run():
        return module.verify_h16_replay(
            work,
            root,
            sha256(root / "QA.json"),
            validate_execution_identity=identity,
            resource_ok=lambda: corruption != "resource",
        )

    if corruption:
        with pytest.raises(InterruptedError if corruption == "resource" else ValueError):
            run()
    else:
        assert run()["queries"] == 64
        assert len(calls) == 2


@pytest.mark.parametrize("change", ["", "code", "device", "preprocessing", "missing_inventory"])
def test_execution_identity_requires_real_declared_inventory(tmp_path, monkeypatch, change):
    work = tmp_path
    base = work / "artifacts/simplex_t"
    fields = {
        "expert_features.py": "extractor_sha256",
        "expert_phase.py": "expert_phase_sha256",
        "query_context_voxel.py": "voxel_sha256",
        "context_raw_union.py": "union_reader_sha256",
    }
    cache = {"preprocessing_sha256": "a" * 64, "torch": "fixture-runtime"}
    code = {}
    relatives = ["src/e_jepa_ttc/simplex_t/" + name for name in fields]
    relatives += [
        "src/e_jepa_ttc/simplex_t/" + name
        for name in ("expanded_inference.py", "h16_qa_execution.py", "h16_qa_plan.py")
    ]
    relatives += ["scripts/run_simplex_t_h16_replay_qa.py"]
    for relative in relatives:
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source", encoding="utf-8")
        code[str(Path(relative))] = sha256(path)
        if path.name in fields:
            cache[fields[path.name]] = sha256(path)
    for relative, value in (
        ("T1/context_features_fp32/IDENTITY.json", cache),
        ("T0/coherent_fp32_extractor_receipt/QA.json", {"runtime": {"device": "fixture-GPU"}}),
    ):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    local = work / "local.json"
    local.write_text(
        json.dumps({"worktree": str(work), "shared_coordination": str(work)}), encoding="utf-8"
    )
    monkeypatch.setattr(module, "verified_ack", lambda *args: {})
    record = {
        "code": code,
        "local_paths_sha256": sha256(local),
        "ack_sha256": "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318",
        "preprocessing_sha256": cache["preprocessing_sha256"],
        "device": "fixture-GPU",
        "torch": cache["torch"],
    }
    if change == "code":
        (work / relatives[0]).write_text("changed", encoding="utf-8")
    elif change == "device":
        record["device"] = "another GPU"
    elif change == "preprocessing":
        record["preprocessing_sha256"] = "b" * 64
    elif change == "missing_inventory":
        record["code"] = {}
    if change:
        with pytest.raises(ValueError):
            module.verify_h16_execution_identity(work, local, record)
    else:
        module.verify_h16_execution_identity(work, local, record)

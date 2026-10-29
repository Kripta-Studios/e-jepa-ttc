"""The D0 capacity option is bound to an immutable real QA, not a free override."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("changed", [None, "report", "union", "index", "preprocessing", "probe"])
def test_capacity_option_rejects_changed_evidence(tmp_path, monkeypatch, changed):
    module = runpy.run_path("scripts/run_simplex_t_cached_io_replay.py")
    validate = module["validated_capacity"]
    report = tmp_path / "qa.json"
    union = tmp_path / "src/e_jepa_ttc/simplex_t/context_raw_union.py"
    probe = tmp_path / "artifacts/simplex_t/T0/probe_union_capacity_20260908.py"
    index = tmp_path / "index/query_context_index.npz"
    prep = tmp_path / "preprocessing.json"
    report.write_text(
        json.dumps(
            dict(
                union_source_sha256="union",
                script_sha256="probe",
                index_sha256="index",
                preprocessing_sha256="preprocessing",
                limitations="fixture",
            )
        )
    )
    hashes = {
        report: "435027b73c2d434441573e866c3ce43323633e3fddf5d35a883795e692e77b03",
        union: "union",
        probe: "probe",
        index: "index",
        prep: "preprocessing",
    }
    if changed is not None:
        target = {
            "report": report,
            "union": union,
            "probe": probe,
            "index": index,
            "preprocessing": prep,
        }[changed]
        hashes[target] = "changed"
    monkeypatch.setitem(validate.__globals__, "sha256", lambda path: hashes[Path(path)])
    args = ["--index", str(index.parent), "--preprocessing-manifest", str(prep)]
    if changed is not None:
        with pytest.raises(ValueError, match="QA"):
            validate(tmp_path, report, args)
    else:
        result = validate(tmp_path, report, args)
        assert result["retained_bytes_max"] == 536870912
        assert result["path"] == str(report.resolve())

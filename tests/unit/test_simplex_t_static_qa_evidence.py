"""Small static-report fixtures; no lint subprocess, model, or optimizer."""

import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.static_qa_evidence import verify_ruff_comparison


@pytest.mark.parametrize(
    "failure", ["", "source", "raw_bytes", "counts", "hidden_new", "new", "resource"]
)
def test_static_diagnostic_evidence(tmp_path: Path, failure: str):
    root = tmp_path / "artifacts/simplex_t/T0/static"
    baseline = root / "baseline_source"
    (baseline / "src").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/a.py"
    previous = baseline / "src/a.py"
    for path in (source, previous):
        path.write_text("x = 1\n", encoding="utf-8")
    config = tmp_path / "pyproject.toml"
    config.write_text("# fixture\n", encoding="utf-8")

    def diagnostic(path: Path, message: str):
        return {
            "filename": str(path),
            "code": "E501",
            "message": message,
            "location": {"row": 1, "column": 1},
        }

    before = [diagnostic(previous, "old")]
    after = [diagnostic(source, "new" if failure in {"new", "hidden_new"} else "old")]
    old_id, new_id = "src/a.py:1:1:E501:old", "src/a.py:1:1:E501:new"
    changed = failure in {"new", "hidden_new"}
    report = {
        "status": "STATIC_FAILURE_IDS_COMPARED_NOT_FULL_QA_OR_SCIENTIFIC_ADMISSION",
        "baseline_commit": "57b39cb2b9a5ec8378755f8f350634c007aa822e",
        "position_sensitive_ids": True,
        "optimizer_updates": 0,
        "tests_executed": 0,
        "common_rule_config_sha256": sha256(config),
        "current_source_sha256": {"src/a.py": sha256(source)},
        "baseline_source_files": [{"path": "src/a.py", "sha256": sha256(previous)}],
        "runs": {
            "baseline": {"exit_code": 1, "failures": 1},
            "current": {"exit_code": 1, "failures": 2 if failure == "counts" else 1},
        },
        "persistent_failure_ids": [] if changed else [old_id],
        "new_or_relocated_failure_ids": [new_id] if failure == "new" else [],
        "resolved_or_relocated_failure_ids": [old_id] if changed else [],
    }
    for name, data in (
        ("COMPARISON.json", report),
        ("BASELINE_RUFF.json", {"diagnostics": before}),
        ("CURRENT_RUFF.json", {"diagnostics": after}),
    ):
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    pins = {
        name: sha256(root / name)
        for name in ("COMPARISON.json", "BASELINE_RUFF.json", "CURRENT_RUFF.json")
    }
    if failure == "source":
        source.write_text("changed", encoding="utf-8")
    if failure == "raw_bytes":
        (root / "CURRENT_RUFF.json").write_text("{}", encoding="utf-8")
    if failure:
        with pytest.raises(InterruptedError if failure == "resource" else ValueError):
            verify_ruff_comparison(
                tmp_path, root, pins=pins, resource_ok=lambda: failure != "resource"
            )
    else:
        result = verify_ruff_comparison(tmp_path, root, pins=pins, resource_ok=lambda: True)
        assert result["persistent_failures"] == 1
        assert result["new_failures"] == 0
        assert result["scientific_admission"] is False

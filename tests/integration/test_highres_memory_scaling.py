from __future__ import annotations

from tests.integration._artifact_helpers import read_artifact


def test_r4_scaling_records_oom_guard_before_global_attention() -> None:
    scaling = read_artifact("artifacts/benchmarks/highres_token_scaling_v1.json")
    r4 = next(row for row in scaling["results"] if row["name"] == "R4")
    assert scaling["status"] == "pass"
    assert r4["tokens"] == 4800
    assert r4["theoretical_oom_guard_required"] is True
    assert r4["theoretical_oom_guard_triggered"] is True
    assert "Global attention is forbidden" in r4["global_guard_error"]

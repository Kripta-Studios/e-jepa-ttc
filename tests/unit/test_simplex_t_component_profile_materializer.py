"""Profile publication depends on complete semantic QA, not merely report files."""

import json
import runpy
import sys
import uuid
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.unit_qa_evidence import UNIT_QA_FILES


@pytest.mark.parametrize("mode", ["complete", "resource", "invalid", "missing"])
def test_profile_requires_semantic_verification(monkeypatch, mode):
    work = Path(__file__).resolve().parents[2]
    root = work / "artifacts/simplex_t/T0" / ("profile_fixture_" + uuid.uuid4().hex)
    for folder in ("unit", "static", "h16"):
        (root / folder).mkdir(parents=True)
    for path in [
        *(root / "unit" / name for name in UNIT_QA_FILES),
        *(
            root / "static" / name
            for name in ("COMPARISON.json", "BASELINE_RUFF.json", "CURRENT_RUFF.json")
        ),
        root / "types.json",
        root / "powershell.json",
    ]:
        path.write_text("{}", encoding="utf-8")
    if mode != "missing":
        (root / "h16/QA.json").write_text("{}", encoding="utf-8")
    base = root / "base.json"
    base.write_text(
        json.dumps(
            dict(
                schema="simplex_t_component_evidence_profile_v1",
                ack_sha256="a" * 64,
                historical_replay={},
                coherent_replay={},
                resume={},
                unit_qa="STALE_MUST_NOT_BE_INHERITED",
            )
        ),
        encoding="utf-8",
    )
    output = root / "final.json"
    argv = [
        "profile",
        "--local-paths",
        str(root / "local.json"),
        "--base-profile",
        str(base),
        "--base-profile-sha256",
        sha256(base),
        "--unit-root",
        str(root / "unit"),
        "--static-root",
        str(root / "static"),
        "--h16-root",
        str(root / "h16"),
        "--types-report",
        str(root / "types.json"),
        "--powershell-report",
        str(root / "powershell.json"),
        "--output",
        str(output),
        "--other-reserved-bytes",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module = runpy.run_path(str(work / "scripts/materialize_simplex_t_component_profile.py"))
    scope = module["main"].__globals__
    scope["admitted"] = lambda _: {
        "has_headroom": True,
        "written_volume_free_bytes": [100_000_000_000],
    }
    calls = []

    def verify(local, candidate, digest, **flags):
        calls.append(candidate)
        assert sha256(candidate) == digest
        assert all(
            flags[key] is True
            for key in (
                "require_h16",
                "require_unit_qa",
                "require_static_qa",
                "require_types",
                "require_powershell",
            )
        )
        data = json.loads(candidate.read_text("utf-8"))
        assert isinstance(data["unit_qa"], dict)
        assert set(data["unit_qa"]["pins"]) == UNIT_QA_FILES
        if mode == "resource":
            raise InterruptedError("RESOURCE_PAUSE")
        if mode == "invalid":
            raise ValueError("invalid real evidence")
        return {}

    scope["verify_component_profile"] = verify
    if mode == "invalid":
        with pytest.raises(ValueError, match="invalid real evidence"):
            module["main"]()
    else:
        assert module["main"]() == {"complete": 0, "resource": 3, "missing": 2}[mode]
    assert output.exists() == (mode == "complete")
    assert len(calls) == (0 if mode == "missing" else 1)
    if mode == "complete":
        monkeypatch.setattr(sys, "argv", [*argv, "--verify-only"])
        assert module["main"]() == 0
        assert calls[-1] == output
    elif mode != "missing":
        assert calls[0].is_file()  # Failed candidate evidence is retained.

"""Freeze CLI wiring is not evidence of real scientific prerequisite completion."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "resource",
        "qa_failure",
        "changed_preparation",
        "verify_missing",
        "verify_existing",
        "verify_mismatch",
    ],
)
def test_freeze_cli_requires_admission_before_publication(tmp_path, monkeypatch, mode):
    source = Path(__file__).resolve().parents[2] / "scripts/freeze_simplex_t_campaign.py"
    spec = importlib.util.spec_from_file_location("freeze_cli", source)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    prepared = tmp_path / "preparation.json"
    prepared.write_text(json.dumps({"contract": {"fixture": True}}))
    pin = dict(
        category="normalizers",
        root="work",
        relative_path=prepared.name,
        sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
    )
    output = tmp_path / "artifacts/freeze.json"
    launch = tmp_path / "launch.json"
    launch.write_text(
        json.dumps(
            dict(
                schema="simplex_t_freeze_launch_v1",
                local_paths="local.json",
                source_configuration="source.json",
                source_configuration_sha256="b" * 64,
                evidence_profile="qa.json",
                evidence_profile_sha256="c" * 64,
                roots={"work": str(tmp_path)},
                files=[pin],
                preparation=pin,
                technical_ledger=pin,
                code_commit="d" * 40,
                output=str(output),
            )
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--launch",
            str(launch),
            "--launch-sha256",
            hashlib.sha256(launch.read_bytes()).hexdigest(),
            "--other-reserved-bytes",
            "0",
            "--own-reserved-bytes",
            "8388608",
        ],
    )
    monkeypatch.setattr(
        entry,
        "admitted",
        lambda _: {
            "has_headroom": mode != "resource",
            "written_volume_free_bytes": [90_000_000_000],
        },
    )
    admissions = []

    def admission(record, **kwargs):
        assert record == {"files": [pin], "source_contract": {"fixture": True}}
        assert kwargs["resource_ok"]()
        admissions.append(record)
        if mode == "qa_failure":
            raise ValueError("real QA missing")

    def publish(destination, **kwargs):
        assert not mode.startswith("verify")
        kwargs["validate_prerequisites"]()
        kwargs["validate_prerequisites"]()
        destination.parent.mkdir()
        destination.write_text("{}")

    def read(path, **kwargs):
        assert path == output
        assert kwargs["expected_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        kwargs["validate_prerequisites"]()
        return dict(
            files=[pin],
            preparation=pin,
            technical_ledger=pin,
            code_commit=("e" if mode == "verify_mismatch" else "d") * 40,
        )

    monkeypatch.setattr(entry, "validate_scientific_admission", admission)
    monkeypatch.setattr(entry, "publish_scientific_freeze", publish)
    monkeypatch.setattr(entry, "read_scientific_freeze", read)
    if mode.startswith("verify"):
        sys.argv.append("--verify-only")
        if mode != "verify_missing":
            output.parent.mkdir()
            output.write_text("preserve this existing fixture")
    if mode == "changed_preparation":
        prepared.write_text("{}")
    if mode in {"qa_failure", "changed_preparation", "verify_mismatch"}:
        with pytest.raises(ValueError):
            entry.main()
    else:
        assert entry.main() == {"resource": 3, "verify_missing": 10}.get(mode, 0)
    assert output.exists() == (mode in {"valid", "verify_existing", "verify_mismatch"})
    assert len(admissions) == {
        "valid": 2,
        "qa_failure": 1,
        "verify_existing": 1,
        "verify_mismatch": 1,
    }.get(mode, 0)
    if mode in {"verify_existing", "verify_mismatch"}:
        assert output.read_text() == "preserve this existing fixture"

"""Real ZIP composition with stubbed scientific authority, never scientific completion."""

import hashlib
import json
import zipfile

import pytest

from e_jepa_ttc.simplex_t import delivery_assembly as assembly
from e_jepa_ttc.simplex_t import delivery_verification as verification
from e_jepa_ttc.simplex_t.bundle_creation import BundleMember


@pytest.mark.parametrize(
    "fault",
    ["none", "payload", "authority", "resource", "reservation", "negative", "bool", "path"],
)
def test_delivery_transport_preserves_documents_and_checks_inputs(tmp_path, monkeypatch, fault):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    pin = BundleMember(source, hashlib.sha256(source.read_bytes()).hexdigest(), 2)
    original = {
        f"postprocessing/{name}.json": pin
        for name in (
            "SCIENTIFIC_GRAPH_COVERAGE",
            "CAMPAIGN_ACCOUNTING",
        )
    }
    if fault in {"negative", "bool"}:
        original["extra.bin"] = BundleMember(
            source, pin.sha256, -100_000_000 if fault == "negative" else True
        )
    if fault == "path":
        original["../extra.bin"] = pin
    calls = []

    def bind():
        calls.append(True)
        if fault == "authority" and len(calls) > 1:
            return {}
        return original

    monkeypatch.setattr(
        assembly,
        "render_delivery_documents",
        lambda *a, **kw: (
            "# Fixture only\n",
            {"holdout_authorized": False},
        ),
    )
    if fault == "payload":
        source.write_text("changed")
    output = tmp_path / "artifacts/delivery"
    arguments = dict(
        work_root=tmp_path,
        analysis_commit="a" * 40,
        bind_verified_members=bind,
        resource_ok=lambda: fault != "resource",
        reserved_output_bytes=1 if fault == "reservation" else 100_000_000,
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            assembly.assemble_delivery(output, **arguments)
        assert not list(output.glob("*.zip"))
        assert not (output / "DELIVERY.json").exists()
        if fault != "authority":
            assert not output.exists()
        return
    result = assembly.assemble_delivery(output, **arguments)
    archive = output / "E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_aaaaaaaaaaaa.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["bundle"]["sha256"]
    assert result["campaign_complete"] is False
    assert len(calls) == 3
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == 5
        inventory = json.loads(bundle.read("CONTENT_MANIFEST.json"))
        assert set(inventory["members"]) == set(bundle.namelist()) - {"CONTENT_MANIFEST.json"}
        for name, item in inventory["members"].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == item["sha256"]
    assert archive.with_suffix(".zip.sha256").read_text().startswith(result["bundle"]["sha256"])
    with pytest.raises(ValueError, match="new companion"):
        assembly.assemble_delivery(output, **arguments)


def test_delivery_metadata_limit_retains_partial_documents_without_archive(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"{}")
    pin = BundleMember(source, hashlib.sha256(b"{}").hexdigest(), 2)
    original = {
        f"postprocessing/{name}.json": pin
        for name in ("SCIENTIFIC_GRAPH_COVERAGE", "CAMPAIGN_ACCOUNTING")
    }
    monkeypatch.setattr(
        assembly,
        "render_delivery_documents",
        lambda *a, **kw: ("x" * 16_777_216, {"fixture": True}),
    )
    output = tmp_path / "artifacts/delivery"
    with pytest.raises(ValueError, match="reserved metadata bound"):
        assembly.assemble_delivery(
            output,
            work_root=tmp_path,
            analysis_commit="a" * 40,
            bind_verified_members=lambda: original,
            resource_ok=lambda: True,
            reserved_output_bytes=100_000_000,
        )
    assert (output / "CODEX_SIMPLEX_T_FINAL_REPORT.md").stat().st_size == 16_777_216
    assert not (output / "NEXT_DECISION_SIMPLEX_T.json").exists()
    assert not list(output.glob("*.zip*"))
    assert not (output / "DELIVERY.json").exists()


@pytest.mark.parametrize(
    "fault", ["none", "report", "manifest", "sidecar", "receipt", "zip", "authority"]
)
def test_delivery_read_only_verification_rebinds_scientific_evidence(tmp_path, monkeypatch, fault):
    source = tmp_path / "source.json"
    source.write_bytes(b"{}")
    pin = BundleMember(source, hashlib.sha256(b"{}").hexdigest(), 2)
    original = {
        f"postprocessing/{name}.json": pin
        for name in ("SCIENTIFIC_GRAPH_COVERAGE", "CAMPAIGN_ACCOUNTING")
    }
    for module in (assembly, verification):
        monkeypatch.setattr(module, "render_delivery_documents", lambda *a, **kw: ("fixture", {}))
    output = tmp_path / "artifacts/delivery"
    arguments = dict(
        work_root=tmp_path,
        analysis_commit="a" * 40,
        bind_verified_members=lambda: original,
        resource_ok=lambda: True,
    )
    assembly.assemble_delivery(output, **arguments, reserved_output_bytes=100_000_000)
    archive = output / "E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_aaaaaaaaaaaa.zip"
    targets = {
        "report": output / "CODEX_SIMPLEX_T_FINAL_REPORT.md",
        "manifest": output / "CONTENT_MANIFEST.json",
        "sidecar": archive.with_suffix(".zip.sha256"),
        "receipt": output / "DELIVERY.json",
        "zip": archive,
    }
    if fault in targets:
        targets[fault].write_bytes(b"{}")
    calls = []

    def bind():
        calls.append(True)
        return {} if fault == "authority" and len(calls) > 1 else original

    arguments["bind_verified_members"] = bind
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    if fault == "none":
        result = verification.verify_delivery(output, **arguments)
        assert result["files_written"] == result["optimizer_updates_executed"] == 0
        assert len(calls) == 2
        assert result["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    else:
        with pytest.raises((ValueError, zipfile.BadZipFile)):
            verification.verify_delivery(output, **arguments)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()} == before

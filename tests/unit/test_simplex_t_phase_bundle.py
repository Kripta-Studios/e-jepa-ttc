"""Actual byte pinning; graph authority simulated, never scientific fit evidence."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.phase_bundle import phase_bundle_members
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


@pytest.mark.parametrize(
    "fault", ["none", "missing_fit", "changed_payload", "path", "pause", "graph"]
)
def test_registered_phase_members(tmp_path, fault):
    flags = dict(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    keys = [fit_key(spec) for spec in registered_graph(**flags) if spec.stage == "T2"]
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_bytes(b"fixture endpoint seal; not a scientific checkpoint")
    endpoint_hash = hashlib.sha256(endpoints.read_bytes()).hexdigest()
    root = tmp_path / "publication"
    root.mkdir()
    fits = {}
    for key in keys:
        path = root / key / "predictions.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(key.encode())
        fits[key] = dict(
            path=f"{key}/predictions.parquet", sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    if fault == "missing_fit":
        fits.pop(keys[-1])
    elif fault == "path":
        fits[keys[0]]["path"] = "../outside.parquet"
    elif fault == "changed_payload":
        (root / keys[0] / "predictions.parquet").write_bytes(b"changed")
    document = dict(
        schema="simplex_t_phase_predictions_v1",
        status="PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS",
        contract=dict(
            stage="T2", availability=flags, freeze_sha256="a" * 64, manifest_sha256=endpoint_hash
        ),
        fits=fits,
    )
    manifest = root / "PREDICTIONS.json"
    manifest.write_text(json.dumps(document))
    publication = CanonicalPublication(
        manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        endpoints,
        endpoint_hash,
        tmp_path,
        flags,
    )
    calls = 0

    def verify():
        nonlocal calls
        calls += 1
        return dict(
            freeze_sha256="a" * 64,
            phase_fit_counts={"T2": len(keys) + int(fault == "graph" and calls > 1)},
        )

    kwargs = dict(
        freeze_sha256="a" * 64, verify_completed_graph=verify, resource_ok=lambda: fault != "pause"
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            phase_bundle_members({"T2": publication}, **kwargs)
        return
    members = phase_bundle_members({"T2": publication}, **kwargs)
    assert len(members) == len(keys) + 2
    for name, member in members.items():
        assert name.startswith("publications/T2/")
        assert hashlib.sha256(member.path.read_bytes()).hexdigest() == member.sha256
        assert member.path.stat().st_size == member.bytes
    assert calls == 2

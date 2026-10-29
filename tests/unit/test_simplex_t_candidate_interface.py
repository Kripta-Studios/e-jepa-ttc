"""Future interface cannot open a holdout or select exploration/replicate winners."""

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import candidate_interface as module
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("mode", ["ok", "missing", "pause", "source"])
def test_canonical_seed7_interface(tmp_path, monkeypatch, latent, mode):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic fixture")
    flags = dict(
        d1=False,
        density=False,
        t3=True,
        latent=latent,
        replicate_scalar=True,
        replicate_latent=latent,
    )
    graph = registered_graph(**flags)
    record = {
        "source_contract": {"availability": flags},
        "canonical_scalar": "TPR-D0-H8-C160",
        "canonical_latent": "LATENT-D0-H8-C160" if latent else None,
        "source_identities": {fit_key(spec): {"inner_oof": "a" * 64} for spec in graph},
        "code_commit": "b" * 40,
        "files": [],
    }
    monkeypatch.setattr(module, "read_scientific_freeze", lambda *args, **kwargs: record)

    def validated(*args, stage, **kwargs):
        return graph, {
            fit_key(spec): {
                "resolved_checkpoint": seal,
                "checkpoint_sha256": sha256(seal),
                "train_source_sha256": "0" * 64 if mode == "source" else "a" * 64,
            }
            for spec in graph
            if spec.stage == stage
        }

    monkeypatch.setattr(module, "validated_phase", validated)
    binding = CanonicalPublication(seal, sha256(seal), seal, sha256(seal), tmp_path, flags)
    phases = {"T2": binding, **({"T4": binding} if latent else {})}
    if mode == "missing":
        phases.pop("T2")
    output = tmp_path / "TEMPORAL_CANDIDATE_FREEZE.json"
    kwargs = dict(
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        phases=phases,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: mode != "pause",
    )
    if mode == "ok":
        result = module.publish_candidate_interface(output, **kwargs)
        assert set(result["candidates"]) == {
            "TPR-D0-H8-C160",
            *(["LATENT-D0-H8-C160"] if latent else []),
        }
        assert all(len(rows) == 3 for rows in result["candidates"].values())
        assert all(
            row["fit"]["seed"] == 7 for rows in result["candidates"].values() for row in rows
        )
        assert not result["holdout_execution_authorized"]
        assert not result["fold_endpoint_aggregation_authorized"]
    else:
        with pytest.raises((ValueError, InterruptedError)):
            module.publish_candidate_interface(output, **kwargs)
        assert not output.exists()

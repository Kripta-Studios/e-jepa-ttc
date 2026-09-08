"""Manifest binding fixtures, not scientific publication or completion evidence."""

from copy import deepcopy

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.delivery_launch_materialization import delivery_launch


@pytest.mark.parametrize("failure", [None, "journal", "identity", "missing", "no_resources"])
def test_delivery_binding_preserves_shared_journal_and_refuses_missing_inputs(tmp_path, failure):
    execution, publication = tmp_path / "execution", tmp_path / "publication"
    execution.mkdir()
    publication.mkdir()
    base = dict(
        stage="T2",
        roots={"work": str(tmp_path)},
        execution=str(execution),
        publication=str(publication),
        availability={},
        local_paths="local.json",
        source_configuration="source.json",
        source_configuration_sha256="a" * 64,
        evidence_profile="qa.json",
        evidence_profile_sha256="b" * 64,
        freeze="freeze.json",
        freeze_sha256="c" * 64,
    )
    templates = {stage: dict(deepcopy(base), stage=stage) for stage in ("T2", "T4")}
    for stage in templates:
        (execution / f"{stage}_ENDPOINTS.json").write_text("{}")
        (publication / f"{stage}_PREDICTIONS.json").write_text("{}")
    accounting = {}
    for name, path in (
        ("journal", execution / "PHYSICAL_WORK.json"),
        ("ledger", tmp_path / "ledger.json"),
        ("reconciliation", tmp_path / "reconciliation.json"),
    ):
        path.write_text("{}")
        accounting[name], accounting[name + "_sha256"] = str(path), sha256(path)
    resources = [{"fixture": "semantic resource validation belongs to worker"}]
    if failure == "journal":
        templates["T4"]["execution"] = str(tmp_path / "another_execution")
    elif failure == "identity":
        templates["T4"]["freeze_sha256"] = "d" * 64
    elif failure == "missing":
        (publication / "T4_PREDICTIONS.json").unlink()
    elif failure == "no_resources":
        resources = []
    kwargs = dict(
        base=base,
        stage_templates=templates,
        accounting=accounting,
        resource_attempts=resources,
        analysis_commit="e" * 40,
        attempt_root=tmp_path / "artifacts" / "T6",
    )
    if failure:
        with pytest.raises((ValueError, FileNotFoundError)):
            delivery_launch(**kwargs)
    else:
        result = delivery_launch(**kwargs)
        assert result["schema"] == "simplex_t_postprocessing_launch_v3"
        assert set(result["publications"]) == {"T2", "T4"}
        assert result["accounting"] == accounting
        assert result["delivery"]["resource_attempts"] == resources
        assert not (tmp_path / "artifacts").exists()

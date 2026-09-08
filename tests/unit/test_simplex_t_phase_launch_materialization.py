"""Metadata fixtures only: no scientific freeze or optimizer is authorized."""

import pytest

from e_jepa_ttc.simplex_t.phase_launch_materialization import technical_phase_launch


def fixture(tmp_path):
    files = []
    launch = dict(
        schema="simplex_t_freeze_launch_v1",
        code_commit="a" * 40,
        roots={"work": str(tmp_path)},
        local_paths="local.json",
        output="freeze.json",
    )
    for field, category in (("source_configuration", "config"), ("evidence_profile", "qa")):
        path = tmp_path / (field + ".json")
        path.write_text("{}")
        launch[field] = str(path)
        launch[field + "_sha256"] = "b" * 64
        files.append(dict(root="work", relative_path=path.name, sha256="b" * 64, category=category))
    frozen = dict(
        schema="simplex_t_scientific_freeze_v1",
        status="FROZEN_INPUTS_STAGE_GATES_REQUIRED",
        code_commit="a" * 40,
        holdout_authorized=False,
        files=files,
        source_contract=dict(
            availability=dict(
                d1=True,
                density=True,
                t3=True,
                latent=True,
                replicate_scalar=True,
                replicate_latent=True,
            )
        ),
    )
    return launch, frozen


@pytest.mark.parametrize("stage", ["T2", "T4"])
def test_exact_paths_and_no_practical_gate_claim(tmp_path, stage):
    launch, frozen = fixture(tmp_path)
    output = tmp_path / "artifacts" / "scientific"
    result = technical_phase_launch(
        stage=stage,
        freeze_launch=launch,
        frozen=frozen,
        freeze_sha256="c" * 64,
        campaign_root=output,
    )
    assert result["execution"] == str(output / "execution")
    assert result["publication"] == str(output / "publication")
    assert result["publications"] == {}
    assert result["availability"] == dict(
        d1=True, density=True, t3=False, latent=True, replicate_scalar=False, replicate_latent=False
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "change", ["T3", "T5", "holdout", "latent", "d1", "density", "pin", "commit", "outside"]
)
def test_rejects_unresolved_or_changed_scope(tmp_path, change):
    launch, frozen = fixture(tmp_path)
    stage, output = "T4", tmp_path / "artifacts" / "scientific"
    if change in {"T3", "T5"}:
        stage = change
    elif change == "holdout":
        frozen["holdout_authorized"] = True
    elif change in {"latent", "d1", "density"}:
        frozen["source_contract"]["availability"][change] = False
    elif change == "pin":
        frozen["files"][0]["sha256"] = "d" * 64
    elif change == "commit":
        frozen["code_commit"] = "e" * 40
    else:
        output = tmp_path.parent / "outside"
    with pytest.raises(ValueError):
        technical_phase_launch(
            stage=stage,
            freeze_launch=launch,
            frozen=frozen,
            freeze_sha256="c" * 64,
            campaign_root=output,
        )

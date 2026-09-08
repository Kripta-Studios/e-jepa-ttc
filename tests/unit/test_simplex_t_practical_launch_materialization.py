"""Practical metadata control flow with synthetic gate callbacks; zero updates."""

from copy import deepcopy

import pytest

from e_jepa_ttc.simplex_t.practical_launch_materialization import resolve_practical_launch


def fixture():
    flags = dict(
        d1=True, density=True, latent=True, t3=True, replicate_scalar=True, replicate_latent=True
    )
    base = dict(availability=dict(flags), stage="T2", freeze_sha256="a" * 64)
    return base, flags


@pytest.mark.parametrize(
    "scalar,latent", [(False, False), (True, False), (False, True), (True, True)]
)
def test_replication_families_resolve_independently(tmp_path, scalar, latent):
    base, flags = fixture()
    original = deepcopy(base)
    calls = []

    def gate(probe):
        family = next(iter(probe["publications"]))
        calls.append(family)
        assert (
            sum(probe["availability"][key] for key in ("replicate_scalar", "replicate_latent")) == 1
        )
        if not {"TPR": scalar, "LATENT": latent}[family]:
            raise ValueError(f"PRACTICAL_GATE_NOT_PASSED: {family}")
        return {"status": "PHASE_INCOMPLETE"}

    result, decision = resolve_practical_launch(
        "T5",
        base=base,
        frozen_availability=flags,
        campaign_root=tmp_path,
        publication=lambda family: {"family": family},
        check_gate=gate,
    )
    assert calls == ["TPR", "LATENT"]
    assert base == original
    assert decision["eligible"] == (scalar or latent)
    assert decision["optimizer_updates"] == 0
    if result is not None:
        assert result["availability"]["replicate_scalar"] == scalar
        assert result["availability"]["replicate_latent"] == latent
    else:
        assert not scalar and not latent


@pytest.mark.parametrize(
    "failure",
    ["WAITING_CANONICAL_PUBLICATION", "corrupt checkpoint", "PRACTICAL_GATE_NOT_PASSED: LATENT"],
)
def test_missing_or_wrong_family_error_is_not_a_negative(tmp_path, failure):
    base, flags = fixture()

    def gate(probe):
        raise ValueError(failure)

    with pytest.raises(ValueError, match=failure):
        resolve_practical_launch(
            "T3",
            base=base,
            frozen_availability=flags,
            campaign_root=tmp_path,
            publication=lambda family: {},
            check_gate=gate,
        )


def test_resource_pause_and_existing_probe_are_not_a_negative(tmp_path):
    base, flags = fixture()
    kwargs = dict(
        base=base,
        frozen_availability=flags,
        campaign_root=tmp_path,
        publication=lambda family: {},
        check_gate=lambda probe: {"status": "PAUSED_RESOURCE"},
    )
    with pytest.raises(InterruptedError):
        resolve_practical_launch("T3", **kwargs)
    (tmp_path / "gate_probes/T3_TPR").mkdir(parents=True)
    with pytest.raises(ValueError, match="remain absent"):
        resolve_practical_launch("T3", **kwargs)


def test_frozen_unavailability_does_not_read_predictions(tmp_path):
    base, flags = fixture()
    flags["t3"] = False

    def forbidden(*args):
        raise AssertionError("unavailable branch must not inspect predictions")

    result, decision = resolve_practical_launch(
        "T3",
        base=base,
        frozen_availability=flags,
        campaign_root=tmp_path,
        publication=forbidden,
        check_gate=forbidden,
    )
    assert result is None and not decision["eligible"]
    assert decision["families"][0]["reason"] == "NOT_FROZEN_AVAILABLE"

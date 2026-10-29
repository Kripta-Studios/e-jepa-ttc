"""Average aligned losses, retaining seed variation and avoiding TTC ensembles."""

import json

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import followup_analysis
from e_jepa_ttc.simplex_t.replication_summary import three_seed_losses
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


def frames():
    index = np.arange(8192)
    base = pd.DataFrame(
        dict(
            sample_token=index.astype(str),
            sequence_id=(index % 9).astype(str),
            track_id="t",
            outer_fold=index % 3,
            target_ttc=np.array([-2.0, 1.0, 4.0, 8.0])[index % 4],
        )
    )
    result = {}
    for h, losses in ((1, (10.0, 12.0, 14.0)), (8, (8.0, 11.0, 16.0))):
        name = f"TPR-D0-H{h}-C160"
        for seed, loss in zip((7, 13, 23), losses, strict=True):
            result[name, seed] = base.assign(
                arm=name, seed=seed, loss=loss, prediction_ttc_s=(-1 if seed == 13 else 1) * 2.0
            )
    return result


def test_mean_losses_are_not_ttc_predictions():
    inputs = frames()
    inputs["TPR-D0-H8-C160", 23] = inputs["TPR-D0-H8-C160", 23].iloc[::-1]
    averaged, report = three_seed_losses(inputs, family="TPR", pool="D0")
    assert report["paired_delta"]["per_seed"] == pytest.approx([-2, -1, 2])
    assert report["paired_delta"]["mean"] == pytest.approx(-1 / 3)
    assert report["scores"]["TPR-D0-H1-C160"]["sample_std"] == pytest.approx(2)
    assert "prediction_ttc_s" not in averaged["TPR-D0-H8-C160"]
    assert averaged["TPR-D0-H8-C160"].loss.iloc[0] == pytest.approx(35 / 3)


@pytest.mark.parametrize("mode", ["missing", "extra", "target", "loss", "seed"])
def test_bad_replicates_fail(mode):
    inputs = frames()
    key = ("TPR-D0-H8-C160", 23)
    if mode == "missing":
        del inputs[key]
    elif mode == "extra":
        inputs["TPR-D0-H8-C160", 42] = inputs[key]
    elif mode == "target":
        inputs[key].loc[0, "target_ttc"] = 1.5
    elif mode == "loss":
        inputs[key].loc[0, "loss"] = np.nan
    else:
        inputs[key]["seed"] = 7
    with pytest.raises(ValueError):
        three_seed_losses(inputs, family="TPR", pool="D0")


@pytest.mark.parametrize("family", ["TPR", "LATENT"])
@pytest.mark.parametrize("d1", [False, True])
def test_sealed_three_seed_integration(tmp_path, monkeypatch, family, d1):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic fixture")
    inputs = {}
    for (name, seed), frame in frames().items():
        canonical = name.replace("TPR", family).replace("D0", "D1" if d1 else "D0")
        inputs[canonical, seed] = frame.assign(arm=canonical)

    def load(*args, stage, **kwargs):
        return {key: frame for key, frame in inputs.items() if (key[1] == 7) == (stage != "T5")}

    def uncertainty(averaged, *, reference, output, resource_check):
        resource_check()
        assert all("prediction_ttc_s" not in frame for frame in averaged.values())
        assert reference.endswith("H1-C160")
        output.mkdir()
        (output / "PAIRED_UNCERTAINTY.json").write_text(json.dumps({"fixture": True}))

    monkeypatch.setattr(followup_analysis, "load_sealed_analysis_arms", load)
    monkeypatch.setattr(followup_analysis, "paired_uncertainty", uncertainty)
    flags = dict(d1=d1, density=False, replicate_scalar=True, replicate_latent=True)
    binding = CanonicalPublication(seal, sha256(seal), seal, sha256(seal), tmp_path, flags)
    output = tmp_path / "analysis"
    result = followup_analysis.analyze_three_seed_family(
        binding,
        binding,
        family=family,
        output=output,
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        expected_queries=pd.DataFrame(),
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: True,
    )
    assert result["summary"]["paired_delta"]["mean"] == pytest.approx(-1 / 3)
    saved = pd.read_parquet(output / "MEAN_LOSSES_NOT_TTC_PREDICTIONS.parquet")
    assert len(saved) == 16384 and "prediction_ttc_s" not in saved

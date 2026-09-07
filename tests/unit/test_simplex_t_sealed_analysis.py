"""Full OLD-shaped analysis composition, without any fitted scientific model."""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.simplex_t import sealed_analysis as module
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import FitSpec
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


@pytest.mark.parametrize(
    "mode", ["ok", "target", "loss", "source", "train", "missing", "repeat", "pause"]
)
def test_all_seed_arms_are_bound(tmp_path: Path, monkeypatch, mode: str):
    frozen_file = tmp_path / "freeze.json"
    frozen_file.write_text("synthetic fixture")
    endpoint_file = tmp_path / "endpoints.json"
    endpoint_file.write_text("synthetic fixture")
    flags = dict(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=True,
        replicate_latent=False,
    )
    graph = [FitSpec("T5", "TPR-D0-H8-C160", fold, seed) for seed in (13, 23) for fold in range(3)]
    record = {
        "source_contract": {"availability": flags},
        "source_identities": {
            fit_key(spec): {"inner_oof": "a" * 64, "outer_dev": "b" * 64} for spec in graph
        },
    }
    endpoints = {
        fit_key(spec): {
            "train_source_sha256": "0" * 64 if mode == "train" else "a" * 64,
            "resolved_checkpoint": endpoint_file,
            "checkpoint_sha256": sha256(endpoint_file),
        }
        for spec in graph
    }
    sequence = np.arange(8192) % 9
    cohort = pd.DataFrame(
        {
            "sample_token": [f"q{i:05d}" for i in range(8192)],
            "sequence_id": sequence.astype(str),
            "track_id": "track",
            "outer_fold": sequence % 3,
            "target_ttc": 2.0,
        }
    )
    monkeypatch.setattr(module, "read_scientific_freeze", lambda *args, **kwargs: record)
    monkeypatch.setattr(module, "validated_phase", lambda *args, **kwargs: (graph, endpoints))

    def predictions(*args, **kwargs):
        selected = (
            graph[:-1] if mode == "missing" else graph + graph[:1] if mode == "repeat" else graph
        )
        for spec in selected:
            kwargs["validate_prerequisites"]()
            frame = cohort.loc[cohort.outer_fold == spec.fold].copy()
            frame["source_sha256"] = "0" * 64 if mode == "source" else "b" * 64
            frame["prediction_ttc_s"] = 2.0
            frame["prediction_phase"] = benchmark_phase(frame.prediction_ttc_s.to_numpy())
            frame["loss"] = 1.0 if mode == "loss" else 0.0
            frame["arm"], frame["seed"] = spec.name, spec.seed
            if mode == "target":
                frame["target_ttc"] = 3.0
            yield spec, frame

    monkeypatch.setattr(module, "iter_published_predictions", predictions)
    binding = CanonicalPublication(
        endpoint_file, sha256(endpoint_file), endpoint_file, sha256(endpoint_file), tmp_path, flags
    )
    kwargs = dict(
        stage="T5",
        freeze=frozen_file,
        freeze_sha256=sha256(frozen_file),
        roots={"work": tmp_path},
        expected_queries=cohort,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: mode != "pause",
    )
    if mode == "ok":
        arms = module.load_sealed_analysis_arms(binding, **kwargs)
        assert set(arms) == {("TPR-D0-H8-C160", 13), ("TPR-D0-H8-C160", 23)}
        assert all(len(frame) == 8192 for frame in arms.values())
    else:
        with pytest.raises((ValueError, InterruptedError)):
            module.load_sealed_analysis_arms(binding, **kwargs)


@pytest.mark.parametrize("d1", [False, True])
def test_t2_factorial_publication_keeps_registered_reference(tmp_path: Path, monkeypatch, d1: bool):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic seal")
    sequence = np.arange(8192) % 9
    cohort = pd.DataFrame(
        {
            "sample_token": [f"q{i:05d}" for i in range(8192)],
            "sequence_id": sequence.astype(str),
            "track_id": "track",
            "outer_fold": sequence % 3,
            "target_ttc": 2.0,
        }
    )
    frames = {}
    cohort["target_ttc"] = np.asarray([-2.0, 1.0, 4.0, 8.0])[np.arange(8192) % 4]
    for d, h, c in product((0, 1) if d1 else (0,), (1, 8), (64, 160)):
        name = f"TPR-D{d}-H{h}-C{c}"
        frames[name, 7] = cohort.assign(
            arm=name,
            seed=7,
            loss=float(d + h + c),
            history_count=h,
            history_span_us=(h - 1) * 50000,
            roi_age_us=10000,
            cold_start=h == 1,
            hull_position="inside",
            signed_phase_error=0.001,
            escape_gain=0.0,
            escape_harm=0.0,
            phase_interval_width=0.1,
            wrong_sign=False,
            phase_interval_covers=True,
            phase_support_saturation=False,
            target_outside_support=False,
            finite_ttc_cap=False,
        )
    monkeypatch.setattr(module, "load_sealed_analysis_arms", lambda *args, **kwargs: frames)
    reference = f"TPR-D{int(d1)}-H1-C160"

    def uncertainty(actual, *, reference: str, output: Path, resource_check):
        assert set(actual) == {name for name, _ in frames}
        assert reference == f"TPR-D{int(d1)}-H1-C160"
        resource_check()
        output.mkdir()
        result = {"status": "synthetic_test_only"}
        (output / "PAIRED_UNCERTAINTY.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(module, "paired_uncertainty", uncertainty)
    binding = CanonicalPublication(seal, sha256(seal), seal, sha256(seal), tmp_path, {"d1": d1})
    output = tmp_path / "analysis"
    result = module.analyze_sealed_t2(
        binding,
        output=output,
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        expected_queries=cohort,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: True,
    )
    effects = pd.read_parquet(output / "FACTOR_EFFECTS.parquet")
    assert len(effects) == 8192 and result["reference"] == reference
    assert ("DxHxC" in effects) == d1
    assert result["optimizer_updates"] == 0 and result["holdout_opened"] is False

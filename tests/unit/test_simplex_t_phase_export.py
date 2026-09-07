"""Atomic phase publication with synthetic predictions and real Parquet I/O."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t import phase_export as module
from e_jepa_ttc.simplex_t.factorial_analysis import paired_factor_effects
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.published_predictions import iter_published_predictions
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.fixture
def export_args(tmp_path, monkeypatch):
    availability = dict.fromkeys(
        ("d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"), False
    )
    graph = registered_graph(**availability)
    cohort = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in range(8192)],
            "sequence_id": [f"s{i % 3}" for i in range(8192)],
            "track_id": [f"t{i}" for i in range(8192)],
            "outer_fold": np.arange(8192) % 3,
        }
    )
    released = []
    sources = SimpleNamespace(graph=graph, release=lambda: released.append(True))

    def predictions(*args, **kwargs):
        for spec in graph:
            yield spec, {}, np.zeros((1, 1), np.int64)

    def frame(sources, spec, *args, **kwargs):
        return cohort.loc[cohort.outer_fold == spec.fold].assign(
            prediction_ttc_s=2.0, arm=spec.name, seed=spec.seed, loss=1.0, source_sha256="c" * 64
        )

    monkeypatch.setattr(module, "iter_phase_predictions", predictions)
    monkeypatch.setattr(module, "development_frame", frame)
    args = dict(
        output=tmp_path / "exports",
        sources=sources,
        manifest=tmp_path / "sealed.json",
        checkpoint_root=tmp_path,
        manifest_sha256="a" * 64,
        freeze_sha256="b" * 64,
        stage="T2",
        availability=availability,
        dev_source_hashes={fit_key(spec): "c" * 64 for spec in graph},
        expected_queries=cohort,
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
        resume=False,
    )
    return args, graph, released


def test_complete_phase_and_byte_identical_resume(export_args):
    args, graph, released = export_args
    state = module.export_phase(**args)
    assert state["status"] == "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"
    assert len(state["fits"]) == len(graph)
    assert sum(row["rows"] for row in state["fits"].values()) == 8192 * 8
    assert released == [True]
    args["resume"] = True
    assert module.export_phase(**args) == state
    assert len(released) == 2
    path = args["output"] / "T2_PREDICTIONS.json"
    frames = list(
        iter_published_predictions(
            path,
            manifest_sha256=compute_file_hash(str(path)),
            endpoint_manifest_sha256=args["manifest_sha256"],
            freeze_sha256=args["freeze_sha256"],
            stage="T2",
            availability=args["availability"],
            expected_queries=args["expected_queries"],
            validate_prerequisites=lambda: None,
            resource_ok=lambda: True,
        )
    )
    assert [spec for spec, _ in frames] == graph
    effects = paired_factor_effects(
        pd.concat([frame for _, frame in frames], ignore_index=True),
        args["expected_queries"],
        d1_available=False,
    )
    assert len(effects) == 8192
    assert (effects[["H", "C", "HxC"]].to_numpy() == 0).all()


def test_resource_pause_keeps_only_complete_fit_and_resumes(export_args):
    args, graph, _ = export_args
    admission = iter([True, True, False])
    args["resource_ok"] = lambda: next(admission)
    state = module.export_phase(**args)
    assert state["status"] == "PAUSED_RESOURCE"
    assert list(state["fits"]) == [fit_key(graph[0])]
    assert len(list(args["output"].rglob("*.parquet"))) == 1
    args.update(resume=True, resource_ok=lambda: True)
    assert module.export_phase(**args)["status"] == "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"


def test_corrupt_existing_predictions_are_not_overwritten(export_args):
    args, graph, _ = export_args
    module.export_phase(**args)
    path = args["output"] / fit_key(graph[0]) / "predictions.parquet"
    path.write_bytes(b"corrupted fixture")
    args["resume"] = True
    with pytest.raises(ValueError, match="existing prediction bytes differ"):
        module.export_phase(**args)
    assert path.read_bytes() == b"corrupted fixture"


def test_orphan_complete_publication_recovered_from_recomputed_bytes(export_args):
    args, graph, _ = export_args
    state = module.export_phase(**args)
    del state["fits"][fit_key(graph[0])]
    (args["output"] / "T2_PREDICTIONS.json").write_text(json.dumps(state), encoding="utf-8")
    args["resume"] = True
    assert len(module.export_phase(**args)["fits"]) == len(graph)


def test_truncated_stream_never_claims_complete(export_args, monkeypatch):
    args, _, _ = export_args
    monkeypatch.setattr(module, "iter_phase_predictions", lambda *a, **k: iter(()))
    with pytest.raises(ValueError, match="incomplete inference stream"):
        module.export_phase(**args)
    state = json.loads((args["output"] / "T2_PREDICTIONS.json").read_text())
    assert state["status"] == "PREPARING_PREDICTIONS"


def test_query_mismatch_rejected_before_any_parquet(export_args, monkeypatch):
    args, _, _ = export_args
    monkeypatch.setattr(module, "development_frame", lambda *a, **k: args["expected_queries"])
    with pytest.raises(ValueError, match="query identities differ"):
        module.export_phase(**args)
    assert not list(args["output"].rglob("*.parquet"))


@pytest.mark.parametrize("failure", ["incomplete", "last_file_corrupt"])
def test_analysis_rejects_bad_publication_before_first_table(export_args, monkeypatch, failure):
    args, graph, _ = export_args
    state = module.export_phase(**args)
    path = args["output"] / "T2_PREDICTIONS.json"
    if failure == "incomplete":
        state["status"] = "PAUSED_RESOURCE"
        path.write_text(json.dumps(state), encoding="utf-8")
    else:
        (args["output"] / fit_key(graph[-1]) / "predictions.parquet").write_bytes(b"corrupt")

    def forbidden(*args, **kwargs):
        raise AssertionError("must validate whole publication before reading any table")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    iterator = iter_published_predictions(
        path,
        manifest_sha256=compute_file_hash(str(path)),
        endpoint_manifest_sha256=args["manifest_sha256"],
        freeze_sha256=args["freeze_sha256"],
        stage="T2",
        availability=args["availability"],
        expected_queries=args["expected_queries"],
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
    )
    with pytest.raises(ValueError):
        next(iterator)


@pytest.mark.parametrize("bad", ["availability", "digest"])
def test_analysis_requires_resolved_pins_before_manifest_access(tmp_path, bad):
    iterator = iter_published_predictions(
        tmp_path / "absent.json",
        manifest_sha256="invalid" if bad == "digest" else "a" * 64,
        endpoint_manifest_sha256="b" * 64,
        freeze_sha256="c" * 64,
        stage="T2",
        availability={"d1": 1} if bad == "availability" else {"d1": False},
        expected_queries=pd.DataFrame(),
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
    )
    with pytest.raises(ValueError):
        next(iterator)

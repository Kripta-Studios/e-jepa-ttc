"""Atomic phase publication with synthetic predictions and real Parquet I/O."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.simplex_t import phase_export as module
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
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
            prediction_ttc_s=2.0, arm=spec.name, seed=spec.seed
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

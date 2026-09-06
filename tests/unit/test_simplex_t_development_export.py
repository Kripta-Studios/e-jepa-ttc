"""Development export integration with explicit synthetic lineage stubs."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t import development_export
from e_jepa_ttc.simplex_t.campaign_sources import CompiledFold
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    graph = registered_graph(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    spec = next(s for s in graph if s.name.startswith("SELECTOR") and s.fold == 0)
    tables = tmp_path / "tables"
    tables.mkdir()
    metadata = pd.DataFrame(
        dict(
            sample_token=["q0", "q1"],
            sequence_id=["s", "s"],
            track_id=["t0", "t1"],
            target_ttc=[2.0, 3.0],
        )
    )
    csv = tables / "outer0_outer_dev.csv"
    metadata.to_csv(csv, index=False)
    table = dict(metadata=metadata, reference=dict(metadata_sha256=compute_file_hash(str(csv))))
    monkeypatch.setattr(development_export, "load_current_inputs", lambda *a, **k: table)
    history = np.array([[-1, 0], [1, 2]])
    source = SimpleNamespace(
        identity_sha256="source",
        population=2,
        length=2,
        history=history,
        anchor_us=np.array([100, 100, 200]),
        available_us=np.array([110, 210, 210]),
    )
    experts = np.array([[4.0, 5.0, np.inf], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]], np.float32)
    np.save(tmp_path / "expert_ttc.npy", experts)
    manifest = tmp_path / "COMPILED.json"
    manifest.write_text(
        json.dumps(
            dict(arrays=dict(expert_ttc=compute_file_hash(str(tmp_path / "expert_ttc.npy"))))
        ),
        encoding="utf-8",
    )
    sources = SimpleNamespace(
        graph=graph,
        source=lambda *a: source,
        historical_root=tmp_path,
        ancestry_sha256="fixture",
        allowed_sequences={"s"},
        folds={0: CompiledFold(tmp_path, compute_file_hash(str(manifest)))},
    )
    output = dict(
        point_phase=np.zeros(2),
        relative_cost=np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]),
        raw_location=np.zeros(2),
        raw_residual=np.zeros(2),
        q10=np.full(2, -0.1),
        q90=np.full(2, 0.1),
    )
    return sources, spec, output, history, table, csv


def test_export_keeps_current_cached_experts_identity_and_actual_history(export_fixture):
    sources, spec, output, history, _, _ = export_fixture
    result = development_export.development_frame(
        sources, spec, output, history, expected_source_sha256="source"
    )
    assert result.sample_token.tolist() == ["q0", "q1"]
    assert result.track_id.tolist() == ["t0", "t1"]
    np.testing.assert_array_equal(result.prediction_ttc_s, [np.inf, 9.0])
    np.testing.assert_array_equal(result.history_span_us, [0, 100])
    np.testing.assert_array_equal(result.roi_age_us, [10, 10])
    assert result.prediction_ttc_infinite.tolist() == [True, False]
    assert result.context_semantics.str.contains("NOT_VERIFIED_OBJECT_HISTORY").all()


@pytest.mark.parametrize("corruption", ["source", "history", "metadata", "order", "expert"])
def test_export_rejects_misalignment_or_mutation(export_fixture, corruption):
    sources, spec, output, history, table, csv = export_fixture
    expected = "changed" if corruption == "source" else "source"
    if corruption == "history":
        history = history[::-1]
    elif corruption == "metadata":
        csv.write_text("changed", encoding="utf-8")
    elif corruption == "order":
        table["metadata"] = table["metadata"].iloc[::-1]
    elif corruption == "expert":
        np.save(sources.historical_root / "expert_ttc.npy", np.ones((3, 3)))
    with pytest.raises(ValueError):
        development_export.development_frame(
            sources, spec, output, history, expected_source_sha256=expected
        )

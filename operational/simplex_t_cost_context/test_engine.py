"""Engine/authority unit checks with zero optimizer steps.

The separately invoked technical.py is the only synthetic optimizer probe.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from e_jepa_ttc.simplex_t import training
from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.physical_accounting import audit_physical_work
from e_jepa_ttc.simplex_t.work_budget import WorkBudget

from . import engine
from .model import MaskedSource
from .technical import SyntheticSource, segments


def test_authorized_scientific_queue_and_technical_work_are_disjoint() -> None:
    rows = engine.ids()
    assert len(rows) == len({v["key"] for v in rows}) == 18
    assert sum(v["updates"] for v in rows) == 45000
    assert all(v["seed"] == 7 and v["updates"] == 2500 for v in rows)
    assert [v["arm"] for v in rows[::3]] == list(engine.ARM_ORDER)
    assert sum(v["family"] == "N2" for v in rows) == 12
    assert sum(v["family"] == "N3" for v in rows) == 6
    assert sum(v["authorized_updates"] for v in segments()) == 80
    assert not {v["key"] for v in rows} & {v["key"] for v in segments()}


def test_factory_binding_restored_on_success_and_failure() -> None:
    factory, objective = training.TemporalRefiner, training.training_loss
    with engine.numeric_binding("SET_AGE_C0"):
        model = training.TemporalRefiner(TemporalConfig(feature_count=17, hidden=160))
        assert sum(v.numel() for v in model.parameters()) == 289765
        assert training.training_loss is not objective
    assert training.TemporalRefiner is factory and training.training_loss is objective
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with engine.numeric_binding("FULL_C0"):
            raise RuntimeError("synthetic failure")
    assert training.TemporalRefiner is factory and training.training_loss is objective


def test_exact_fit_argument_recipe_without_running_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_fit(
        source: training.QuerySource,
        config: TemporalConfig,
        output: Path,
        **kwargs: object,
    ) -> dict:
        model = training.TemporalRefiner(config)
        data = source.gather(torch.tensor([0, 1]))
        output_model = model(*data[:4])
        loss = training.training_loss(
            output_model, data[4], data[3], data[5], source.population, selector_only=False
        )
        assert torch.isfinite(loss)
        calls.append(dict(config=config, output=output, **kwargs))
        return {"fake_no_optimizer": True}

    monkeypatch.setattr(training, "fit", fake_fit)
    source = MaskedSource(SyntheticSource(), "FULL_C0")
    actual = engine.fit_arm(
        source, "FULL_C0", tmp_path, seed=7, freeze_sha256="a" * 64, resource_ok=lambda: True
    )
    assert actual["fake_no_optimizer"]
    assert calls[0]["device"] == "cpu" and calls[0]["stop_after"] == 2500
    assert calls[0]["seed"] == 7 and calls[0]["config"] == TemporalConfig(hidden=160)
    with pytest.raises(ValueError):
        engine.fit_arm(
            source, "FULL_C0", tmp_path, seed=13, freeze_sha256="a" * 64, resource_ok=lambda: True
        )


def test_physical_journal_preserves_crash_uncertainty(tmp_path: Path) -> None:
    graph = {engine.ids()[0]["key"]: 2500}
    key = next(iter(graph))
    budget = WorkBudget(tmp_path / "journal.json", graph, technical_reserved=0)
    budget.transition("begin", key, 0)
    budget.transition("checkpoint", key, 100, checkpoint_sha256="a" * 64)
    budget.transition("begin", key, 100)
    state = budget.transition("recover", key, 100, checkpoint_sha256="a" * 64)
    assert state["accounting"]["scientific_saved_updates"] == 100
    assert state["accounting"]["scientific_uncertain_lost_upper"] == 100
    audit = audit_physical_work(
        state, expected_graph=graph, technical_reserved=0, resource_ok=lambda: True
    )
    assert audit["recorded_work_upper"] == 200
    assert audit["optimizer_updates_executed"] == 0


def test_union_guard_and_resource_receipts_use_only_own_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out, n1 = tmp_path / "night", tmp_path / "h16"
    out.mkdir()
    monkeypatch.setattr(engine, "OUT", out)
    monkeypatch.setattr(engine, "EXEC", out / "execution")
    monkeypatch.setattr(engine, "N1", n1)
    monkeypatch.setattr(
        engine,
        "memory",
        lambda: dict(
            available=3 * 1024**3,
            tree_rss=1024**3,
            commit_headroom=2 * 1024**3,
            free_disk=15_000_000_000,
        ),
    )
    engine.atomic_json(
        out / "WINDOW_AUTHORIZATION.json", dict(deadline_utc="2099-01-01T00:00:00+00:00")
    )
    resources = engine.Resources(training=True)
    assert resources()
    engine.atomic_json(out / "TECHNICAL_WORK.json", dict(reserved_updates=201))
    assert not resources()
    reasons = engine.record(out / "execution/RESOURCES.json")["reasons"]
    assert "UNION_LOGICAL_OR_REPLAY_CAP_EXCEEDED" in reasons
    assert not n1.exists()

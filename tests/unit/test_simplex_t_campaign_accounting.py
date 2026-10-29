"""Accounting composition fixtures; no optimizer work or scientific endpoints are produced."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from e_jepa_ttc.simplex_t import campaign_accounting as accounting
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    def save(name, obj):
        path = tmp_path / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    flags = dict.fromkeys(
        ("d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"), False
    )
    required = registered_graph(**flags)
    possible = registered_graph(
        **{**flags, **dict.fromkeys(("t3", "latent", "replicate_scalar", "replicate_latent"), True)}
    )
    digest = "a" * 64
    fits, events, endpoints = {}, [], {}
    for spec in required:
        key = fit_key(spec)
        for update in range(0, 2500, 100):
            events.extend(
                [
                    dict(operation="begin", key=key, completed=update, checkpoint_sha256=None),
                    dict(
                        operation="checkpoint",
                        key=key,
                        completed=update + 100,
                        checkpoint_sha256=digest,
                    ),
                ]
            )
        fits[key] = dict(
            completed=2500, uncertain_lost_upper=0, pending=None, checkpoint_sha256=digest
        )
        endpoints[key] = dict(checkpoint_sha256=digest, resolved_checkpoint=tmp_path / key)
    state = dict(
        schema="simplex_t_physical_work_v1",
        graph={fit_key(s): 2500 for s in possible},
        technical_reserved=0,
        fits=fits,
        events=events,
        accounting=dict(
            scientific_saved_updates=2500 * len(required),
            scientific_uncertain_lost_lower=0,
            scientific_uncertain_lost_upper=0,
            technical_reserved_not_execution_claim=0,
            full_graph_physical_work_upper=2500 * len(possible),
        ),
    )
    ledger, ledger_hash = save(
        "ledger.json", dict(schema="simplex_t_technical_budget_v1", reservations={})
    )
    reconciliation, reconciliation_hash = save(
        "reconciliation.json",
        dict(
            status="TECHNICAL_EXECUTION_RECORDS_RECONCILED_WITH_EVIDENCE_CLASSES",
            ledger_sha256=ledger_hash,
            operations=[],
            reserved_updates=0,
            recorded_executed_updates=0,
            historical_noninstrumented_reconciliation_updates=0,
            engine_journal_or_instrumented_receipt_updates=0,
            unknown_or_unmatched_reservations=[],
            scientific_updates_in_these_records=0,
        ),
    )
    journal, journal_hash = save("PHYSICAL_WORK.json", state)
    coverage = dict(
        freeze_sha256="b" * 64,
        phase_fit_counts={"T2": len(required)},
        resolved_availability=flags,
        fits_completed=len(required),
        scientific_updates_completed=2500 * len(required),
    )
    # These two boundaries have their own real manifest/endpoint suites. Here the
    # actual physical-event replay and technical-receipt readers remain unmocked.
    monkeypatch.setattr(accounting, "validated_phase", lambda *a, **kw: (required, endpoints))
    binding = SimpleNamespace(
        checkpoint_root=tmp_path,
        endpoints=tmp_path / "endpoints.json",
        endpoints_sha256="c" * 64,
        availability=flags,
    )
    kwargs = dict(
        journal_sha256=journal_hash,
        freeze_sha256="b" * 64,
        phases={"T2": binding},
        reconciliation=reconciliation,
        reconciliation_sha256=reconciliation_hash,
        ledger=ledger,
        ledger_sha256=ledger_hash,
        work_root=tmp_path,
        verify_completed_graph=lambda: copy.deepcopy(coverage),
        resource_ok=lambda: True,
    )
    return SimpleNamespace(
        journal=journal,
        state=state,
        endpoints=endpoints,
        coverage=coverage,
        kwargs=kwargs,
        save=save,
        binding=binding,
    )


def test_accounting_composes_registered_coverage_and_real_readers(campaign):
    before = campaign.journal.read_bytes()
    result = accounting.verify_campaign_accounting(campaign.journal, **campaign.kwargs)
    assert result["scientific_fits_completed"] == 24
    assert result["recorded_total_updates_lower"] == 60000
    assert result["recorded_total_updates_upper"] == 60000
    assert result["optimizer_updates_executed"] == 0
    assert result["campaign_complete"] is False
    assert campaign.journal.read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "freeze",
        "phase",
        "count",
        "saved",
        "events",
        "endpoint",
        "root",
        "pause",
        "graph_changes",
        "journal_changes",
        "technical_hash",
    ],
)
def test_accounting_rejects_unbound_or_changed_evidence(campaign, fault):
    if fault == "hash":
        campaign.kwargs["journal_sha256"] = "0" * 64
    elif fault == "freeze":
        campaign.coverage["freeze_sha256"] = "0" * 64
    elif fault == "phase":
        campaign.kwargs["phases"] = {}
    elif fault == "count":
        campaign.coverage["fits_completed"] -= 1
    elif fault == "saved":
        campaign.coverage["scientific_updates_completed"] -= 1
    elif fault == "events":
        campaign.state["events"].pop()
        _, campaign.kwargs["journal_sha256"] = campaign.save("PHYSICAL_WORK.json", campaign.state)
    elif fault == "endpoint":
        campaign.endpoints.pop(next(iter(campaign.endpoints)))
    elif fault == "root":
        campaign.binding.checkpoint_root = campaign.journal.parent.parent
    elif fault == "pause":
        campaign.kwargs["resource_ok"] = lambda: False
    elif fault == "technical_hash":
        campaign.kwargs["ledger_sha256"] = "0" * 64
    else:
        calls = 0

        def changing():
            nonlocal calls
            calls += 1
            result = copy.deepcopy(campaign.coverage)
            if calls == 2:
                if fault == "graph_changes":
                    result["fits_completed"] -= 1
                else:
                    campaign.journal.write_bytes(b"{}")
            return result

        campaign.kwargs["verify_completed_graph"] = changing
    with pytest.raises((ValueError, InterruptedError)):
        accounting.verify_campaign_accounting(campaign.journal, **campaign.kwargs)


@pytest.mark.parametrize("changed_state", [False, True])
def test_status_only_checkpoint_resave_requires_exact_training_state(
    campaign, monkeypatch, changed_state
):
    key = next(iter(campaign.endpoints))
    checkpoint = dict(weights=[1, 2], optimizer={"step": 2500}, status="COMPLETE")
    campaign.state["fits"][key]["checkpoint_state_sha256"] = accounting.state_digest(
        {k: v for k, v in checkpoint.items() if k != "status"}
    )
    _, campaign.kwargs["journal_sha256"] = campaign.save("PHYSICAL_WORK.json", campaign.state)
    campaign.endpoints[key]["checkpoint_sha256"] = "d" * 64
    if changed_state:
        checkpoint["optimizer"]["step"] = 2499
    monkeypatch.setattr(accounting, "load_checkpoint", lambda path: copy.deepcopy(checkpoint))
    if changed_state:
        with pytest.raises(ValueError, match="training states differ"):
            accounting.verify_campaign_accounting(campaign.journal, **campaign.kwargs)
    else:
        result = accounting.verify_campaign_accounting(campaign.journal, **campaign.kwargs)
        assert result["checkpoint_training_identity_bound"] is True

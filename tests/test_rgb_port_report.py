"""Zero-update tests for evidence-only RGB-PORT reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from operational.rgb_port.report import EXPECTED_CONTRASTS, FIT_IDS, build_report


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(path: Path) -> None:
    _write(
        path,
        {
            "schema": "rgb_port_execution_v1",
            "tasks": [
                {
                    "id": fit_id,
                    "command": [
                        "python",
                        "-m",
                        "fit",
                        "--manifest-sha",
                        "sha256://manifest.json",
                    ],
                }
                for fit_id in FIT_IDS
            ],
        },
    )


def _metric(mid: float) -> dict[str, Any]:
    return {
        "group_macro_bucket_MiD": mid,
        "coverage": 1.0,
        "formula_admitted": True,
        "diagnostics": {
            "rte_pct": 2.0,
            "crucial_mae_s": 0.1,
            "sign_error_rate": 0.01,
            "p90_ae_s": 0.2,
            "p95_ae_s": 0.3,
        },
    }


def _complete_run(run: Path) -> None:
    _write(run / "SOURCE_FREEZE.json", {"schema": "freeze", "status": "FROZEN"})
    _write(run / "SPLIT_MANIFEST.json", {"grouping_level": "sequence_proxy"})
    _write(run / "TECHNICAL_JOURNAL.json", {"completed": 7, "pending_upper": 0})
    events: dict[str, Any] = {
        "technical": {"category": "technical", "charged_updates": 7, "fit_id": FIT_IDS[0]}
    }
    for fit_id in FIT_IDS:
        fit = run / "fits" / fit_id
        checkpoint = fit / "checkpoint_last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(fit_id.encode())
        _write(
            fit / "CHECKPOINT_RECEIPT.json",
            {
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "completed_updates": 3,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "identity_sha256": "a" * 64,
            },
        )
        _write(
            fit / "UPDATE_JOURNAL.json",
            {
                "completed_updates": 3,
                "durable_updates": 3,
                "recovery_upper": 1,
                "pending_update_upper": 0,
            },
        )
        _write(fit / "RUN_PROVENANCE.json", {"fit_id": fit_id, "status": "COMPLETE"})
        _write(
            fit / "ENDPOINT_FREEZE.json",
            {
                "schema": "rgb_port_fit_endpoint_freeze_v1",
                "fit_id": fit_id,
                "files": {
                    name: _sha(fit / name)
                    for name in (
                        "checkpoint_last.pt",
                        "CHECKPOINT_RECEIPT.json",
                        "UPDATE_JOURNAL.json",
                        "RUN_PROVENANCE.json",
                    )
                },
            },
        )
        _write(fit / "curve_000003.json", {"update": 3, "loss": 0.25})
        events[f"science-{fit_id}"] = {
            "category": "scientific",
            "charged_updates": 3,
            "fit_id": fit_id,
        }
        events[f"recovery-{fit_id}"] = {
            "category": "recovery",
            "charged_updates": 1,
            "fit_id": fit_id,
        }
    _write(run / "ACCOUNTING.json", {"events": events})
    scores = {
        fit_id: {"native": _metric(float(index + 1)), "common_cap60": _metric(float(index + 2))}
        for index, fit_id in enumerate(FIT_IDS[-6:])
    }
    gates = {
        contrast: {
            "passed": index % 2 == 0,
            "checks": {"coverage_equal": True},
            "bootstrap": {"status": "COMPLETE"},
        }
        for index, contrast in enumerate(EXPECTED_CONTRASTS)
    }
    _write(
        run / "evaluation" / "V_RESULTS.json",
        {
            "schema": "rgb_port_v_campaign_evaluation_v1",
            "status": "COMPLETE",
            "scores": scores,
            "screening_gates": gates,
        },
    )
    _write(
        run / "profiles" / "ROUTE_COSTS.json",
        {
            "schema": "rgb_port_route_costs_v1",
            "status": "COMPLETE",
            "profiles": {"FULL_OWN_EVENT": {"median_ms": 1.2, "p95_ms": 1.5}},
        },
    )
    _write(run / "transfer" / "dev32_score" / "RESULT.json", {"status": "COMPLETE"})
    _write(
        run / "transfer" / "fcwd" / "FCWD_RGB_STATUS.json",
        {"status": "BLOCKED_EXTERNAL", "reason": "RGB_TO_EVENT_CALIBRATION_MISSING"},
    )
    _write(run / "NEXT_DECISION.json", {"owner": "queue", "decision": "preserve"})


def test_report_complete_uses_evidence_and_preserves_queue_decision(tmp_path: Path) -> None:
    run, output, config = tmp_path / "run", tmp_path / "report", tmp_path / "execution.json"
    _config(config)
    _complete_run(run)
    decision_before = (run / "NEXT_DECISION.json").read_bytes()

    receipt = build_report(run=run, config_path=config, output=output)

    assert receipt["status"] == "COMPLETE"
    assert receipt["optimizer_updates"] == 0
    assert receipt["optional_branches"]["FCWD_RGB_STATUS"] == "BLOCKED_EXTERNAL"
    assert receipt["accounting"]["scientific_durable"] == 36
    assert receipt["accounting"]["technical_durable"] == 7
    assert receipt["accounting"]["recovery_durable_upper"] == 12
    assert receipt["accounting"]["ledger_matches_journals"] is True
    assert (run / "NEXT_DECISION.json").read_bytes() == decision_before
    curves = sorted((output / "curves").glob("*.json"))
    assert [path.stem for path in curves] == sorted(FIT_IDS)
    assert all(json.loads(path.read_text(encoding="utf-8"))["source_count"] == 1 for path in curves)
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "nativo" in report and "común ±60 s" in report
    assert all(contrast in report for contrast in EXPECTED_CONTRASTS)
    assert "no representa contacto físico calibrado" in report
    card = (output / "model_cards" / "RGB_E_MODEL_CARD.md").read_text(encoding="utf-8")
    assert "predicción E_CTX" in card and "BLOCKED_EXTERNAL" in card
    manifest = json.loads((output / "HASH_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["outputs"]) == 15
    assert all((output / item["path"]).is_file() for item in manifest["outputs"])


def test_report_incomplete_is_not_a_negative_result(tmp_path: Path) -> None:
    run, output, config = tmp_path / "run", tmp_path / "report", tmp_path / "execution.json"
    run.mkdir()
    _config(config)

    receipt = build_report(run=run, config_path=config, output=output)

    assert receipt["status"] == "INCOMPLETE"
    assert receipt["scientific_v_analysis_available"] is False
    assert set(receipt["curve_artifacts"]) == set(FIT_IDS)
    assert all(state["status"] == "MISSING" for state in receipt["endpoint_states"].values())
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "no se interpreta como resultado negativo" in report
    assert "INCOMPLETA" in report
    assert "operational.rgb_port.run resume" in report
    assert "sha256://" not in report

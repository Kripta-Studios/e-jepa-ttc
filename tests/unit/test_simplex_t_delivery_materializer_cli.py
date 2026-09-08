"""Real CLI metadata composition with labelled fixtures; not T6 admission."""

import json
import runpy
import sys
import uuid
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def test_delivery_metadata_cli_is_idempotent_and_retains_unknown_attempts(monkeypatch):
    work = Path(__file__).resolve().parents[2]
    root = work / "artifacts/simplex_t/T0" / ("delivery_cli_fixture_" + uuid.uuid4().hex)
    logs = root / "logs"
    for name in ("launches", "execution", "publication", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return sha256(path)

    freeze = root / "freeze.json"
    freeze_hash = write(freeze, {"source_contract": {"availability": {"latent": False}}})
    base = dict(
        schema="simplex_t_frozen_phase_launch_v2",
        stage="T2",
        roots={"work": str(work)},
        execution=str(root / "execution"),
        publication=str(root / "publication"),
        availability={},
        local_paths="fixture",
        source_configuration="fixture",
        source_configuration_sha256="a" * 64,
        evidence_profile="fixture",
        evidence_profile_sha256="b" * 64,
        freeze=str(freeze),
        freeze_sha256=freeze_hash,
        resource_receipt=str(root / "unused.json"),
    )
    write(root / "launches/T2.json", base)
    for stage in ("T3", "T5"):
        write(
            root / "launches" / f"{stage}.decision.json",
            dict(
                schema="simplex_t_practical_launch_decision_v1",
                stage=stage,
                freeze_sha256=freeze_hash,
                eligible=False,
            ),
        )
    for path in (
        root / "execution/T2_ENDPOINTS.json",
        root / "execution/PHYSICAL_WORK.json",
        root / "publication/T2_PREDICTIONS.json",
    ):
        write(path, {"synthetic_metadata_not_verified_science": True})
    reconciliation = root / "reconciliation.json"
    reconciliation_hash = write(reconciliation, {"synthetic": True})
    for index in (0, 1):
        receipt = root / f"resource{index}.json"
        if index == 0:
            write(receipt, {"synthetic_receipt_not_semantically_verified": True})
        launch = root / f"attempt{index}.json"
        digest = write(launch, dict(base, resource_receipt=str(receipt)))
        write(
            logs / f"attempt{index}.run.log.report.json",
            dict(schema="simplex_t_phase_attempt_v1", launch=str(launch), launch_sha256=digest),
        )
    module = runpy.run_path(str(work / "scripts/materialize_simplex_t_delivery.py"))
    monkeypatch.setitem(
        module["main"].__globals__,
        "admitted",
        lambda paths: {"has_headroom": True, "written_volume_free_bytes": [100_000_000_000]},
    )
    args = [
        "materialize",
        "--campaign-root",
        str(root),
        "--run-root",
        str(logs),
        "--reconciliation",
        str(reconciliation),
        "--reconciliation-sha256",
        reconciliation_hash,
        "--analysis-commit",
        "c" * 40,
        "--other-reserved-bytes",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", [*args, "--verify-only"])
    assert module["main"]() == 10
    assert not (root / "launches/T6.json").exists()
    monkeypatch.setattr(sys, "argv", args)
    assert module["main"]() == 0
    initial = sha256(root / "launches/T6.json")
    assert module["main"]() == 0
    assert sha256(root / "launches/T6.json") == initial
    discovery = json.loads((root / "launches/T6_resource_discovery.json").read_text("utf-8"))
    assert discovery["observed_terminal_attempts"] == 1
    assert len(discovery["missing_terminal_receipts"]) == 1
    write(root / "publication/T2_PREDICTIONS.json", {"changed": True})
    with pytest.raises(ValueError, match="differs"):
        module["main"]()

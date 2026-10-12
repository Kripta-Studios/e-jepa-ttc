"""Frozen H-selected primary candidates versus the full historical event system on V."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, sha, write
from operational.rgb_port.infer_experts import _parents

from .bootstrap import compare as bootstrap
from .common import ROOT, RUN, config, owner
from .data import role_rows
from .evaluate import freeze_primary

REFERENCE = "E_CTX_MATCHED"
DIRECTORY = RUN / "system_compare"


def register() -> dict:
    """Bind the existing gates and historical comparator without opening outcome values."""
    folder = owner() / "predictions/V"
    receipt = read(folder / "HEAD_PREDICTIONS.json")
    manifest = read(owner() / "V_MANIFEST.json")
    artifact = receipt["artifacts"][REFERENCE]
    source = folder / f"{REFERENCE}.csv"
    parents = _parents(owner(), ["E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED", REFERENCE])
    normalizer = sha(owner() / "features/H/E_NORMALIZER.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["population"] != 16432
        or artifact["rows"] != 16432
        or not artifact["native_unfiltered"]
        or sha(source) != artifact["csv_sha256"]
        or receipt["manifest_v_sha256"] != sha(owner() / "V_MANIFEST.json")
        or receipt["rows_sha256"] != manifest["rows_sha256"]
        or sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]
        or receipt["endpoint_bindings"][REFERENCE]["checkpoint_sha256"] != parents[REFERENCE]
        or receipt["endpoint_bindings"][REFERENCE]["normalizer_sha256"] != normalizer
    ):
        raise ValueError("Historical full-system comparator is not admitted")
    record = {
        "reference": REFERENCE,
        "reference_predictions_sha256": artifact["csv_sha256"],
        "reference_receipt_sha256": sha(folder / "HEAD_PREDICTIONS.json"),
        "historical_parents": parents,
        "normalizer_sha256": normalizer,
        "V_manifest_sha256": sha(owner() / "V_MANIFEST.json"),
        "population": 16432,
        "candidate_scope": "only previously frozen PRIMARY_SELECTION_FINAL.ranked; maximum two",
        "gates": "unchanged operational.range_transfer.analyze.gate; no numerical relaxation",
        "cost": "require completed paired raw E2E audit; report p50/p95/p99/mean ratios",
        "cost_threshold": "no new numerical threshold introduced by this report",
        "interpretation": (
            "seed7 internal development contrast; replication and interpretation required"
        ),
        "context_difference": "primary P2 versus historical E_CTX experts with native history",
        "system_promotion": False,
        "official_comparison": False,
        "sources": {
            name: sha(ROOT / name)
            for name in (
                "operational/joint_measurement/system_compare.py",
                "operational/range_transfer/analyze.py",
                "operational/rgb_port/evaluate.py",
                "operational/joint_measurement/bootstrap.py",
            )
        },
    }
    freeze(DIRECTORY / "REGISTRATION.json", record)
    return record


def require_cohort(frame: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Reject omissions, duplicates, changed groups or changed targets; keep invalid predictions."""
    if len(frame) != len(rows) or not frame.sample_token.is_unique:
        raise ValueError("Comparison requires every query exactly once")
    for name in ("sample_token", "group_id", "target_ttc"):
        if not np.array_equal(frame[name].to_numpy(), rows[name].to_numpy()):
            raise ValueError(f"Comparison cohort differs: {name}")


def compare() -> dict:
    """Produce the original metric gates and paired E2E evidence without V-based reselection."""
    endpoints = freeze_primary()["endpoints"]
    selection_path = RUN / "PRIMARY_SELECTION_FINAL.json"
    selection = read(selection_path)
    names = selection["ranked"]
    if (
        len(names) > 2
        or len(set(names)) != len(names)
        or any(name not in config()["fits"] for name in names)
    ):
        raise ValueError("Comparison must use the previously admitted H candidate list")
    contract = register()
    if not names:
        result = {
            "status": "NO_H_ADMITTED_PRIMARY_CANDIDATE",
            "V_outcomes_opened": False,
            "selection_sha256": sha(selection_path),
            "system_promotion": False,
            "next": "continue observation-driven authorized P/H research",
        }
        freeze(DIRECTORY / "RESULT.json", result)
        return result
    cost_root = RUN / "cost/primary_raw_e2e"
    cost = read(cost_root / "COMPLETE.json")
    cost_binding = read(cost_root / "REGISTRATION.json")
    if (
        cost["status"] != "COMPLETE"
        or not cost["E2E"]
        or cost["binding"] != digest(cost_binding)
        or cost_binding["endpoints"] != endpoints
        or cost_binding["primary_selection_sha256"] != sha(selection_path)
        or cost_binding["historical_parents"] != contract["historical_parents"]
    ):
        raise ValueError("System comparison requires matching complete raw E2E evidence")
    rows = role_rows("V")
    reference_path = owner() / "predictions/V" / f"{REFERENCE}.csv"
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    require_cohort(reference, rows)
    reference = reference[["sample_token", "group_id", "target_ttc", "prediction_ttc"]].copy()
    baseline_dir = DIRECTORY / "predictions/V" / REFERENCE
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_file = baseline_dir / "predictions.parquet"
    reference.to_parquet(baseline_file, index=False)
    reference_metrics = metrics(
        reference.target_ttc.to_numpy(),
        reference.prediction_ttc.to_numpy(),
        reference.group_id.to_numpy(),
    )
    write(baseline_dir / "METRICS.json", reference_metrics)
    freeze(
        baseline_dir / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "rows": len(reference),
            "sha256": sha(baseline_file),
            "source_csv_sha256": contract["reference_predictions_sha256"],
            "scope": "unchanged historical full-system predictions, same original V targets",
        },
    )
    results = {}
    for name in names:
        directory = RUN / "predictions/V" / name
        receipt = read(directory / "COMPLETE.json")
        identity = read(directory / "IDENTITY.json")
        path = directory / "predictions.parquet"
        if (
            receipt["status"] != "COMPLETE"
            or sha(path) != receipt["sha256"]
            or identity["endpoint"] != endpoints[name]
            or identity["role"] != "V"
            or receipt["binding"] != digest(identity)
        ):
            raise ValueError("Candidate V prediction binding changed")
        frame = pd.read_parquet(path)
        require_cohort(frame, rows)
        outcome = metrics(
            frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id.to_numpy()
        )
        ratios = {
            key: cost["results"][name][key] / cost["results"][REFERENCE][key]
            for key in ("p50_seconds", "p95_seconds", "p99_seconds", "mean_seconds")
        }
        interval = bootstrap(baseline_file, path, DIRECTORY / "bootstrap" / name)
        results[name] = {
            "gates": gate(reference_metrics, outcome),
            "metrics": outcome,
            "raw_E2E_cost_ratios": ratios,
            "group_bootstrap": interval,
            "predictions_sha256": receipt["sha256"],
        }
    result = {
        "status": "INTERNAL_SYSTEM_COMPARISON_COMPLETE",
        "reference": REFERENCE,
        "reference_metrics": reference_metrics,
        "candidates": results,
        "selection_sha256": sha(selection_path),
        "cost_sha256": sha(cost_root / "COMPLETE.json"),
        "contract_sha256": sha(DIRECTORY / "REGISTRATION.json"),
        "system_promotion": False,
        "official_comparison": False,
        "scope": "fixed H-selected seed7 endpoints; exposed V development, no new recipe selected",
        "next": (
            "interpret full-system effects and continue only evidence-enabled authorized branches"
        ),
    }
    freeze(DIRECTORY / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else compare())

"""Compare frozen H-selected RGB candidates with historical complete systems on V."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, sha

from .bootstrap import compare as bootstrap
from .common import ROOT, RUN, owner
from .data import role_rows
from .historical_multimodal import admit
from .rgb_cost import finalize_selection
from .rgb_evaluate import freeze_endpoints
from .rgb_train import REGISTRATION
from .system_compare import require_cohort

DIRECTORY = RUN / "rgb/system_compare"
REFERENCES = ("R_CTX", "F_TRUE")


def register() -> dict:
    """Freeze comparison scope without reading V predictions or changing selection."""
    admission_path = RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json"
    admission = read(admission_path)
    if admission["status"] != "ADMITTED_AS_HISTORICAL_COMPARATORS":
        raise ValueError("Historical multimodal comparators require prior admission")
    record = {
        "historical_admission_sha256": sha(admission_path),
        "references": {name: admission["heads"][name] for name in REFERENCES},
        "candidate_scope": "only rgb/SELECTION_FINAL.json ranked, frozen using P/H and cost",
        "population": "complete ordered V; internal exposed development",
        "gates": "unchanged operational.range_transfer.analyze.gate",
        "context": "RGB P2 versus historical expert histories; historical heads trained on H",
        "modalities": {"R_CTX": "RGB reference", "F_TRUE": "RGB+event reference"},
        "cost_scope": "accuracy comparison only; paired full-pipeline cost remains required",
        "system_promotion": False,
        "official_comparison": False,
        "sources": {
            name: sha(ROOT / name)
            for name in (
                "operational/joint_measurement/rgb_system_compare.py",
                "operational/joint_measurement/rgb_cost.py",
                "operational/joint_measurement/rgb_evaluate.py",
                "operational/joint_measurement/historical_multimodal.py",
                "operational/joint_measurement/system_compare.py",
                "operational/joint_measurement/bootstrap.py",
                "operational/range_transfer/analyze.py",
                "operational/rgb_port/evaluate.py",
            )
        },
    }
    freeze(DIRECTORY / "REGISTRATION.json", record)
    return record


def candidate_frame(directory: Path, endpoint: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Require the admitted endpoint, V binding and complete cohort before scoring."""
    identity = read(directory / "IDENTITY.json")
    receipt = read(directory / "COMPLETE.json")
    path = directory / "predictions.parquet"
    if (
        receipt["status"] != "COMPLETE"
        or identity["role"] != "V"
        or identity["endpoint"] != endpoint
        or identity["fit_id"] != directory.name
        or receipt["binding"] != digest(identity)
        or receipt["sha256"] != sha(path)
        or receipt["rows"] != len(rows)
        or identity["registration_sha256"] != sha(REGISTRATION)
        or identity["source"] != sha(ROOT / "operational/joint_measurement/rgb_evaluate.py")
        or identity["scorer_source"] != sha(ROOT / "operational/range_transfer/analyze.py")
        or identity["rgb_model_source"] != sha(ROOT / "operational/joint_measurement/rgb_model.py")
        or identity["role_manifest_sha256"] != sha(owner() / "V_MANIFEST.json")
        or identity["rows_sha256"] != read(owner() / "V_MANIFEST.json")["rows_sha256"]
    ):
        raise ValueError("RGB candidate V prediction identity or coverage changed")
    frame = pd.read_parquet(path)
    require_cohort(frame, rows)
    return frame


def score(frame: pd.DataFrame) -> dict:
    """Use the native scorer, retaining invalid predictions in coverage metrics."""
    return metrics(
        frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id.to_numpy()
    )


def compare() -> dict:
    """Score frozen candidates without V-based ranking, promotion or new inference."""
    contract = register()
    selection = finalize_selection()
    selection_sha = sha(RUN / "rgb/SELECTION_FINAL.json")
    names = selection["ranked"]
    if not names:
        result = {
            "status": "NO_H_ADMITTED_RGB_CANDIDATE",
            "V_outcomes_opened": False,
            "selection_sha256": selection_sha,
            "contract_sha256": sha(DIRECTORY / "REGISTRATION.json"),
            "system_promotion": False,
            "official_comparison": False,
        }
        freeze(DIRECTORY / "RESULT.json", result)
        return result
    endpoints = freeze_endpoints()["endpoints"]
    if len(set(names)) != len(names) or any(
        name not in endpoints or not selection["decisions"][name]["eligible"] for name in names
    ):
        raise ValueError("RGB comparison requires the frozen H-admitted candidate list")
    # Revalidate original weights, normalization, ancestry and prediction hashes
    # once at this durable boundary, before opening any V prediction values.
    admission = admit()
    if any(admission["heads"][name] != contract["references"][name] for name in REFERENCES):
        raise ValueError("Historical comparison endpoints changed")
    rows = role_rows("V")
    candidates = {}
    for name in names:
        directory = RUN / "rgb/predictions/V" / name
        candidates[name] = score(candidate_frame(directory, endpoints[name], rows))
    references, comparisons = {}, {}
    for reference in REFERENCES:
        source = owner() / "predictions/V" / f"{reference}.csv"
        if sha(source) != contract["references"][reference]["V_prediction_sha256"]:
            raise ValueError("Historical prediction bytes changed")
        # Installed pandas supports round_trip; its stubs omit this valid option.
        frame = pd.read_csv(source, float_precision="round_trip")  # pyright: ignore[reportCallIssue, reportArgumentType]
        require_cohort(frame, rows)
        path = DIRECTORY / "predictions/V" / reference / "predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[["sample_token", "group_id", "target_ttc", "prediction_ttc"]].to_parquet(
            path, index=False
        )
        references[reference] = score(frame)
        comparisons[reference] = {}
        for name in names:
            candidate = RUN / "rgb/predictions/V" / name / "predictions.parquet"
            comparisons[reference][name] = {
                "gates": gate(references[reference], candidates[name]),
                "group_bootstrap": bootstrap(
                    path, candidate, DIRECTORY / "bootstrap" / reference / name
                ),
                "predictions_sha256": sha(candidate),
            }
    result = {
        "status": "INTERNAL_ACCURACY_COMPARISON_COMPLETE",
        "references": references,
        "candidates": candidates,
        "comparisons": comparisons,
        "selection_sha256": selection_sha,
        "contract_sha256": sha(DIRECTORY / "REGISTRATION.json"),
        "system_promotion": False,
        "official_comparison": False,
        "candidate_reselection": False,
        "scope": contract["population"],
        "cost_scope": contract["cost_scope"],
        "modalities": contract["modalities"],
    }
    freeze(DIRECTORY / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else compare())

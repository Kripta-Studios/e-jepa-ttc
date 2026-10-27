"""Audit local artifact coverage and materialize the compact final candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from runpod_final_preservation import RECEIPTS, ROOT, publish

DESTINATION = Path("D:/E_JEPA_POD2_RETURN_20261027")
FINAL = "workspace/ejepa_final_runs/final/"
BASE = "PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF"
REQUIRED = (
    "final_fits",
    "final_generated",
    "final_inputs",
    "event_runtime",
    "frozen_code",
    "event_top_metadata",
    "final_runtime",
    "final_delivery",
    "final_diagnostics",
    "test12_probe_code",
    "test12_event_metadata",
    "test12_rgb_metadata",
    "rgb_fold_closure",
    "frozen_project_metadata",
    "final_auxiliary_runtime",
    *(
        f"event_r{rotation}_{kind}"
        for rotation in (2, 3, 4)
        for kind in ("fits", "generation", "support")
    ),
)


def build_index() -> tuple[dict, dict]:
    receipts = []
    for path in RECEIPTS.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "LOCAL_VERIFIED" and value.get("member_sha_verified") is True:
            receipts.append(value)
    index = {}
    groups = {}
    for receipt in sorted(receipts, key=lambda value: value["verified_timestamp"]):
        archive = Path(receipt["local_path"])
        if not archive.is_file() or archive.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Missing or resized local archive: {archive}")
        manifest = json.loads((RECEIPTS / f"{receipt['group']}.manifest.json").read_text())
        groups[receipt["group"]] = receipt
        for name, item in manifest["files"].items():
            index[name] = dict(
                item,
                archive=str(archive),
                member=name,
                archive_sha256=receipt["sha256"],
                group=receipt["group"],
            )
    return index, groups


def content(index: dict, name: str) -> bytes:
    entry = index[name.lstrip("/")]
    with tarfile.open(entry["archive"], "r:*") as archive:
        data = archive.extractfile(entry["member"]).read()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Local member no longer matches verified manifest: {name}")
    return data


def candidate(index: dict) -> dict:
    output = DESTINATION / "FINAL_CANDIDATE"
    output.mkdir(exist_ok=True)
    seal = json.loads(content(index, FINAL + "MODEL_SEAL.json"))
    wanted = dict(seal["deployment_dependencies"])
    wanted["PHASE_ONLY_EVENT_S7_TRAIN40"] = seal["trained_endpoints"][
        "PHASE_ONLY_EVENT_S7_TRAIN40"
    ]["checkpoint_sha256"]
    mapping = {}
    for label, expected in wanted.items():
        matches = [name for name, item in index.items() if item["sha256"] == expected]
        if not matches:
            raise ValueError(f"Deployment dependency missing locally: {label}")
        name = min(
            matches, key=lambda value: (not value.endswith("checkpoint_last.pt"), len(value))
        )
        target = output / "checkpoints" / f"{label}.pt"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content(index, name))
        mapping[label] = {
            "sha256": expected,
            "local_path": str(target),
            "source_archive": index[name]["archive"],
            "source_member": name,
        }
    for label, expected in seal["normalizers"].items():
        name = next(name for name, item in index.items() if item["sha256"] == expected)
        target = output / f"{label}_normalizer.json"
        target.write_bytes(content(index, name))
        mapping[f"{label}_normalizer"] = {
            "sha256": expected,
            "local_path": str(target),
            "source_archive": index[name]["archive"],
            "source_member": name,
        }
    selected = {
        "MODEL_SEAL.json": FINAL + "MODEL_SEAL.json",
        "PHASE_ONLY_HEAD.ts": FINAL + "PHASE_ONLY_HEAD.ts",
        "PHASE_ONLY_HEAD.json": FINAL + "PHASE_ONLY_HEAD.json",
        "OOF_RECIPE.json": seal["recipe_path"].lstrip("/"),
        "predictions.csv": FINAL + "test12/predictions.csv",
        BASE + ".zip": FINAL + "test12/" + BASE + ".zip",
        BASE + ".json": FINAL + "test12/" + BASE + ".json",
        "PREDICTION_RECEIPT.json": FINAL + "test12/PREDICTION_RECEIPT.json",
    }
    for filename, member in selected.items():
        if member in index:
            (output / filename).write_bytes(content(index, member))
            mapping[filename] = {
                "sha256": index[member]["sha256"],
                "local_path": str(output / filename),
                "source_archive": index[member]["archive"],
                "source_member": member,
            }
    config_names = {
        "SCIENTIFIC_CONFIG.json",
        "GROUP_FOLDS.json",
        "FULL_REFIT_PLAN.json",
        "TRAINING_DAG.json",
        "DATA_AND_PARENT_BINDINGS.json",
        "BUDGET_PROFILE.json",
    }
    for name, entry in index.items():
        if Path(name).name in config_names:
            target = output / "configs" / (Path(name).stem + "_" + entry["sha256"][:16] + ".json")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content(index, name))
    for name in index:
        if name.startswith("workspace/ejepa_final_diagnostics/") and name.endswith(
            (".md", "REPORT.json")
        ):
            target = (
                output
                / "diagnostic_summaries"
                / name.removeprefix("workspace/ejepa_final_diagnostics/")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content(index, name))
    publish(
        output / "PORTABLE_PATHS.json",
        {
            "schema": "ejepa_final_candidate_local_paths_v1",
            "immutable_original_seal": True,
            "components": mapping,
            "full_artifact_index": str(DESTINATION / "RESTORE_INDEX.json"),
            "export_scope": "TorchScript PHASE_ONLY head, not sensor-to-TTC pipeline",
        },
    )
    return {"directory": str(output), "components": mapping, "checkpoint_count": len(wanted)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()
    if not args.candidate_only:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/runpod_final_preservation.py"),
                "--groups",
                "final_auxiliary_runtime",
                "--refresh",
            ],
            check=True,
        )
    index, groups = build_index()
    publish(
        RECEIPTS / "RESTORE_INDEX.json",
        {"schema": "pod2_local_restore_index_v1", "timestamp": time.time(), "members": index},
    )
    shutil.copy2(RECEIPTS / "RESTORE_INDEX.json", DESTINATION / "RESTORE_INDEX.json")
    result = candidate(index)
    if args.candidate_only:
        publish(
            RECEIPTS / "CANDIDATE_MATERIALIZED.json",
            dict(result, status="MODEL_MATERIALIZED_PREDICTIONS_MAY_BE_PENDING"),
        )
        print(
            json.dumps(
                {"candidate": result["directory"], "checkpoints": result["checkpoint_count"]}
            )
        )
        return
    missing = sorted(set(REQUIRED) - set(groups))
    if missing:
        raise ValueError(f"Preservation groups still missing: {missing}")
    event_sequences = sorted(name for name in groups if name.startswith("test12_events_"))
    prediction_sequences = sorted(name for name in groups if name.startswith("test12_predictions_"))
    tensors = [
        name
        for name in index
        if name.startswith("workspace/ejepa_test12_events/tensors/") and name.endswith(".npz")
    ]
    fragments = [
        name
        for name in index
        if name.startswith(FINAL + "test12/fragments/") and name.endswith(".json")
    ]
    if (
        len(event_sequences) != 12
        or len(prediction_sequences) != 12
        or len(tensors) != 5805
        or len(fragments) != 5805
    ):
        raise ValueError("Local Test12 sequence/tensor/prediction coverage is incomplete")
    package = json.loads(content(index, FINAL + "test12/" + BASE + ".json"))
    prediction = json.loads(content(index, FINAL + "test12/PREDICTION_RECEIPT.json"))
    if package.get("status") != "PACKAGED_NOT_SUBMITTED" or prediction.get("status") != "COMPLETE":
        raise ValueError("Final package or predictions are not admitted")
    payload = content(index, FINAL + "test12/" + BASE + ".zip")
    if hashlib.sha256(payload).hexdigest() != package["zip_sha256"]:
        raise ValueError("Package ZIP detached from receipt")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("Unexpected submission ZIP layout")
        submission = json.loads(archive.read("submission.json"))
    if len(submission["results"]) != 6762:
        raise ValueError("Official submission coverage is not 6,762 tokens")
    stopped = json.loads(content(index, "workspace/ejepa_final_diagnostics/COLLECTOR_STOPPED.json"))
    if stopped.get("status") != "STOPPED":
        raise ValueError("Final diagnostic collector was not durably stopped")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_final_diagnostics_report.py"),
            "--archive-root",
            str(DESTINATION),
            "--output",
            str(DESTINATION / "FINAL_CANDIDATE/diagnostics"),
            "--mode",
            "final",
            "--source",
            str(ROOT / "artifacts/runpod_migration_20261026"),
        ],
        check=True,
    )
    tooling = DESTINATION / "FINAL_CANDIDATE/tools"
    tooling.mkdir(exist_ok=True)
    for path in [
        *ROOT.glob("scripts/runpod_final_*.py"),
        *ROOT.glob("scripts/runpod_test12_*.py"),
        *ROOT.glob("tests/unit/test_runpod_final_*.py"),
        *ROOT.glob("tests/unit/test_runpod_test12_*.py"),
    ]:
        shutil.copy2(path, tooling / path.name)
    evidence = DESTINATION / "FINAL_CANDIDATE/local_evidence"
    evidence.mkdir(exist_ok=True)
    source = ROOT / "artifacts/final_diagnostics_20261027"
    if source.is_dir():
        shutil.copytree(source, evidence / "final_diagnostics_20261027", dirs_exist_ok=True)
    operations = evidence / "final_operations"
    operations.mkdir(exist_ok=True)
    for path in RECEIPTS.parent.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".txt", ".log"}:
            shutil.copy2(path, operations / path.name)
    for path in (ROOT / "artifacts/runpod_migration_20261026").glob("test12*20261027*.txt"):
        shutil.copy2(path, operations / path.name)
    receipt_copy = DESTINATION / "PRESERVATION_RECEIPTS"
    receipt_copy.mkdir(exist_ok=True)
    for path in RECEIPTS.glob("*.json"):
        if path.name != "RESTORE_INDEX.json":
            shutil.copy2(path, receipt_copy / path.name)
    (DESTINATION / "FINAL_CANDIDATE/README.md").write_text(
        "# Final TRAIN40 candidate\n\n"
        "The nine checkpoint files and normalizers are exact SHA-verified copies. "
        "MODEL_SEAL.json is preserved unchanged; PORTABLE_PATHS.json maps its dependencies "
        "to local files. PHASE_ONLY_HEAD.ts exports the fusion head only, not the complete "
        "sensor-to-TTC pipeline.\n\n"
        "The submission ZIP is packaged, not submitted to CodaBench. Test12 ground truth "
        "is private; these predictions do not establish official accuracy or SOTA.\n\n"
        "All generated tensors, OOF features, optimizer states, predictions, diagnostics "
        "and runtime evidence remain in the adjacent verified archives. RESTORE_INDEX.json "
        "in the parent folder maps original paths to archive members and SHA-256 values. "
        "PRESERVATION_RECEIPTS contains archive and member verification receipts. "
        "Public raw media are represented by pinned source revisions and SHA receipts.\n",
        encoding="utf-8",
    )
    local_extras = {}
    for path in (DESTINATION / "FINAL_CANDIDATE").rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                local_extras[str(path)] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                }
    closure = {
        "schema": "pod2_local_final_preservation_v1",
        "status": "LOCAL_FINAL_PRESERVATION_COMPLETE",
        "timestamp": time.time(),
        "groups": {
            k: {"archive": v["local_path"], "sha256": v["sha256"]} for k, v in groups.items()
        },
        "unique_members": len(index),
        "event_sequences": 12,
        "event_tensors": len(tensors),
        "prediction_sequences": 12,
        "canonical_prediction_fragments": len(fragments),
        "official_predictions": 6762,
        "candidate": result["directory"],
        "deployment_checkpoints": 9,
        "raw_public_media_policy": (
            "Pinned revision and SHA receipts retained; public media may be downloaded again, "
            "no model computation required."
        ),
        "scientific_tests_or_inference_reexecuted_by_preservation": False,
        "all_archive_members_previously_sha_verified": True,
        "official_submission_executed": False,
        "materialized_candidate_and_report_files": local_extras,
    }
    publish(RECEIPTS / "LOCAL_FINAL_PRESERVATION_COMPLETE.json", closure)
    publish(DESTINATION / "LOCAL_FINAL_PRESERVATION_COMPLETE.json", closure)
    print(json.dumps({k: v for k, v in closure.items() if k != "groups"}))


if __name__ == "__main__":
    main()

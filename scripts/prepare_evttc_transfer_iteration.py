"""Prepare portable first-iteration inputs from actual local evidence, without inference.

Paths are explicit arguments. Generated scopes retain unknown ancestry/acquisitions.
This does not authorize training or independently certify the source receipts.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import h5py
import numpy as np

from operational.evttc_transfer_audit.common import file_sha, read_document, write_receipt
from operational.evttc_transfer_audit.lineage import MODEL_IDS


def prepare(args: argparse.Namespace) -> None:
    """Inventory actual resources and construct explicit diagnostic-only inputs."""
    output = args.output
    manifest = read_document(args.manifest)
    identities = read_document(args.identities)
    if identities["status"] != "verified":
        raise ValueError("Identity audit must pass before preparing target clock maps")
    matches = {r["competition_sequence"]: r["matches"][0] for r in identities["sequences"]}
    exclusions = sorted({m["sequence_id"] for m in matches.values()})
    write_receipt(output / "exclusions.json", {"excluded_recordings": exclusions})
    records = []
    sources = {}
    label_sources = {}
    bytes_total = 0
    for entry in manifest["sequences"]:
        sid = entry["sequence_id"]
        path = args.data_root / entry["local_path"] / entry["event_hdf5"]
        labels = path.parent / entry["ttc_csv"]
        sources[sid] = path
        label_sources[sid] = labels
        size = path.stat().st_size
        bytes_total += size
        with h5py.File(path, "r") as handle:
            ts = handle["blackflys/right/ts"]
            rgb = handle["blackflys/right/data"]
            if not isinstance(ts, h5py.Dataset) or not isinstance(rgb, h5py.Dataset):
                raise ValueError("Expected RGB clock and frame datasets")
            first_frame = np.asarray(rgb[0])
            sensor = {
                "rgb_frames": len(ts),
                "rgb_clock_start_us": int(ts[0]),
                "rgb_clock_end_us": int(ts[-1]),
                "rgb_shape": list(first_frame.shape),
                "first_frame_sha256": hashlib.sha256(first_frame.tobytes()).hexdigest(),
            }
        records.append(
            {
                "sequence_id": sid,
                "recording_group": sid,
                "acquisition_group": "unknown",
                "aliases": [],
                "role": "excluded" if sid in exclusions else "development_exposed",
                "review_status": "unresolved",
                "relationship_evidence": [],
                "relationship_reason": (
                    "Distinct filenames and nominal split_group do not prove separate acquisitions"
                ),
                "historical_exposure": [
                    "scores_inspected",
                    "targets_inspected",
                    "design_use_unknown",
                ],
                "exposure_evidence": [
                    "README.md: historical 468 labeled queries across 32 sequences"
                ],
                "scenario_family": entry["scenario_family"],
                "speed_bucket": entry["speed_bucket"],
                "target_type": entry["target_type"],
                "hdf5_bytes": size,
                "labels_present": labels.is_file(),
                "labels_sha256": file_sha(labels),
                "identity_evidence_sha256": file_sha(args.identities),
                **sensor,
            }
        )
    inventory = {
        "schema": "evttc_transfer_population_v1",
        "protocol_id": "evttc_transfer_v2",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "records": records,
    }
    write_receipt(output / "population.json", inventory)
    portable = read_document(args.candidate / "PORTABLE_PATHS.json")
    recipe = read_document(args.candidate / "OOF_RECIPE.json")
    seal = read_document(args.candidate / "MODEL_SEAL.json")
    scope_path = args.candidate / "diagnostics/evidence/DEPLOYMENT_TRAIN40_SCOPE.json"
    scope = read_document(scope_path)
    stages = {s["stage_id"]: s for s in recipe["stages"]}
    alias = {"PAIR_E": "PAIR_E_MATCHED"}
    inverse_alias = {v: k for k, v in alias.items()}
    components = []
    for name in MODEL_IDS:
        stage = stages.get(alias.get(name, name), {})
        parents = [
            portable["components"][inverse_alias.get(p, p)]["sha256"]
            for p in stage.get("depends_on", [])
        ]
        producer = scope.get("producers", {}).get(name, {})
        config = producer.get("training_config", {})
        components.append(
            {
                "component_id": name,
                "artifact_sha256": portable["components"][name]["sha256"],
                "parent_artifact_hashes": parents if stage else None,
                "deployment_dependencies": {
                    p: seal["deployment_dependencies"].get(p) for p in stage.get("depends_on", [])
                },
                "training_scope_declared": (
                    "eAP full TRAIN40; OOF training ancestry distinct from deployment graph"
                ),
                "training_manifest_hashes": [recipe["native_train40_rows_sha256"]],
                "training_groups": scope.get("rgb_union", {}).get("group_ids")
                if name.startswith("R_")
                else None,
                "checkpoint_selection_groups": None,
                "architecture_selection_groups": None,
                "normalization_fit_groups": None,
                "ancestry_complete": False,
                "teacher_artifact_hash": config.get(
                    "representation_teacher_cache_artifact_sha256", "unknown"
                ),
                "supervision": config.get(
                    "representation_supervision", "see recipe and native training loss"
                ),
                "stage_declared": {k: v for k, v in stage.items() if k != "depends_on"},
                "evidence_refs": [
                    "MODEL_SEAL.json",
                    "OOF_RECIPE.json",
                    "DEPLOYMENT_TRAIN40_SCOPE.json",
                ],
                "evidence_sha256": {
                    "MODEL_SEAL.json": file_sha(args.candidate / "MODEL_SEAL.json"),
                    "OOF_RECIPE.json": file_sha(args.candidate / "OOF_RECIPE.json"),
                    "DEPLOYMENT_TRAIN40_SCOPE.json": file_sha(scope_path),
                },
                "unknown_reason": (
                    "Historical architecture/checkpoint selection and recursive teacher scopes "
                    "not certified by portable seal"
                ),
            }
        )
    write_receipt(output / "components.json", {"components": components})
    ledger = [
        json.loads(line)
        for line in (args.run / "all_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    queries = []
    predictions: dict[str, list[dict[str, Any]]] = {"PHASE": [], "PHASE_event": [], "PHASE_rgb": []}
    for row in ledger:
        identity = row["identity"]
        sid = identity["sequence_id"]
        qid = f"{sid}:{identity['row_index']}"
        queries.append(
            {
                "query_id": qid,
                "sequence_id": sid,
                "group": sid,
                "query_us": identity["anchor_us"],
                "timestamp_text": identity["timestamp_text"],
            }
        )
        for method, key in (
            ("PHASE", "model_ttc"),
            ("PHASE_event", "event_ttc"),
            ("PHASE_rgb", "rgb_ttc"),
        ):
            predictions[method].append({"query_id": qid, "prediction": row["prediction"][key]})
    labels = {}
    for competition, match in matches.items():
        path = label_sources[match["sequence_id"]]
        with path.open(encoding="utf-8") as stream:
            table = list(csv.reader(stream, delimiter="\t"))
        labels[competition] = [[int(Decimal(r[1]) * 1_000_000), float(r[4])] for r in table]
    write_receipt(output / "queries.json", queries)
    write_receipt(output / "labels.json", labels)
    write_receipt(
        output / "clock_map.json", {sid: m["timestamp_offset_us"] for sid, m in matches.items()}
    )
    for method, rows in predictions.items():
        write_receipt(output / f"predictions_{method}.json", rows)
    manifest_text = (args.package / "MANIFIESTO.md").read_text(encoding="utf-8")
    package_checks = []
    for name, size, checksum in re.findall(
        r"\| \[[^]]+\]\(([^)]+)\) \| (\d+) \| `([0-9a-f]{64})`", manifest_text
    ):
        path = args.package / name
        package_checks.append(
            {"path": name, "match": path.stat().st_size == int(size) and file_sha(path) == checksum}
        )
    with zipfile.ZipFile(args.package_zip) as archive:
        crc = archive.testzip()
    if len(package_checks) != 20 or not all(p["match"] for p in package_checks) or crc:
        raise ValueError("Supplied plan package failed integrity checks")
    preflight = {
        "schema": "evttc_transfer_preflight_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": sys.version,
        "platform": platform.platform(),
        "lockfile_sha256": file_sha(ROOT / "uv.lock"),
        "hdf5_recordings": len(records),
        "hdf5_bytes": bytes_total,
        "small_frame_reads": len(records),
        "disk_free_bytes": shutil.disk_usage(output).free,
        "original_artifacts": {
            name: file_sha(args.run / name)
            for name in ("RUN_FREEZE.json", "results.zip", "all_rows.jsonl")
        },
        "plan_zip_sha256": file_sha(args.package_zip),
        "package_checks": package_checks,
        "zip_crc": "verified",
        "training_performed": False,
        "environment_note": (
            "uv sync --locked --no-editable in .venv-transfer; "
            "corrupted torch cache redownloaded with --no-cache"
        ),
    }
    write_receipt(output / "preflight.json", preflight)
    print(
        json.dumps(
            {
                "records": len(records),
                "excluded": len(exclusions),
                "queries": len(queries),
                "components": len(components),
            }
        )
    )


def main() -> None:
    """Build audit inputs from explicitly selected local resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "identities",
        "data-root",
        "candidate",
        "run",
        "package",
        "package-zip",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

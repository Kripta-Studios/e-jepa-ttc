"""Read-only transfer audits; outputs are new receipts, never signed-run edits."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from operational.evttc_transfer_audit.common import file_sha, read_document, write_receipt
from operational.evttc_transfer_audit.identity import compare_reference, match_sources
from operational.evttc_transfer_audit.lineage import audit_lineage
from operational.evttc_transfer_audit.protocol import validate_protocol
from operational.evttc_transfer_audit.scoring import score_rows
from operational.evttc_transfer_audit.targets import align_targets


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    """Run predeclared exposed queries with passive tracing and an untraced parity control."""
    import torch

    from operational.evttc_competition.geometry import load_calibration
    from operational.evttc_competition.prepare import prepare_query
    from operational.evttc_competition.readers import SensorSequence
    from operational.evttc_competition.runtime import CompetitionRuntime
    from operational.evttc_transfer_audit.diagnostics import trace_query
    from operational.evttc_transfer_audit.provenance import (
        assert_sources_unchanged,
        bind_candidate,
        snapshot_sources,
    )
    from scripts.final_train40_transfer import verify_candidate

    source_hashes = snapshot_sources(ROOT)
    config = read_document(args.diagnostic_config)
    protocol = read_document(args.protocol)
    if config["scope"] != "exposed_diagnostic" or protocol["status"] == "rejected":
        raise ValueError("Diagnosis requires an explicit exposed scope and non-rejected protocol")
    _, candidate_binding = verify_candidate(args.candidate)
    bind_candidate(candidate_binding, protocol)
    queries = read_document(args.query_manifest)
    requested = {(q["sequence_id"], q["query_us"]): q["query_id"] for q in queries}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    runtime = CompetitionRuntime.from_candidate(args.candidate, device=args.device)
    receipts = []
    for sequence in sorted({q["sequence_id"] for q in queries}):
        with SensorSequence(args.data_root, sequence) as reader:
            rows = reader.bbox_rows()
            indices = sorted(
                {len(rows) - 1 if i == "last" else int(i) for i in config["row_indices"]}
            )
            calibration = load_calibration(
                args.data_root / sequence,
                kind="slider_candidate" if sequence.startswith("Slider-") else "road_right",
            )
            for index in indices:
                if index < 0 or index >= len(rows):
                    raise ValueError(
                        f"Predeclared row index outside population: {sequence}:{index}"
                    )
                row = rows[index]
                query_us = row["anchor_us"]
                if (sequence, query_us) not in requested:
                    raise ValueError("Diagnostic selection outside query manifest")
                prepared = prepare_query(reader, rows, row, calibration)
                baseline = runtime.predict(prepared.events, prepared.rgb, query_us)
                traced = trace_query(runtime, prepared.events, prepared.rgb, query_us)
                traced["query_id"] = requested[sequence, query_us]
                traced["input_details"] = prepared.details
                traced["untraced_prediction"] = baseline.model_ttc
                traced["parity"] = traced["prediction"]["model_ttc"] == baseline.model_ttc
                traced["scope"] = config["scope"]
                path = args.output / "traces" / f"{sequence}_{index}.json"
                write_receipt(path, traced)
                receipts.append(
                    {
                        "query_id": traced["query_id"],
                        "trace": str(path.relative_to(args.output)),
                        "sha256": file_sha(path),
                        "parity": traced["parity"],
                    }
                )
        print(f"{sequence}: {len(indices)} traced queries", flush=True)
    assert_sources_unchanged(ROOT, source_hashes)
    _, final_binding = verify_candidate(args.candidate)
    if candidate_binding != final_binding:
        raise ValueError("Candidate changed during diagnosis")
    return {
        "schema": "evttc_transfer_diagnosis_v1",
        "status": "diagnostic_partial" if all(r["parity"] for r in receipts) else "rejected",
        "scope": config["scope"],
        "traces": receipts,
        "labels_read": False,
        "config_sha256": file_sha(args.diagnostic_config),
        "protocol_sha256": file_sha(args.protocol),
        "query_manifest_sha256": file_sha(args.query_manifest),
        "source_sha256": source_hashes,
        "candidate_binding": candidate_binding,
    }


def identity(args: argparse.Namespace) -> dict[str, Any]:
    """Reproduce every reference overlap against all manifest recordings."""
    manifest = read_document(args.manifest)
    sources = {}
    for entry in manifest["sequences"]:
        key = entry["sequence_id"]
        if key in sources:
            raise ValueError(f"Duplicate source ID: {key}")
        sources[key] = args.data_root / entry["local_path"] / entry["event_hdf5"]
        if not sources[key].is_file():
            raise FileNotFoundError(sources[key])
    reference = read_document(args.reference_evidence)
    reference = reference.get("evidence", {}).get("recording_overlap", reference)
    rows = []
    for expected in reference["sequences"]:
        name = expected["competition_sequence"]
        result = match_sources(args.competition_root / name / "data.hdf5", sources)
        result["competition_sequence"] = name
        result["reference_equal"] = compare_reference(result, expected)
        rows.append(result)
        print(f"{name}: {result['status']}, {result['query_frames']} frames", flush=True)
    return {
        "schema": "evttc_transfer_identity_v1",
        "status": "verified"
        if rows and all(row["reference_equal"] for row in rows)
        else "rejected",
        "manifest_sha256": file_sha(args.manifest),
        "reference_evidence_sha256": file_sha(args.reference_evidence),
        "labels_read": False,
        "training_performed": False,
        "sequences": rows,
        "limitations": ["RGB identity only; acquisition independence is not certified."],
    }


def main(argv: list[str] | None = None) -> int:
    """Return 0 completed, 1 contract violation, 2 usage, 3 missing resource."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("identity", help="Reproduce exact RGB overlap without TTC")
    for flag in ("manifest", "data-root", "competition-root", "reference-evidence", "output"):
        command.add_argument(f"--{flag}", type=Path, required=True)
    flags = {
        "lineage": ("candidate", "component-inventory", "exclusions", "output"),
        "protocol": ("inventory", "identities", "lineage", "policy", "output"),
        "targets": ("queries", "labels-manifest", "clock-map", "target-contract", "output"),
        "score": ("predictions", "targets", "policy", "output"),
        "diagnose": (
            "data-root",
            "candidate",
            "query-manifest",
            "protocol",
            "diagnostic-config",
            "output",
        ),
    }
    for name, arguments in flags.items():
        command = commands.add_parser(name, help=f"Generate a scoped {name} receipt")
        for flag in arguments:
            command.add_argument(f"--{flag}", type=Path, required=True)
        if name == "diagnose":
            command.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "identity":
            result = identity(args)
        elif args.command == "lineage":
            result = audit_lineage(
                args.candidate,
                read_document(args.component_inventory),
                read_document(args.exclusions),
            )
        elif args.command == "protocol":
            result = validate_protocol(
                read_document(args.policy),
                read_document(args.inventory),
                read_document(args.identities),
                read_document(args.lineage),
            )
        elif args.command == "targets":
            rows = align_targets(
                read_document(args.queries),
                read_document(args.labels_manifest),
                read_document(args.clock_map),
                read_document(args.target_contract),
            )
            result = {
                "schema": "evttc_transfer_targets_v1",
                "status": "diagnostic_partial",
                "rows": rows,
            }
        elif args.command == "diagnose":
            result = diagnose(args)
        else:
            targets = read_document(args.targets)
            result = score_rows(
                read_document(args.predictions), targets["rows"], read_document(args.policy)
            )
        write_receipt(args.output / "report.json", result)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 1 if result["status"] == "rejected" else 0
    except FileNotFoundError as error:
        print(f"Missing resource: {error}", file=sys.stderr)
        return 3
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Contract violation: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

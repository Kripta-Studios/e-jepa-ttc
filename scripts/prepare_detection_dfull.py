"""Materialize full readable train admission while preserving selection and split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "readiness", "preflight", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_path = args.source / "d4096_manifest.json"
    source = json.loads(source_path.read_text())
    old = json.loads((args.source / "media_admission.json").read_text())
    if digest(args.source / "media_admission.json") != source["media_admission"]["receipt_sha256"]:
        raise ValueError("Prior media admission differs")
    readiness = json.loads((args.readiness / "readiness.json").read_text())
    candidate_path = args.readiness / "dfull_candidate_tokens.json"
    if digest(candidate_path) != readiness["candidate_tokens_sha256"]:
        raise ValueError("Candidate token population differs")
    tokens = json.loads(candidate_path.read_text())["sample_tokens"]
    if len(tokens) != len(set(tokens)) or len(tokens) != readiness["full_train_readable"]["frames"]:
        raise ValueError("Full training cardinality differs")
    ledger_path = args.preflight / "token_status.jsonl"
    summary = json.loads((args.preflight / "summary.json").read_text())
    if digest(ledger_path) != summary["progress_files"]["token_status_sha256"]:
        raise ValueError("Incremental preflight ledger differs")
    final_media = copy.deepcopy(old["final_media"])
    for line in ledger_path.read_text().splitlines():
        row = json.loads(line)
        if row["status"] != "PASS":
            continue
        token = row["sample_token"]
        if token in final_media:
            raise ValueError("Incremental media overlaps the existing admission")
        final_media[token] = {
            key: row[key]
            for key in (
                "role",
                "sequence_id",
                "rgb_shard_path",
                "rgb_member_path",
                "member_size_bytes",
                "member_sha256",
                "image_wh",
            )
        } | {"source_owner": "rgb_full_preflight_v1"}
    full_selection = {t for t, r in old["final_media"].items() if r["role"] == "full_selection"}
    if set(final_media) != set(tokens) | full_selection:
        raise ValueError("Full admission population differs")
    manifest = copy.deepcopy(source)
    manifest.pop("media_admission")
    manifest["roles"]["train"]["sample_tokens"] = tokens
    manifest["sampling"]["train"] = (
        "all readable non-overflow frames from the frozen 24 train sequences"
    )
    manifest["limitations"] = [
        "sequence_id grouping only; cross-sequence acquisition linkage unverified",
        "selection is development-exposed, not final audit",
        "no public test labels read",
        "three invalid unannotated train media excluded explicitly",
    ]
    args.output.mkdir(parents=True)
    raw_path = args.output / "dfull_manifest_before_admission.json"
    write(raw_path, manifest)
    receipt = copy.deepcopy(old)
    receipt.update(
        {
            "source_manifest_sha256": digest(raw_path),
            "source_manifest_canonical_sha256": canonical(manifest),
            "source_full_preflight_summary_sha256": digest(args.preflight / "summary.json"),
            "source_full_preflight_token_ledger_sha256": digest(ledger_path),
            "source_full_preflight_shard_ledger_sha256": digest(
                args.preflight / "shard_status.jsonl"
            ),
            "implementation_sha256": digest(Path(__file__)),
            "excluded": readiness["excluded_media"],
            "replacement": {
                "policy": "exclude unreadable train media; no synthetic replacements",
                "prior_admission_sha256": digest(args.source / "media_admission.json"),
                "prior_manifest_sha256": digest(source_path),
                "readiness_sha256": digest(args.readiness / "readiness.json"),
            },
            "new_train_token_population_sha256": canonical(tokens),
            "final_counts": {
                "train": len(tokens),
                "full_selection": len(full_selection),
                "total": len(final_media),
            },
            "final_media": final_media,
        }
    )
    receipt_path = args.output / "media_admission.json"
    write(receipt_path, receipt)
    manifest["media_admission"] = {
        "receipt_path": "media_admission.json",
        "receipt_sha256": digest(receipt_path),
        "source_manifest_sha256": digest(raw_path),
        "train_token_population_sha256": canonical(tokens),
    }
    path = args.output / "dfull_manifest.json"
    write(path, manifest)
    shutil.copy2(args.source / "population_audit.json", args.output / "population_audit.json")
    write(
        args.output / "BUILD_RECEIPT.json",
        {
            "schema": "phase_dfull_build_v1",
            "train_count": len(tokens),
            "full_selection_count": len(full_selection),
            "manifest_sha256": digest(path),
            "media_admission_sha256": digest(receipt_path),
            "same_selection": manifest["roles"]["selection"] == source["roles"]["selection"],
            "same_split_sha256": manifest["candidate_split_sha256"],
            "implementation_sha256": digest(Path(__file__)),
        },
    )
    print(json.dumps({"train": len(tokens), "manifest_sha256": digest(path)}))


if __name__ == "__main__":
    main()

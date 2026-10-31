"""Combine existing D4096 admission with incremental full-train media inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    """Hash an evidence artifact without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count annotations and sequence-local annotated identities."""
    identities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["trainable_counts"])
        for category, values in row["trainable_identities"].items():
            identities[category].update((row["sequence_id"], value) for value in values)
    return {
        "frames": len(rows),
        "sequences": len({r["sequence_id"] for r in rows}),
        "trainable_annotations": dict(sorted(counts.items())),
        "trainable_sequence_track_identities": {k: len(v) for k, v in sorted(identities.items())},
    }


def main() -> None:
    """Write a candidate token list and readiness evidence; never launch training."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "admission", "frames", "preflight", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    manifest = read(args.manifest)
    admission = read(args.admission)
    if sha(args.admission) != manifest["media_admission"]["receipt_sha256"]:
        raise ValueError("D4096 media admission identity differs")
    if admission["status"] != "PASS":
        raise ValueError("D4096 is not admitted")
    existing = set(manifest["roles"]["train"]["sample_tokens"])
    rows = [json.loads(line) for line in args.frames.read_text().splitlines()]
    train = {r["sample_token"]: r for r in rows if r["role"] == "train"}
    if not existing <= train.keys():
        raise ValueError("Existing training population differs")
    preflight = read(args.preflight / "summary.json")
    ledger = args.preflight / "token_status.jsonl"
    if sha(ledger) != preflight["progress_files"]["token_status_sha256"]:
        raise ValueError("Incremental media ledger differs")
    checks = [json.loads(line) for line in ledger.read_text().splitlines()]
    observed = [r["sample_token"] for r in checks]
    if len(set(observed)) != len(observed) or set(observed) != set(train) - existing:
        raise ValueError("Incremental audit does not cover exactly the additional train population")
    shard_ledger = args.preflight / "shard_status.jsonl"
    if sha(shard_ledger) != preflight["progress_files"]["shard_status_sha256"]:
        raise ValueError("Shard ledger differs")
    for line in shard_ledger.read_text().splitlines():
        row = json.loads(line)
        if row["shard_failure_count"] or not row.get("stable_during_audit"):
            raise ValueError("Unresolved shard-level failure requires a separate audit")
    bad = [r for r in checks if r["status"] != "PASS"]
    accepted = sorted(existing | {r["sample_token"] for r in checks if r["status"] == "PASS"})
    if any(train[token]["trainer_overflow"] for token in accepted):
        raise ValueError("Candidate contains query overflow")
    sources = {
        "manifest": args.manifest,
        "admission": args.admission,
        "frames": args.frames,
        "preflight_summary": args.preflight / "summary.json",
        "token_ledger": ledger,
        "shard_ledger": shard_ledger,
        "implementation": Path(__file__),
    }
    result = {
        "schema": "phase_detection_full_train_readiness_v1",
        "status": "MEDIA_AUDITED_CANDIDATE_NOT_TRAINING_RELEASE",
        "sources": {k: {"path": str(v), "sha256": sha(v)} for k, v in sources.items()},
        "d4096": summarize([train[t] for t in sorted(existing)]),
        "full_train_raw": summarize(list(train.values())),
        "full_train_readable": summarize([train[t] for t in accepted]),
        "excluded_media": bad,
        "candidate_split_sha256": manifest["candidate_split_sha256"],
        "selection_and_calibration_unchanged": True,
        "remaining_gates": [
            "Materialize versioned training manifest/config and admission bindings.",
            "Profile bounded input cache and actual throughput for enlarged population.",
            "Set exposure budget and validation schedule before launching.",
            "Acquisition-level independence is still unverified.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    tokens_path = args.output / "dfull_candidate_tokens.json"
    tokens_path.write_text(
        json.dumps(
            {
                "sample_tokens": accepted,
                "scope": "Readable train candidates only; not a configured training release.",
            },
            indent=2,
        )
        + "\n"
    )
    result["candidate_tokens_sha256"] = sha(tokens_path)
    (args.output / "readiness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"readable_frames": len(accepted), "excluded_frames": len(bad)}))


if __name__ == "__main__":
    main()

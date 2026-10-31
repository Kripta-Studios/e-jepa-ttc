"""Summarize verified R2-rare results and audit the executed sampling calendar."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .compare_detection_r0_r2 import compact, digest, write
else:
    from compare_detection_r0_r2 import compact, digest, write


def read(path: Path) -> Any:
    """Read one UTF-8 JSON evidence artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluation(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Check file identity and recompute every serialized ground-truth hash."""
    if digest(path) != expected_sha:
        raise ValueError(f"Evaluation file differs: {path}")
    payload = read(path)
    expected = payload["metrics_v2"]["population"]["ground_truth_sha256_by_token"]
    if set(expected) != set(payload["ground_truth_v2"]):
        raise ValueError("Evaluation ground-truth population differs")
    for token, frame in payload["ground_truth_v2"].items():
        encoded = json.dumps(
            frame, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != expected[token]:
            raise ValueError(f"Serialized GT differs: {token}")
    result = compact(payload["metrics_v2"])
    del payload
    gc.collect()
    return result, expected


def audit_calendar(
    runs: list[tuple[Path, int, int]], plan: dict[str, Any], tokens: list[str]
) -> dict[str, Any]:
    """Verify all committed updates, excluding replayed uncheckpointed history."""
    all_tokens: list[str] = []
    update_ids: list[int] = []
    source_hashes = {}
    for run, start, end in runs:
        path = run / "metrics.jsonl"
        source_hashes[str(path)] = digest(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        committed = [row for row in rows if start <= row["update"] <= end]
        if [row["update"] for row in committed] != list(range(start, end + 1)):
            raise ValueError(f"Incomplete committed update ledger: {run}")
        for row in committed:
            update = row["update"]
            expected = [tokens[i] for i in plan["indices"][(update - 1) * 4 : update * 4]]
            if row["sample_tokens"] != expected:
                raise ValueError(f"Executed sampler differs at update {update}")
            all_tokens.extend(expected)
            update_ids.append(update)
    if update_ids != list(range(1, 8193)):
        raise ValueError("Combined training history must cover exactly 8192 updates")
    counts = Counter(all_tokens)
    return {
        "status": "PASS",
        "updates": len(update_ids),
        "exposures": len(all_tokens),
        "unique_frames": len(counts),
        "maximum_exposures_per_frame": max(counts.values()),
        "ordered_tokens_sha256": hashlib.sha256(
            json.dumps(all_tokens, separators=(",", ":")).encode()
        ).hexdigest(),
        "ledger_sha256": source_hashes,
        "scope": "Committed lineage; replayed updates after lost checkpoint excluded.",
    }


def main() -> None:
    """Generate machine-readable analysis only after the controller verifies completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "candidate", "baseline", "plan", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    comparison_path = args.campaign / "COMPARISON.json"
    comparison = read(comparison_path)
    if read(args.campaign / "STATUS.json")["status"] != "COMPLETED_VERIFIED":
        raise ValueError("Campaign has not passed completion verification")
    if not comparison["same_population"]:
        raise ValueError("Comparison population differs")
    baseline = read(args.baseline)
    admission = read(args.campaign / f"RESUME_ADMISSION_{args.candidate.name}.json")
    calendar = audit_calendar(
        [
            (args.campaign / "smoke8", 1, 8),
            (args.campaign / "r2_rare_seed7", 9, admission["update"]),
            (args.candidate, admission["update"] + 1, 8192),
        ],
        read(args.plan),
        read(args.manifest)["roles"]["train"]["sample_tokens"],
    )
    result: dict[str, Any] = {
        "schema": "phase_r2_rare_final_analysis_v1",
        "status": "ANALYZING",
        "source_sha256": {
            "comparison": digest(comparison_path),
            "baseline": digest(args.baseline),
            "plan": digest(args.plan),
            "manifest": digest(args.manifest),
            "analysis_script": digest(Path(__file__)),
        },
        "calendar": calendar,
        "acceptance_checks": comparison["acceptance_checks"],
        "screening_pass": comparison["screening_pass"],
        "full": {},
        "monitor": {},
        "scope": "Single seed; exposed development selection, not official eAP or TTC.",
    }
    replay_path = args.campaign / "REPLAY_4096_PARITY.json"
    result["replayed_update_parity"] = read(replay_path)
    result["source_sha256"]["replayed_update_parity"] = digest(replay_path)
    for update in ("4096", "8192"):
        previous = baseline["runs"]["r2"]["full"][update]
        current = comparison["evaluations"][update]
        a, a_hashes = evaluation(Path(previous["path"]), previous["sha256"])
        b, b_hashes = evaluation(Path(current["path"]), current["sha256"])
        if a_hashes != b_hashes:
            raise ValueError(f"Ground-truth contents differ at update {update}")
        result["full"][update] = {"r2": a, "r2_rare": b, "AP_delta": b["AP"] - a["AP"]}
        write(args.output / "analysis.json", result)
    r2_run = Path(baseline["runs"]["r2"]["full"]["8192"]["path"]).parent.parent
    for name, runs in (
        ("r2", [r2_run]),
        ("r2_rare", [args.campaign / "r2_rare_seed7", args.candidate]),
    ):
        by_update = {}
        for run in runs:
            for path in sorted((run / "evaluation").glob("selection_update_*.json")):
                update = int(path.stem.rsplit("_", 1)[1])
                if run.name == "r2_rare_seed7" and update > admission["update"]:
                    continue
                value = read(path)["metrics_v2"]
                by_update[update] = {
                    "update": update,
                    "AP": value["dev_AP101_supported_classes"],
                    "per_class_AP": {
                        k: v["AP101_mean_thresholds"] for k, v in value["per_class"].items()
                    },
                    "sha256": digest(path),
                }
        result["monitor"][name] = [by_update[k] for k in sorted(by_update)]
    result["status"] = "COMPLETE"
    write(args.output / "analysis.json", result)
    print(json.dumps({"screening_pass": result["screening_pass"], "calendar": calendar}))


if __name__ == "__main__":
    main()

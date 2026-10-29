"""Fail closed if distributed public evidence differs from its manifest."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: object, b: object) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    manifest = json.loads((ROOT / "data/public_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Public evidence hash mismatch: {item['path']}")
    registry = json.loads((ROOT / "data/public_evidence_registry.json").read_text(encoding="utf-8"))
    early = json.loads((ROOT / "data/early_results.json").read_text(encoding="utf-8"))
    middle = json.loads((ROOT / "data/middle_results.json").read_text(encoding="utf-8"))
    late = json.loads((ROOT / "data/late_results.json").read_text(encoding="utf-8"))["rows"]
    with (ROOT / "data/transfer_queries.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for record in registry["records"]:
        if not record.get("projection_distributed"):
            continue
        ledger = record["public_ledger"]
        selector = record["public_selector"]
        if ledger == "data/early_results.json" and not any(
            row["evidence_id"] == record["evidence_id"] for row in early
        ):
            raise ValueError(f"Unresolved public selector: {record['evidence_id']}")
        if ledger == "data/late_results.json" and not any(
            row["evidence_id"] == record["evidence_id"] for row in late
        ):
            raise ValueError(f"Unresolved public selector: {record['evidence_id']}")
        if ledger == "data/middle_results.json":
            value = middle
            for part in selector.strip("/").split("/"):
                if part not in value:
                    raise ValueError(f"Unresolved public selector: {record['evidence_id']}")
                value = value[part]
        if ledger == "data/transfer_queries.csv" and not rows:
            raise ValueError("Empty public transfer projection")
    if len(rows) != 554 or sum(bool(row["target_exclusion"]) for row in rows) != 40:
        raise ValueError("Unexpected public transfer population")
    historical = {row["query_id"] for row in rows if row["cohort"] == "historical46"}
    extended = {row["query_id"] for row in rows if row["cohort"] == "extended_new"}
    historical_groups = {row["group_id"] for row in rows if row["cohort"] == "historical46"}
    extended_groups = {row["group_id"] for row in rows if row["cohort"] == "extended_new"}
    if historical & extended or len(historical_groups & extended_groups) != 3:
        raise ValueError(
            "Transfer cohorts must be query-disjoint and share exactly three sequence groups"
        )
    models = ["PHASE_ONLY_TRAIN40_S7", "Garl_full", "E_CTX_TRAIN40", "R_CTX_TRAIN40"]
    checks = 0
    for cohort, expected_n in [("historical46", 46), ("extended_new", 468)]:
        valid = [row for row in rows if row["cohort"] == cohort and not row["target_exclusion"]]
        if len(valid) != expected_n or not all(float(row["target_ttc"]) > 0 for row in valid):
            raise ValueError(f"Unexpected valid population: {cohort}")
        for model in models:
            mae = statistics.mean(
                abs(float(row[model]) - float(row["target_ttc"])) for row in valid
            )
            rte = statistics.mean(
                100
                * abs(float(row[model]) - float(row["target_ttc"]))
                / abs(float(row["target_ttc"]))
                for row in valid
            )
            summary = next(row for row in late if row["cohort"] == cohort and row["model"] == model)
            if not close(mae, summary["MAE_s"]) or not close(rte, summary["RTE_pct"]):
                raise ValueError(f"Transfer recomputation mismatch: {cohort}/{model}")
            checks += 1
    official = json.loads((ROOT / "data/official_benchmark.json").read_text(encoding="utf-8"))
    scores = json.loads((ROOT / "data/official_scoring_scores.json").read_text(encoding="utf-8"))
    phase = official["phase_submission"]
    comparator = official["competitor"]
    if phase["id"] != 1016512 or comparator["id"] != 817519 or phase["rank"] != 1:
        raise ValueError("Official submission identity or rank mismatch")
    if (
        sha256(ROOT / "data/official_scoring_scores.json")
        != official["provenance"]["B02"]["original_sha256"]
    ):
        raise ValueError("Original official scorer JSON identity mismatch")
    for key, value in scores.items():
        if not close(value, phase["metrics"][key]):
            raise ValueError(f"Official scorer projection mismatch: {key}")
    overall = sum(
        scores[key] * weight for key, weight in official["benchmark"]["band_weights"].items()
    )
    if not close(overall, scores["overall_MiD"]):
        raise ValueError("Official weighted Overall MiD mismatch")
    entries = official["leaderboard_entries"]
    if entries[0]["id"] != phase["id"] or any(
        entries[i]["metrics"]["overall_MiD"] > entries[i + 1]["metrics"]["overall_MiD"]
        for i in range(len(entries) - 1)
    ):
        raise ValueError("Public ranking is inconsistent with lower-is-better Overall MiD")
    for key, value in entries[0]["metrics"].items():
        if not close(round(scores[key], 2), value):
            raise ValueError(f"Official displayed score mismatch: {key}")
    print(
        json.dumps(
            {
                "public_hashes_verified": len(manifest["files"]),
                "transfer_recomputations": checks,
                "official_score_fields_verified": len(scores),
                "official_weighted_MiD": overall,
                "official_public_rank": phase["rank"],
                "query_overlap": 0,
                "shared_sequence_groups": 3,
            }
        )
    )


if __name__ == "__main__":
    main()

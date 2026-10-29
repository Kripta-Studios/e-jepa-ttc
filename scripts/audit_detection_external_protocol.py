"""Audit public eAP benchmark metadata without accessing private benchmark data.

Only the bounded public URLs in ``SOURCES`` are requested.  The script never
requests scoring-program bytes, reference data, test labels, or authenticated
resources.  Raw response bodies are hashed but are not persisted because the
CodaBench competition response contains short-lived signed image URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/detection_sota_20261029/evidence/external_protocol_v2"
MAX_RESPONSE_BYTES = 2_000_000
USER_AGENT = "phase-ttc-public-protocol-audit/2"

EAP_README_COMMIT = "2c5f3a74ba5b693b746820a0a355684e3597dc2d"
GARL_README_COMMIT = "256661242b8a7f5e56aa3c1c02348b30f6e89de6"
CATALOG_COMMITS = (
    "8f259188cc80d54e63339ffe789fb2e56a28455d",
    "3f35e946c71dc49440e94e1329d10d0681381253",
    "e3e53d476cfcc275edb49afbe40c335b9923e7b6",
)
EXPECTED_CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)
EXPECTED_MISSING_TRAIN_IDS = (
    "AUIkq1ZJRv",
    "CEOFFwY7c8",
    "F6XHHaiwJg",
    "OwA5qqkewc",
    "mXRa2Ux9ok",
    "uXfPJNOXVY",
)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    pinned: bool


SOURCES = (
    Source("codabench_competition", "https://www.codabench.org/api/competitions/17291/", False),
    Source(
        "codabench_admin_results",
        "https://www.codabench.org/api/competitions/17291/results/",
        False,
    ),
    Source(
        "codabench_public_leaderboard", "https://www.codabench.org/api/leaderboards/19173/", False
    ),
    Source(
        "codabench_stale_competition", "https://www.codabench.org/api/competitions/16717/", False
    ),
    Source(
        "eap_readme_20260627",
        f"https://raw.githubusercontent.com/NAIL-HNU/eAP_dataset/{EAP_README_COMMIT}/README.md",
        True,
    ),
    Source(
        "garl_readme_20260629",
        f"https://raw.githubusercontent.com/NAIL-HNU/Garl-TTC/{GARL_README_COMMIT}/README.md",
        True,
    ),
    Source(
        "project_page_index",
        f"https://raw.githubusercontent.com/NAIL-HNU/eAP_dataset/{CATALOG_COMMITS[-1]}/index.html",
        True,
    ),
    *(
        Source(
            f"release_catalog_{commit[:8]}",
            "https://raw.githubusercontent.com/NAIL-HNU/eAP_dataset/"
            f"{commit}/assets/data/release_catalog.json",
            True,
        )
        for commit in CATALOG_COMMITS
    ),
    Source(
        "dataset_info",
        f"https://raw.githubusercontent.com/NAIL-HNU/Garl-TTC/{GARL_README_COMMIT}/"
        "configs/dataset_info.json",
        True,
    ),
    Source(
        "train_split",
        f"https://raw.githubusercontent.com/NAIL-HNU/Garl-TTC/{GARL_README_COMMIT}/"
        "configs/splits/train.txt",
        True,
    ),
    Source(
        "test_split",
        f"https://raw.githubusercontent.com/NAIL-HNU/Garl-TTC/{GARL_README_COMMIT}/"
        "configs/splits/test.txt",
        True,
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _fetch(source: Source, *, timeout_seconds: float) -> tuple[dict[str, Any], bytes]:
    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        _require(len(body) <= MAX_RESPONSE_BYTES, f"response exceeds limit: {source.name}")
        status = response.status
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    return (
        {
            "name": source.name,
            "url": source.url,
            "pinned": source.pinned,
            "status": status,
            "content_type": content_type,
            "response_bytes": len(body),
            "raw_sha256": _sha256(body),
        },
        body,
    )


def _as_json(bodies: dict[str, bytes], name: str) -> Any:
    return json.loads(bodies[name].decode("utf-8"))


def _lines(body: bytes) -> list[str]:
    return [line for line in body.decode("utf-8").splitlines() if line]


def _build_audit(source_rows: list[dict[str, Any]], bodies: dict[str, bytes]) -> dict[str, Any]:
    statuses = {row["name"]: row["status"] for row in source_rows}
    _require(statuses["codabench_competition"] == 200, "official competition API unavailable")
    _require(statuses["codabench_admin_results"] == 403, "results access contract changed")
    _require(statuses["codabench_public_leaderboard"] == 200, "public leaderboard unavailable")
    _require(statuses["codabench_stale_competition"] == 404, "stale competition status changed")
    _require(
        all(row["status"] == 200 for row in source_rows if row["pinned"]),
        "pinned source unavailable",
    )

    competition = _as_json(bodies, "codabench_competition")
    pages = {page["title"]: page["content"] for page in competition["pages"]}
    _require(competition["id"] == 17291, "unexpected competition id")
    _require(competition["published"] is True, "competition is not published")
    _require(len(competition["phases"]) == 1, "unexpected phase count")
    _require(len(competition["leaderboards"]) == 1, "unexpected leaderboard count")
    phase = competition["phases"][0]
    leaderboard = competition["leaderboards"][0]
    _require(phase["id"] == 28840 and phase["name"] == "test", "unexpected official phase")
    _require(phase["end"] == "2030-12-31T23:59:59Z", "phase end changed")
    _require(leaderboard["id"] == 19173, "leaderboard id changed")
    columns = [column["key"] for column in leaderboard["columns"]]
    _require(columns == ["AP", "ATE", "ASE", "AOE"], "leaderboard metrics changed")
    overview = pages["Overview"]
    submission_page = pages["Submission"]
    data_page = pages["Data"]
    _require(all(f"`{name}`" in overview for name in EXPECTED_CLASSES), "class contract changed")
    _require(
        all(f"{value:.1f} m" in overview for value in (0.5, 1.0, 2.0, 4.0)),
        "threshold contract changed",
    )
    _require(
        all(f"`{name}`" in submission_page for name in EXPECTED_CLASSES),
        "submission classes changed",
    )
    _require("40 public train sequences" in data_page, "official public train count changed")
    _require("12 public test sequences" in data_page, "official public test count changed")

    task = phase["tasks"][0]
    private_assets = {item["type"]: item for item in task["public_datasets"]}
    scoring = private_assets["scoring_program"]
    reference = private_assets["reference_data"]
    _require(
        scoring["is_public"] is False and scoring["file_size"] == 9821.0, "scorer metadata changed"
    )
    _require(reference["is_public"] is False, "reference-data visibility changed")
    _require(competition["make_programs_available"] is False, "program availability changed")

    public_leaderboard = _as_json(bodies, "codabench_public_leaderboard")
    _require(public_leaderboard == {"submissions": []}, "public leaderboard is no longer empty")
    results_error = _as_json(bodies, "codabench_admin_results")
    _require("not a competition admin" in results_error["detail"], "unexpected results denial")

    eap_readme = bodies["eap_readme_20260627"].decode("utf-8")
    garl_readme = bodies["garl_readme_20260629"].decode("utf-8")
    _require("competitions/16717" in eap_readme, "historical eAP link changed")
    _require("competitions/17291" in garl_readme, "new official detection link missing")
    _require("competitions/17289" in garl_readme, "official TTC link missing")

    project_page = bodies["project_page_index"].decode("utf-8")
    _require("Train: 46 sequences" in project_page, "described train count changed")
    train_ids = _lines(bodies["train_split"])
    test_ids = _lines(bodies["test_split"])
    _require(len(train_ids) == 46 and len(set(train_ids)) == 46, "train split is not 46 unique ids")
    _require(len(test_ids) == 12 and len(set(test_ids)) == 12, "test split is not 12 unique ids")
    _require(set(train_ids).isdisjoint(test_ids), "official train/test ids overlap")

    catalog_history: list[dict[str, Any]] = []
    latest_catalog_ids: set[str] = set()
    for commit in CATALOG_COMMITS:
        catalog = _as_json(bodies, f"release_catalog_{commit[:8]}")
        ids = {asset["asset_id"] for asset in catalog["assets"]}
        _require(len(ids) == len(catalog["assets"]), f"duplicate catalog id at {commit}")
        missing_occurrences = sorted(set(EXPECTED_MISSING_TRAIN_IDS) & ids)
        _require(not missing_occurrences, f"missing ids unexpectedly present at {commit}")
        catalog_history.append(
            {
                "commit": commit,
                "asset_count": len(ids),
                "missing_id_occurrences": missing_occurrences,
            }
        )
        if commit == CATALOG_COMMITS[-1]:
            latest_catalog_ids = ids
    missing = tuple(sorted(set(train_ids) - latest_catalog_ids))
    _require(
        missing == tuple(sorted(EXPECTED_MISSING_TRAIN_IDS)), "46-to-40 reconciliation changed"
    )
    _require(
        latest_catalog_ids <= set(train_ids), "catalog contains an id outside official train split"
    )

    dataset_info = _as_json(bodies, "dataset_info")
    _require(len(dataset_info) == 58, "dataset_info sequence count changed")
    expected_metadata_keys = {
        "weather",
        "road_type",
        "contains_hdr",
        "targets_per_frame",
        "relative_speed",
        "split",
    }
    missing_metadata: dict[str, Any] = {}
    for sequence_id in EXPECTED_MISSING_TRAIN_IDS:
        metadata = dataset_info[sequence_id]
        _require(
            set(metadata) == expected_metadata_keys, f"unexpected metadata keys for {sequence_id}"
        )
        _require(metadata["split"] == "train", f"missing id is not marked train: {sequence_id}")
        missing_metadata[sequence_id] = metadata

    return {
        "schema": "phase_ttc_external_detection_protocol_audit_v2",
        "audit_date": "2026-10-29",
        "scope": "public metadata only; no private scorer/reference/test labels requested",
        "official_contract": {
            "competition_id": competition["id"],
            "title": competition["title"],
            "published": competition["published"],
            "classes": list(EXPECTED_CLASSES),
            "center_distance_thresholds_m": [0.5, 1.0, 2.0, 4.0],
            "leaderboard_columns": columns,
            "primary_metric": "AP",
            "phase": {
                "id": phase["id"],
                "name": phase["name"],
                "start": phase["start"],
                "end": phase["end"],
                "status": phase["status"],
            },
            "public_sequence_counts": {"train": 40, "test": 12},
            "submission_format": "eap_native_detection_result JSON in root-level submission.json",
        },
        "private_metric_assets": {
            "scoring_program": {"is_public": scoring["is_public"], "metadata_size_bytes": 9821},
            "reference_data": {
                "is_public": reference["is_public"],
                "metadata_size_bytes": int(reference["file_size"]),
            },
            "make_programs_available": competition["make_programs_available"],
            "make_input_data_available": competition["make_input_data_available"],
            "bytes_requested": 0,
        },
        "leader_status": {
            "admin_results_endpoint_status": statuses["codabench_admin_results"],
            "public_leaderboard_endpoint_status": statuses["codabench_public_leaderboard"],
            "public_submission_rows": 0,
            "official_leader_verifiable": False,
            "reason": "admin results are forbidden and the public leaderboard response is empty",
        },
        "link_reconciliation": {
            "eap_readme_commit": EAP_README_COMMIT,
            "eap_readme_commit_date": "2026-06-27",
            "stale_detection_competition_id": 16717,
            "stale_api_status": statuses["codabench_stale_competition"],
            "garl_readme_commit": GARL_README_COMMIT,
            "garl_readme_commit_date": "2026-06-29",
            "current_detection_competition_id": 17291,
            "ttc_competition_id": 17289,
        },
        "release_reconciliation": {
            "described_train_sequences": 46,
            "official_train_split_ids": 46,
            "latest_public_catalog_assets": len(latest_catalog_ids),
            "missing_public_media_ids": list(EXPECTED_MISSING_TRAIN_IDS),
            "catalog_history": catalog_history,
            "dataset_info_sha256": next(
                row["raw_sha256"] for row in source_rows if row["name"] == "dataset_info"
            ),
            "train_split_sha256": next(
                row["raw_sha256"] for row in source_rows if row["name"] == "train_split"
            ),
            "missing_ids_public_evidence": {
                "catalog_entries": 0,
                "media_urls": 0,
                "file_sizes": 0,
                "file_hashes": 0,
                "label_files": 0,
                "categorical_metadata": missing_metadata,
            },
            "scientific_interpretation": (
                "categorical metadata does not establish media/label availability, additional "
                "diversity, or acquisition independence"
            ),
        },
        "claim_boundary": {
            "local_protocol_is_official_score": False,
            "official_metric_implementation_bytes_available": False,
            "leaderboard_rank_available": False,
            "six_missing_sequences_usable": False,
            "next_valid_claim": (
                "official score/rank requires an admitted CodaBench submission or public scorer; "
                "the six ids require released media, labels, hashes, and an exposure audit"
            ),
        },
    }


def _render_readme(audit: dict[str, Any]) -> bytes:
    contract = audit["official_contract"]
    release = audit["release_reconciliation"]
    text = f"""# External detection protocol evidence v2

Audit date: `{audit["audit_date"]}`. Public metadata only.

The official public route is CodaBench competition `{contract["competition_id"]}` with eight
classes, center-distance thresholds 0.5/1/2/4 m, and leaderboard columns AP/ATE/ASE/AOE.
The scoring program is private; only its public metadata size (9,821 bytes) was observed.
No scoring-program, reference-data, or private-label bytes were requested.

The public results-admin endpoint returned 403 and the public leaderboard endpoint returned
an empty submission list. Therefore this audit verifies the contract but not a leader, score,
rank, or equivalence of the local evaluator to the private scorer.

The official train list describes 46 IDs, while every inspected release-catalog revision
contains 40 assets. The six absent IDs are `{", ".join(release["missing_public_media_ids"])}`.
They have categorical `dataset_info` rows, but no catalog media URL, size, hash, label file,
or evidence of acquisition independence. They are not usable data in the current campaign.

Regenerate with bounded public GETs:

```powershell
python scripts/audit_detection_external_protocol.py
python scripts/audit_detection_external_protocol.py --check
```
"""
    return text.encode()


def _manifest(files: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema": "phase_ttc_external_detection_protocol_manifest_v2",
        "files": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(files.items())
        },
    }


def _verify_existing(output_dir: Path) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(
        manifest["schema"] == "phase_ttc_external_detection_protocol_manifest_v2", "bad manifest"
    )
    for name, expected in manifest["files"].items():
        data = (output_dir / name).read_bytes()
        _require(len(data) == expected["bytes"], f"size mismatch: {name}")
        _require(_sha256(data) == expected["sha256"], f"hash mismatch: {name}")


def _run(*, timeout_seconds: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    bodies: dict[str, bytes] = {}
    for source in SOURCES:
        row, body = _fetch(source, timeout_seconds=timeout_seconds)
        source_rows.append(row)
        bodies[source.name] = body
    return _build_audit(source_rows, bodies), source_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _require(0 < args.timeout_seconds <= 60, "timeout must be in (0, 60]")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    audit, source_rows = _run(timeout_seconds=args.timeout_seconds)
    audit_bytes = _json_bytes(audit)
    source_bytes = _json_bytes(
        {
            "schema": "phase_ttc_external_detection_protocol_sources_v2",
            "max_response_bytes": MAX_RESPONSE_BYTES,
            "raw_bodies_persisted": False,
            "sources": source_rows,
        }
    )
    readme_bytes = _render_readme(audit)
    files = {"audit.json": audit_bytes, "sources.json": source_bytes, "README.md": readme_bytes}
    manifest_bytes = _json_bytes(_manifest(files))
    if args.check:
        _verify_existing(output_dir)
        _require((output_dir / "audit.json").read_bytes() == audit_bytes, "live contract drift")
        _require((output_dir / "README.md").read_bytes() == readme_bytes, "generated README drift")
        stored_sources = json.loads((output_dir / "sources.json").read_text(encoding="utf-8"))
        stored_by_name = {row["name"]: row for row in stored_sources["sources"]}
        for row in source_rows:
            stored = stored_by_name[row["name"]]
            _require(stored["status"] == row["status"], f"status drift: {row['name']}")
            if row["pinned"]:
                _require(
                    stored["raw_sha256"] == row["raw_sha256"],
                    f"pinned source drift: {row['name']}",
                )
        print(f"PASS: {len(source_rows)} bounded public sources; contract and evidence verified")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    print(f"WROTE: {output_dir.relative_to(ROOT)} from {len(source_rows)} bounded public sources")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

"""Independently verify archive rows against the measured inference ledger."""

from __future__ import annotations

import math
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read


def verify_artifact_bindings(
    root: Path,
    historical_run: Path,
    population: Path,
    exclusions: Path,
    road_targets: Path,
    slider_contract: Path,
    source_root: Path,
    *,
    current_commit: str,
) -> None:
    """Reject substitution of model, evidence inputs, dependencies or checked commit."""
    freeze = read(root / "replay/RUN_FREEZE.json")
    evaluation = read(root / "evaluation/LOCAL_EVALUATION.json")
    checks = read(root / "FINAL_CHECKS.json")
    digest = file_sha(root / "FINAL_MODEL/MODEL.json")
    if not (
        digest == freeze["model_sha256"] == evaluation["model_sha256"] == checks["model_sha256"]
    ):
        raise ValueError("Model identity binding differs across delivery artifacts")
    if (
        checks["status"] != "passed"
        or file_sha(root / "replay/results.zip") != checks["results_zip_sha256"]
    ):
        raise ValueError("Final checks binding differs")
    if checks["git_commit"] != current_commit:
        raise ValueError("Final checked commit differs")
    config = read(root / "training/CONFIG.json")
    if Counter(config["inputs"].values()) != Counter(map(file_sha, (population, exclusions))):
        raise ValueError("Training input binding differs")
    targets = [road_targets, *(slider_contract / f"{s}_gt.csv" for s in (750, 1000))]
    if Counter(evaluation["target_sources"].values()) != Counter(map(file_sha, targets)):
        raise ValueError("Target source binding differs")
    audit = read(root / "audit/HISTORICAL_AUDIT.json")
    for name, expected in audit["source_hashes"].items():
        if file_sha(historical_run / name) != expected:
            raise ValueError("Historical source binding differs")
    supplement = read(root / "audit/SOURCE_SUPPLEMENT.json")
    for name, expected in supplement["files"].items():
        if file_sha(source_root / name) != expected:
            raise ValueError("Supplemental executable source binding differs")


def verify_results(
    archive: Path,
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Sequence[tuple[int, str]]],
) -> dict[str, Any]:
    """Require exact identity, signed values and costs; reject duplicate/extra rows."""
    by_key = {}
    for row in records:
        identity = row["identity"]
        key = (identity["sequence_id"], identity["row_index"])
        if key in by_key:
            raise ValueError("duplicate ledger identity")
        by_key[key] = row
    wanted = {(seq, i) for seq, rows in expected.items() for i in range(len(rows))}
    if set(by_key) != wanted:
        raise ValueError("Incomplete or extra ledger identities")
    costs = []
    negative = 0
    with zipfile.ZipFile(archive) as z:
        names = [f"{seq}/results.txt" for seq in expected]
        if sorted(z.namelist()) != sorted(names) or z.testzip() is not None:
            raise ValueError("Wrong archive member inventory or CRC")
        for seq, queries in expected.items():
            lines = z.read(f"{seq}/results.txt").decode("utf-8").splitlines()
            if lines[0] != "Index\ttimestamp (s)\tttc (s)\tcost time(s)":
                raise ValueError("Wrong submission header")
            if len(lines) != len(queries) + 1:
                raise ValueError("Wrong sequence row count")
            for i, ((index, stamp), line) in enumerate(zip(queries, lines[1:], strict=True)):
                fields = line.split("\t")
                row = by_key[(seq, i)]
                identity = row["identity"]
                if (
                    len(fields) != 4
                    or fields[:2] != [str(index), stamp]
                    or identity["index"] != index
                    or identity["timestamp_text"] != stamp
                    or row["status"] != "ok"
                ):
                    raise ValueError("Changed query identity or status")
                prediction, cost = map(float, fields[2:])
                if not math.isfinite(prediction) or not math.isfinite(cost) or cost < 0:
                    raise ValueError("Nonfinite prediction or invalid measured cost")
                if prediction != row["prediction_ttc_seconds"] or cost != row["cost_time_seconds"]:
                    raise ValueError("Archive differs from inference ledger")
                costs.append(cost)
                negative += prediction < 0
    return {
        "rows": len(costs),
        "negative": negative,
        "mean_cost_seconds": sum(costs) / len(costs),
        "exact_ledger_match": True,
    }

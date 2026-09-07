"""Recheck historical TRAIN replay evidence without rerunning frozen experts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def verify_historical_replay(work: Path, report: Path, expected_sha256: str) -> dict:
    """Check pinned diagnostic payloads and all twelve historical producer families.

    This preserves the original failed coherent-replay attempt. Historical
    component parity uses its documented runtime routes, not that failed output.
    No target labels, raw media or expert checkpoints are opened here. Separate
    ancestry checks and coherent FP32 production-cache QA remain mandatory.
    """
    if report.stat().st_size > 1_048_576 or sha256(report) != expected_sha256:
        raise ValueError("historical replay report changed")
    record = json.loads(report.read_text(encoding="utf-8"))
    if record.get("status") != "HISTORICAL64_COMPONENT_REPLAY_PARITY_PASSED":
        raise ValueError("historical component parity report required")
    base = work.resolve(strict=True)
    files = {}
    for relative, digest in record["sources"].items():
        path = (base / relative).resolve(strict=True)
        if not path.is_relative_to(base) or path.stat().st_size > 16_777_216:
            raise ValueError("replay evidence path or size invalid")
        if sha256(path) != digest:
            raise ValueError("historical replay dependency changed")
        if path.name in files:
            raise ValueError("ambiguous replay dependency name")
        files[path.name] = path

    def read(name: str) -> dict:
        return json.loads(files[name].read_text(encoding="utf-8"))

    families = record["families"]
    expected = {
        f"outer{o}/{inner}"
        for o in range(3)
        for inner in ("inner0", "inner1", "inner2", "outer_dev")
    }
    tokens = [token for values in families.values() for token in values]
    if set(families) != expected or len(tokens) != 64 or len(set(tokens)) != 64:
        raise ValueError("historical replay family or query coverage differs")
    raw = read("QUERY_CONTEXT_RAW_INPUT_64.json")["queries"]
    if len(raw) != 64 or {row["token"] for row in raw} != set(tokens):
        raise ValueError("raw-input query coverage differs")
    for row in raw:
        windows = row["results"]
        if len(windows) != 3 or {w["window"] for w in windows} != {0, 1, 2}:
            raise ValueError("three historical input windows required")
        if any(w["bit_identical"] is not True or any(w["max_abs_by_channel"]) for w in windows):
            raise ValueError("raw historical input parity failed")
    positions = read("A5_HISTORICAL_BATCH_POSITION.json")["results"]
    if len(positions) != 64 or {r["token"] for r in positions} != set(tokens):
        raise ValueError("A5 token parity coverage differs")
    for row in positions:
        if (
            row["token"] not in families[row["family"]]
            or row["max_abs_pair_token_by_cudnn_tf32"]["True"] != 0
            or row["abs_phase_error_by_cudnn_tf32"]["True"] != 0
        ):
            raise ValueError("A5 historical extraction parity failed")
    precision = read("HISTORICAL_DETERMINISTIC_VALIDATION.json")["results"]
    pairs = read("PAIR_HISTORICAL_GPU_LAYOUT.json")["results"]
    if (
        len(precision) != 24
        or {(r["family"], r["expert"]) for r in precision}
        != {(f, e) for f in expected for e in ("A5", "C2F")}
        or len(pairs) != 12
        or {r["family"] for r in pairs} != expected
    ):
        raise ValueError("historical prediction coverage differs")
    for family in sorted(expected):
        with np.load(files[family.replace("/", "_") + ".npz"], allow_pickle=False) as archive:
            if archive["tokens"].tolist() != families[family]:
                raise ValueError("historical prediction query order differs")
            target = archive["expected_ttc"].astype(np.float32)
            if target.shape != (len(families[family]), 3):
                raise ValueError("historical expert prediction shape differs")
            for column, expert in enumerate(("A5", "C2F")):
                row = next(r for r in precision if (r["family"], r["expert"]) == (family, expert))
                actual = np.asarray(row["modes"]["bf16_cudnn_tf32"]["ttc"], dtype=np.float32)
                if not np.array_equal(actual, target[:, column]):
                    raise ValueError("historical A5/C2F prediction parity failed")
            pair = next(r for r in pairs if r["family"] == family)
            if pair["tokens"] != families[family] or not np.array_equal(
                np.asarray(pair["pair_from_signed_features_ttc"], dtype=np.float32), target[:, 2]
            ):
                raise ValueError("historical PAIR prediction parity failed")
    return {
        "status": "HISTORICAL_COMPONENT_EVIDENCE_REVERIFIED",
        "queries": 64,
        "families": 12,
        "historical_predictions": 192,
        "raw_windows": 192,
        "new_optimizer_updates": 0,
        "scientific_stage_authorized": False,
        "report_sha256": expected_sha256,
        "limitation": "Historical component routes only; not coherent FP32 cache QA or ancestry.",
    }

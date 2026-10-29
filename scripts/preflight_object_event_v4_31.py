#!/usr/bin/env python3
"""Fail-closed preflight for a sanitized v4.31 cache; it never opens mixed NPZs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from e_jepa_ttc.data.object_event_v4_31 import (  # noqa: E402, I001
    ADAPT_SEQUENCES,
    OWNERSHIP_MARKER,
    PROJECTED_COLUMNS,
    SPLIT_CONTRACT,
    SPLIT_PATH,
    allocate_quotas,
    sha256_file,
    load_split_contract,
    strict_json,
)


def run(config: Path, cache: Path, *, full: bool) -> dict[str, Any]:
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("source"), dict):
        raise ValueError("config must declare the v4.31 source mapping")
    source_raw = os.path.expandvars(str(raw["source"].get("train_parquet")))
    event_root_raw = os.path.expandvars(str(raw.get("event_root", "")))
    if "${" in source_raw or "${" in event_root_raw or not event_root_raw:
        raise ValueError("config source/event-root templates must resolve before preflight")
    source_path = str(Path(source_raw).resolve())
    split_path = Path(os.path.expandvars(str(raw.get("split", ""))))
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if split_path.resolve() != SPLIT_PATH.resolve() or raw.get("split_sha256") != sha256_file(
        split_path
    ):
        raise ValueError("config does not bind the authoritative v4.31 split SHA")
    split_contract = load_split_contract(split_path)
    if split_contract != SPLIT_CONTRACT:
        raise ValueError("loaded split contract differs from the locked code contract")
    effective = dict(raw)
    effective_source = dict(raw["source"])
    effective_source["train_parquet"] = source_path
    effective["source"] = effective_source
    effective["event_root"] = str(Path(event_root_raw).resolve())
    effective["split"] = str(split_path.resolve())
    config_identity = hashlib.sha256(strict_json(effective).encode("utf-8")).hexdigest()
    source_sha = str(raw["source"].get("sha256"))
    manifest = json.loads((cache / "manifest.json").read_text())
    cache_schema = json.loads(
        (ROOT / "schemas/object_event_v4_31_sanitized_cache_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        jsonschema.validate(manifest, cache_schema)
    except jsonschema.ValidationError as exc:
        raise ValueError("sanitized cache manifest violates its strict schema") from exc
    if not (cache / OWNERSHIP_MARKER).is_file():
        raise ValueError("sanitized cache ownership marker absent")
    marker = json.loads((cache / OWNERSHIP_MARKER).read_text(encoding="utf-8"))
    if marker != {
        "artifact": "object_event_v4_31",
        "owner": "e_jepa_ttc",
        "config_identity": config_identity,
        "source_identity": f"{source_path}:{source_sha}",
    }:
        raise ValueError("sanitized cache ownership marker content differs")
    if manifest.get("artifact_type") != "object_event_v4_31_sanitized_cache_v1":
        raise ValueError("not a sanitized v4.31 cache")
    expected_mode = "full" if full else "diagnostic"
    expected = 4096 if full else 512
    if manifest.get("mode") != expected_mode or manifest.get("count") != expected:
        raise ValueError("manifest mode/count differs from requested preflight mode")
    source = manifest.get("source")
    if not isinstance(source, dict) or source != {
        "path": source_path,
        "sha256": source_sha,
        "projection": list(PROJECTED_COLUMNS),
    }:
        raise ValueError("manifest source/projection provenance differs from config")
    representation = manifest.get("representation")
    if not isinstance(representation, dict) or representation.get("id") != "v4_30_common_roi":
        raise ValueError("manifest representation is not the locked v4.30 common ROI")
    if representation.get("interval") != "[start,end)" or representation.get("shape") != [
        3,
        12,
        128,
        128,
    ]:
        raise ValueError("manifest representation interval/shape differs from contract")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance != {
        "boxes_transient_only": True,
        "targets_opened": False,
    }:
        raise ValueError("manifest provenance contract differs")
    split = manifest.get("split")
    if split != {
        "path": str(split_path.resolve()),
        "sha256": sha256_file(split_path),
        "version": split_contract["version"],
    }:
        raise ValueError("manifest split identity differs from the authoritative contract")
    opened = manifest.get("opened_paths")
    if not isinstance(opened, list) or source_path not in opened:
        raise ValueError("manifest opened_paths must include the source parquet")
    forbidden = ("annotations", "test_inputs.parquet", "validation", "development", "evttc")
    if any(token in str(path).lower() for path in opened for token in forbidden):
        raise PermissionError("manifest records a forbidden opened path")
    if any("v4_30" in str(path).lower() for path in (cache,)):
        raise PermissionError("auditor cannot use a v4.30 source directory")
    events_meta = manifest.get("events")
    delta_meta = manifest.get("delta_t_s")
    if not isinstance(events_meta, dict) or not isinstance(delta_meta, dict):
        raise ValueError("manifest lacks array metadata")
    if (
        events_meta.get("path") != "events.npy"
        or events_meta.get("dtype") != "float16"
        or events_meta.get("shape") != [expected, 3, 12, 128, 128]
        or delta_meta.get("path") != "delta_t_s.npy"
        or delta_meta.get("dtype") != "float32"
        or manifest.get("rows_path") != "rows.jsonl"
    ):
        raise ValueError("manifest array paths, dtypes, or shapes differ from contract")
    events = np.load(cache / "events.npy", mmap_mode="r")
    delta = np.load(cache / "delta_t_s.npy", mmap_mode="r")
    if (
        events.shape != (expected, 3, 12, 128, 128)
        or events.dtype != np.float16
        or len(delta) != expected
    ):
        raise ValueError("cache count/shape contract failed")
    if delta.dtype != np.float32 or not np.isfinite(delta).all() or not np.all(delta > 0):
        raise ValueError("delta_t_s must be finite positive float32")
    if not np.isfinite(np.asarray(events)).all():
        raise ValueError("events.npy contains nonfinite values")
    rows = (cache / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    if len(rows) != expected:
        raise ValueError("rows.jsonl count differs from cache mode")
    parsed = [json.loads(line) for line in rows]
    allowed_row_keys = {"row_index", "row_sha256", "sequence_id", "pool", "delta_t_s"}
    if any(not isinstance(row, dict) or set(row) != allowed_row_keys for row in parsed):
        raise ValueError("rows.jsonl fields differ from the sanitized allowlist")
    if [row.get("row_index") for row in parsed] != list(range(expected)):
        raise ValueError("rows.jsonl indices are not contiguous")
    identities = [row.get("row_sha256") for row in parsed]
    if len(set(identities)) != expected:
        raise ValueError("rows.jsonl contains duplicate identities")
    quotas = allocate_quotas(full=full)
    observed = {
        sequence: sum(row.get("sequence_id") == sequence for row in parsed) for sequence in quotas
    }
    if observed != quotas or any(row.get("sequence_id") not in quotas for row in parsed):
        raise ValueError("rows.jsonl sequence pools or quotas differ from locked split")
    for row in parsed:
        expected_pool = "adaptation" if row["sequence_id"] in ADAPT_SEQUENCES else "audit"
        if row.get("pool") != expected_pool:
            raise ValueError("rows.jsonl pool differs from locked split")
    if events_meta.get("sha256") != sha256_file(cache / "events.npy"):
        raise ValueError("events.npy hash differs from manifest")
    if delta_meta.get("sha256") != sha256_file(cache / "delta_t_s.npy"):
        raise ValueError("delta_t_s.npy hash differs from manifest")
    rows_hash = hashlib.sha256((cache / "rows.jsonl").read_bytes()).hexdigest()
    if manifest.get("rows_sha256") is not None and manifest["rows_sha256"] != rows_hash:
        raise ValueError("rows.jsonl hash differs from manifest")
    return {
        "artifact_type": "object_event_v4_31_preflight_v1",
        "status": "passed",
        "config_version": raw["audit_version"],
        "count": expected,
        "opened_paths": [
            str(cache / "manifest.json"),
            str(cache / "events.npy"),
            str(cache / "delta_t_s.npy"),
            str(cache / "rows.jsonl"),
            str(cache / OWNERSHIP_MARKER),
        ],
        "forbidden_access": [],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_operator_audit_v4_31.yaml",
    )
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--full", action="store_true")
    a = p.parse_args()
    try:
        print(strict_json(run(a.config, a.cache, full=a.full)))
    except Exception as exc:
        print(
            strict_json(
                {
                    "artifact_type": "object_event_v4_31_preflight_v1",
                    "status": "invalid_incomplete",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

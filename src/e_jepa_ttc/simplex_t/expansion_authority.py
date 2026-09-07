"""Import the exact Stage70 supplementary temporal ACK, never infer new roles."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

ACK_SHA256 = "b8d2e49aad4ee3d922042f053ebdcf4f0c4511e665f822770c6d0f5f4bf419be"
REQUEST_SHA256 = "a649e397bd941ae37a95100b93fdb3f767157c79eeb653d38f8956a678191dfc"


def verify_expansion_authority(local_paths: Path, *, resource_ok: Callable[[], bool]) -> dict:
    """Rehash acknowledged administrative inputs without decoding labels or media.

    Recognition covers supplied-ROI exposure dependencies, not online annotation
    latency, unseen lag support, checkpoint parity or permission to refit experts.
    Exact ACK bytes bind the complete semantics and exclusion lists. All 29
    references are bounded metadata/source files, not referenced raw media.
    """
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    shared = paths.get("shared_coordination")
    if not shared:
        raise ValueError("WAITING_EXPANDED_TIME_RECOGNITION: coordination path missing")
    path = Path(shared) / "SIMPLEX_T_STAGE70_EXPANSION_ACK.json"
    if not path.is_file():
        raise ValueError("WAITING_EXPANDED_TIME_RECOGNITION: supplementary ACK missing")
    if path.stat().st_size > 1_048_576 or sha256(path) != ACK_SHA256:
        raise ValueError("supplementary temporal ACK bytes changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record["request_id"] != "SIMPLEX_T_2026-09-07_EXPANSION_TIME_ROI"
        or record["request"]["sha256"] != REQUEST_SHA256
        or record["scope"]["D1_EXPANSION"]["recognized"] is not True
        or record["scope"]["DENSE_OLD"]["recognized"] is not True
        or record["historical_experts"]["preprocessing_replaced"] is not False
        or record["time_and_roi_contract"]["online_causal_availability_at_sensor_anchor_accredited"]
        is not False
    ):
        raise ValueError("supplementary ACK does not recognize the audited input contract")
    evidence = record["evidence"]
    if len(evidence) != 29:
        raise ValueError("complete supplementary evidence set required")
    total = 0
    # Each source is bounded at 32 MB and released by sha256 before the next.
    # Admission brackets this metadata-only transaction; querying the entire
    # Windows process tree separately for all 29 files costs more than hashing.
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: supplementary temporal evidence")
    for entry in evidence:
        source = Path(entry["path"])
        if (
            entry["bytes"] > 32_000_000
            or source.stat().st_size != entry["bytes"]
            or sha256(source) != entry["sha256"]
        ):
            raise ValueError("supplementary temporal evidence changed: " + source.name)
        total += entry["bytes"]
    if not resource_ok():
        raise InterruptedError("PAUSED_RESOURCE: supplementary temporal evidence")
    if sha256(path) != ACK_SHA256:
        raise ValueError("supplementary ACK changed during verification")
    return {
        "status": "EXPANDED_TEMPORAL_AUTHORITY_VERIFIED_NOT_REPLAY_ADMISSION",
        "path": str(path.resolve()),
        "sha256": ACK_SHA256,
        "evidence": evidence,
        "bytes_hashed": total,
        "scope": record["scope"],
        "time_and_roi_contract": record["time_and_roi_contract"],
        "remaining_execution_obligations": record["remaining_execution_obligations"],
        "optimizer_updates": 0,
    }

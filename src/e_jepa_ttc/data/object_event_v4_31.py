"""Data-contract utilities for the TTC-label-free, train-box-conditioned v4.31 audit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_SHA256 = "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6"
PROJECTED_COLUMNS = (
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "events_path",
    "event_windows_us",
    "boxes_xyxy",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = REPOSITORY_ROOT / "data/splits/object_event_v4_31_train_only_v1.json"
OWNERSHIP_MARKER = ".object_event_v4_31_owned.json"


def strict_json(value: object) -> str:
    """Serialize only finite strict JSON."""
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def scientific_metadata(
    *,
    artifact_type: str,
    evidence_type: str,
    protocol_version: str,
    protocol_sha256: str,
    artifact_sha256: str,
) -> dict[str, str]:
    """Minimum repository-wide provenance fields for an emitted scientific artifact."""
    commit = os.environ.get("GIT_COMMIT")
    if not commit:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else ""
    if len(commit) < 7:
        raise RuntimeError("v4.31 artifacts require a resolvable Git commit")
    return {
        "schema_version": "1.0",
        "evidence_type": evidence_type,
        "code_commit": commit,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_sha256": artifact_sha256,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_contract(path: Path = SPLIT_PATH) -> dict[str, Any]:
    """Load and structurally validate the sole authoritative v4.31 split contract."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "source",
        "adaptation_sequences",
        "audit_sequences",
        "minimum_track_gap_us",
        "selection",
        "full",
        "diagnostic",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("v4.31 split contract keys differ from the locked schema")
    adaptation = raw["adaptation_sequences"]
    audit = raw["audit_sequences"]
    if (
        not isinstance(adaptation, list)
        or not isinstance(audit, list)
        or len(adaptation) != 4
        or len(audit) != 5
        or len(set(adaptation + audit)) != 9
        or not all(isinstance(item, str) and item for item in adaptation + audit)
    ):
        raise ValueError("v4.31 split sequence pools are invalid")
    if raw["selection"] != "salted_sha256_identity" or raw["minimum_track_gap_us"] != 100_000:
        raise ValueError("v4.31 split selection or gap differs from the locked contract")
    for mode, expected_adaptation, expected_total in (("diagnostic", 64, 512), ("full", 512, 4096)):
        section = raw[mode]
        if (
            not isinstance(section, dict)
            or section.get("adaptation_per_sequence") != expected_adaptation
        ):
            raise ValueError(f"v4.31 {mode} adaptation quota differs from contract")
        audit_quota = section.get("audit_per_sequence")
        if not isinstance(audit_quota, dict) or set(audit_quota) != set(audit):
            raise ValueError(f"v4.31 {mode} audit quota keys differ from contract")
        if (
            sum([expected_adaptation] * len(adaptation) + list(audit_quota.values()))
            != expected_total
        ):
            raise ValueError(f"v4.31 {mode} total quota differs from contract")
    if raw["diagnostic"].get("nested_full_prefix") is not True:
        raise ValueError("diagnostic split must be the per-sequence full prefix")
    return raw


SPLIT_CONTRACT = load_split_contract()
ADAPT_SEQUENCES = tuple(SPLIT_CONTRACT["adaptation_sequences"])
AUDIT_SEQUENCES = tuple(SPLIT_CONTRACT["audit_sequences"])


def reject_forbidden_path(path: Path | str, *, source: bool = False) -> None:
    """Apply component-aware allow/deny rules without rejecting ``GarlTTC`` itself."""
    candidate = Path(path)
    normalized = candidate.as_posix().lower()
    components = {part.lower() for part in candidate.parts}
    forbidden_components = {"annotations", "validation", "test", "development", "dev"}
    forbidden_alias = any(
        "evttc" in part or "development" in part for part in components
    )
    if (
        components.intersection(forbidden_components)
        or forbidden_alias
        or "test_inputs.parquet" in normalized
    ):
        raise PermissionError(f"v4.31 forbidden input path: {path}")
    if "v4_30" in normalized or "mixed" in normalized:
        raise PermissionError(f"v4.31 cannot open mixed/v4.30 cache path: {path}")
    if source and candidate.suffix.lower() != ".parquet":
        raise PermissionError("v4.31 source must be the SHA-verified parquet projection")


def resolve_event_path(events_path: str | Path, *, event_root: Path | None = None) -> Path:
    """Resolve a projected relative event path and keep it below the configured root."""
    if event_root is None:
        configured = os.environ.get("E_JEPA_TTC_EVENT_ROOT")
        if not configured:
            raise ValueError("event root must be passed explicitly or set E_JEPA_TTC_EVENT_ROOT")
        event_root = Path(configured)
    raw = Path(events_path)
    if raw.is_absolute():
        raise ValueError("events_path must be relative to the configured event root")
    resolved = (event_root / raw).resolve()
    root = event_root.resolve()
    if root != resolved and root not in resolved.parents:
        raise PermissionError("events_path escapes configured event root")
    reject_forbidden_path(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"resolved events_path is absent: {resolved}")
    return resolved


def validate_projection(columns: Iterable[str]) -> None:
    """Require exactly the audited parquet projection and no hidden field."""
    actual = tuple(columns)
    if actual != PROJECTED_COLUMNS:
        raise ValueError(f"projection must be exactly {PROJECTED_COLUMNS!r}")


def identity(row: Mapping[str, Any]) -> str:
    """Stable non-label identity for ranking and traceable sanitized rows."""
    return "|".join(
        str(row[key])
        for key in ("sequence_id", "sample_token", "track_id", "public_track_id", "timestamp_us")
    )


def salted_rank(row: Mapping[str, Any], salt: str = "object-event-v4.31-train-only-v1") -> bytes:
    return hashlib.sha256(f"{salt}|{identity(row)}".encode()).digest()


def allocate_quotas(*, full: bool) -> dict[str, int]:
    """Locked exact sequence quotas. Diagnostic is a per-sequence full prefix."""
    section = SPLIT_CONTRACT["full" if full else "diagnostic"]
    return {
        **{item: section["adaptation_per_sequence"] for item in ADAPT_SEQUENCES},
        **{str(item): int(value) for item, value in section["audit_per_sequence"].items()},
    }


def select_split(
    rows: Iterable[Mapping[str, Any]], *, full: bool, minimum_gap_us: int | None = None
) -> list[dict[str, Any]]:
    """Rank only by identity then select disjoint train-only sequence quotas.

    The gap rule is per sequence/track and deliberately ignores event rates and motion.
    """
    required_gap = int(SPLIT_CONTRACT["minimum_track_gap_us"])
    if minimum_gap_us is not None and minimum_gap_us != required_gap:
        raise ValueError("minimum track gap must come from the locked split contract")
    quotas = allocate_quotas(full=full)
    available: dict[str, list[Mapping[str, Any]]] = {key: [] for key in quotas}
    for row in rows:
        seq = str(row.get("sequence_id"))
        if seq in available:
            available[seq].append(row)
    chosen: list[dict[str, Any]] = []
    for sequence, quota in quotas.items():
        ordered = sorted(available[sequence], key=salted_rank)
        accepted: list[Mapping[str, Any]] = []
        accepted_by_track: dict[str, list[int]] = {}
        for row in ordered:
            track = str(row["track_id"])
            timestamp = int(row["timestamp_us"])
            prior = accepted_by_track.get(track, [])
            if any(abs(timestamp - previous) < required_gap for previous in prior):
                continue
            accepted.append(row)
            accepted_by_track.setdefault(track, []).append(timestamp)
            if len(accepted) == quota:
                break
        if len(accepted) != quota:
            raise ValueError(f"quota missing for {sequence}: {len(accepted)}/{quota}")
        chosen.extend(dict(row) for row in accepted)
    return chosen


def sanitize_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Emit a row that carries no box, target, timing window, or raw path."""
    sequence_id = str(row["sequence_id"])
    return {
        "row_index": index,
        "row_sha256": hashlib.sha256(identity(row).encode()).hexdigest(),
        "sequence_id": sequence_id,
        "pool": "adaptation" if sequence_id in ADAPT_SEQUENCES else "audit",
        "delta_t_s": float(
            np.asarray(row["frame_timestamps_us"], dtype=np.float64)[-1]
            - np.asarray(row["frame_timestamps_us"], dtype=np.float64)[-2]
        )
        / 1_000_000.0,
    }


@dataclass
class AtomicDirectory:
    """Sibling staging writer with marker-verified, recoverable promotion.

    A pre-existing output is quarantined only after a complete sibling staging tree
    exists.  A failed promotion restores that output; neither path is selected by a
    glob or recursively removed without checking its generated sibling name.
    """

    target: Path
    force: bool = False
    config_identity: str | None = None
    source_identity: str | None = None
    staging: Path | None = None

    def _marker(self) -> dict[str, str | None]:
        return {
            "artifact": "object_event_v4_31",
            "owner": "e_jepa_ttc",
            "config_identity": self.config_identity,
            "source_identity": self.source_identity,
        }

    def _generated_sibling(self, path: Path, kind: str) -> bool:
        return path.parent == self.target.parent and path.name.startswith(
            f".{self.target.name}.{kind}-"
        )

    def _discard_generated_sibling(self, path: Path, kind: str) -> None:
        """Delete only a verified generated staging/quarantine sibling."""
        if not self._generated_sibling(path, kind):
            raise PermissionError(f"refusing to remove unowned path: {path}")
        shutil.rmtree(path)

    def __enter__(self) -> Path:
        if self.target.exists() and not self.force:
            raise FileExistsError(f"target exists: {self.target}; use --force")
        if self.target.exists():
            marker_path = self.target / OWNERSHIP_MARKER
            if not marker_path.is_file():
                raise PermissionError("--force requires the v4.31 ownership marker")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker != self._marker():
                raise PermissionError("--force ownership marker content differs")
        self.staging = self.target.with_name(f".{self.target.name}.staging-{uuid.uuid4().hex}")
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.staging is None:
            return
        if exc_type is not None:
            if self.staging.exists():
                self._discard_generated_sibling(self.staging, "staging")
            return
        (self.staging / OWNERSHIP_MARKER).write_text(strict_json(self._marker()), encoding="utf-8")
        backup = self.target.with_name(f".{self.target.name}.previous-{uuid.uuid4().hex}")
        if self.target.exists():
            os.replace(self.target, backup)
        try:
            os.replace(self.staging, self.target)
        except Exception as promotion_error:
            if backup.exists():
                try:
                    os.replace(backup, self.target)
                except Exception as restore_error:
                    raise RuntimeError(
                        f"promotion failed and backup remains at {backup}"
                    ) from restore_error
            raise promotion_error
        if backup.exists():
            self._discard_generated_sibling(backup, "previous")

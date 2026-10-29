#!/usr/bin/env python
"""Migrate legacy eAP JEPA run metadata to the v4 artifact contract.

This command consumes the real ``metrics.json`` emitted by the trainer.  It is
an explicit migration: dry-run never mutates a run, and a failed old run is
never relabelled as a completed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MIGRATION_VERSION = "jepa_pretrain_run_v4"
LEGACY_ARTIFACT_TYPES = {
    "eap_geo_on_demand_pretraining_v1",
    "eap_ssl_on_demand_pretraining_v1",
    "eap_geo_v2_on_demand_pretraining_v1",
}
SAMPLING_PROVENANCE_FIELDS = (
    "uses_ttc_for_sampling",
    "uses_boxes_for_sampling",
    "uses_category_for_sampling",
    "uses_depth_for_sampling",
    "uses_masks_for_sampling",
    "uses_3d_for_sampling",
    "uses_future_labels_for_sampling",
)


def _add_sampling_provenance(payload: dict[str, Any]) -> None:
    """Upgrade legacy artifacts with explicit, auditable sampling flags."""

    regime = str(payload.get("pretraining_regime", ""))
    config = payload.get("trainer_config", {})
    if not isinstance(config, dict):
        config = {}
    provenance = payload.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Legacy provenance must be an object.")
    geometry = regime in {"eap_geo", "eap_geo_v2"}
    version = payload.get(
        "geometry_target_version",
        config.get("geometry_target_version", provenance.get("geometry_target_version", "v1")),
    )
    strategy = payload.get(
        "geometry_sampling_strategy",
        config.get(
            "geometry_sampling_strategy",
            provenance.get("geometry_sampling_strategy", "nearest"),
        ),
    )
    inferred = {
        "uses_ttc_for_sampling": False,
        "uses_boxes_for_sampling": geometry,
        "uses_category_for_sampling": (
            geometry and version == "v2" and strategy == "balanced_tracks"
        ),
        "uses_depth_for_sampling": geometry and strategy == "nearest",
        "uses_masks_for_sampling": False,
        "uses_3d_for_sampling": geometry,
        "uses_future_labels_for_sampling": False,
    }
    for field in SAMPLING_PROVENANCE_FIELDS:
        value = provenance.setdefault(field, inferred[field])
        if type(value) is not bool:
            raise ValueError(f"Invalid non-boolean sampling provenance field: {field}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("artifact_sha256", None)
    encoded = json.dumps(
        clean,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _upgrade_mapping(payload: dict[str, Any], *, artifact: bool) -> None:
    """Apply the historical Geo2 field migration used by unit tests and runs."""

    config = payload.get("trainer_config", {})
    provenance = payload.get("provenance", {})
    version = payload.get(
        "geometry_target_version",
        config.get("geometry_target_version", provenance.get("geometry_target_version")),
    )
    strategy = payload.get(
        "geometry_sampling_strategy",
        config.get("geometry_sampling_strategy", provenance.get("geometry_sampling_strategy")),
    )
    if version != "v2" or strategy != "balanced_tracks":
        raise ValueError(
            "Refusing Geo2 provenance migration without v2 targets and balanced_tracks."
        )
    payload["pretraining_regime"] = "eap_geo_v2"
    if artifact:
        payload["artifact_type"] = "eap_geo_v2_on_demand_pretraining_v1"
    payload["uses_labels_for_window_sampling"] = True
    provenance = payload.setdefault("provenance", {})
    provenance.update(
        {
            "geometry_target_version": "v2",
            "geometry_sampling_strategy": "balanced_tracks",
            "uses_labels_for_window_sampling": True,
            "uses_balanced_track_sampling": True,
            "uses_visibility_targets": True,
            "uses_collision_corridor_targets": True,
            "uses_object_roi": False,
        }
    )
    fingerprint_payload = {
        "pretraining_dataset_id": payload.get("pretraining_dataset_id"),
        "pretraining_regime": payload["pretraining_regime"],
        "geometry_target_version": "v2",
        "geometry_sampling_strategy": "balanced_tracks",
        "source_seed": payload.get("source_seed", payload.get("trainer_config", {}).get("seed")),
        "split_artifact_sha256": payload.get("split_artifact_sha256"),
        "inventory_artifact_sha256": payload.get("inventory_artifact_sha256"),
        "trainer_config": payload.get("trainer_config"),
    }
    payload["run_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    _add_sampling_provenance(payload)


def _normalise_legacy_payload(source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    artifact_type = str(source.get("artifact_type", ""))
    if artifact_type not in LEGACY_ARTIFACT_TYPES:
        raise ValueError(
            f"Unsupported pretraining artifact type {artifact_type!r}; "
            "migration is explicit and does not guess arbitrary JSON."
        )
    payload = dict(source)
    changes: list[str] = []
    regime = str(payload.get("pretraining_regime", ""))
    if regime == "eap_geo" or artifact_type.startswith("eap_geo"):
        if (
            payload.get("geometry_target_version") == "v2"
            or payload.get("trainer_config", {}).get("geometry_target_version") == "v2"
        ):
            _upgrade_mapping(payload, artifact=True)
            payload["artifact_type"] = "eap_geo_v2_pretraining_run_v4"
            changes.append("geo2 provenance promoted to eap_geo_v2")
        else:
            payload["pretraining_regime"] = "eap_geo"
            payload["artifact_type"] = "eap_geo_pretraining_run_v4"
    else:
        payload["pretraining_regime"] = "eap_ssl"
        payload["artifact_type"] = "eap_ssl_pretraining_run_v4"
    _add_sampling_provenance(payload)
    payload["schema_version"] = "v4"
    payload["artifact_contract"] = "schemas/jepa_pretrain_run_v4.schema.json"
    payload["migration_version"] = MIGRATION_VERSION
    payload.setdefault("evidence_type", "legacy_artifact_migration")
    payload.setdefault("code_commit", payload.get("git_commit", "unknown"))
    payload.setdefault("protocol_version", "legacy_unversioned")
    payload.setdefault("protocol_sha256", "unknown")
    payload.setdefault("created_at", payload.get("start_time", "unknown"))
    changes.extend(
        [
            "canonical metrics.json artifact consumed",
            "schema_version added",
            "artifact_contract added",
        ]
    )
    return payload, changes


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def migrate(
    input_artifact: str | Path,
    output_artifact: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate one real trainer artifact and return a provenance report."""

    source_path = Path(input_artifact)
    if source_path.is_dir():
        candidates = (source_path / "metrics.json", source_path / "summary.json")
        source_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not source_path.is_file():
        raise FileNotFoundError(f"Pretraining artifact is missing: {source_path}")
    source_sha256 = _sha256(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise TypeError("Pretraining artifact must contain a JSON object.")
    migrated, changes = _normalise_legacy_payload(source)
    migrated["source_sha256"] = source_sha256
    migrated["migration_changes"] = changes
    migrated["artifact_sha256"] = _canonical_digest(migrated)
    destination = Path(output_artifact) if output_artifact is not None else source_path
    report = {
        "artifact_type": "eap_jepa_artifact_migration_v4",
        "migration_version": MIGRATION_VERSION,
        "input_artifact": source_path.as_posix(),
        "output_artifact": destination.as_posix(),
        "source_sha256": source_sha256,
        "output_sha256": hashlib.sha256(
            (json.dumps(migrated, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
        ).hexdigest(),
        "dry_run": dry_run,
        "changes": changes,
        "status": "planned" if dry_run else "migrated",
    }
    if not dry_run:
        _atomic_json_write(destination, migrated)
        report_path = destination.parent / "artifact_migration_v4.json"
        _atomic_json_write(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact", required=True)
    parser.add_argument("--output-artifact")
    parser.add_argument(
        "--output-dir",
        help="Deprecated directory form; the input artifact is resolved inside it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    input_artifact = args.input_artifact
    if args.output_dir:
        input_artifact = str(Path(args.output_dir) / args.input_artifact)
    report = migrate(
        input_artifact,
        args.output_artifact,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

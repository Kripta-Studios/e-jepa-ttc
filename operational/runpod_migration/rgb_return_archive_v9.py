"""Global RGB return overlay preserving narrowly-scoped V9 compatibility evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import rgb_return_archive as v8

V9_SCHEMA = "ejepa_rgb_legacy_decoded_proof_compatibility_v9"
V9_POLICY = "NEW_EXACTLY_ONE; LEGACY_ALL_MEMBERS_CLOSED_THEN_FIRST_DECODED"
V9_ROTATIONS = (2, 3)


def _source_by_hash(sources: Sequence[Path], expected: object, name: str) -> Path:
    matches = [
        path.resolve(strict=True)
        for path in sources
        if path.name == name and v8.sha256(path.resolve(strict=True)) == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"V9 source binding must resolve exactly once: {name}")
    return matches[0]


def _validate_v9(
    downstream: Path, operational_sources: Sequence[Path]
) -> tuple[Path, Path]:
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V9.json"
    value = v8.read_json(sidecar)
    source = _source_by_hash(
        operational_sources, value.get("source_sha256"), "runpod_rgb_context_v9.py"
    )
    v6_source = _source_by_hash(
        operational_sources, value.get("context_v6_sha256"), "runpod_rgb_context_v6.py"
    )
    parent_version = value.get("parent_context_version")
    parent_name = (
        "runpod_rgb_context_v7.py"
        if parent_version == 7
        else "runpod_rgb_context_v8.py"
        if parent_version == 8
        else ""
    )
    parent = _source_by_hash(
        operational_sources, value.get("parent_context_sha256"), parent_name
    )
    if (
        value.get("schema") != V9_SCHEMA
        or value.get("status") != "PASS"
        or v8.sha256(v6_source) != v8.RGB_CONTEXT_V6_SHA256
        or parent.name != parent_name
        or value.get("policy") != V9_POLICY
        or value.get("original_markers_rewritten") is not False
        or value.get("optimizer_updates_inferred") is not False
    ):
        raise ValueError("RGB V9 legacy-proof compatibility binding changed")
    return sidecar, source


def collect_rgb_return(
    downstream_runs: Mapping[int, Path],
    a5_runs: Mapping[int, Path],
    c2f_runs: Mapping[int, Path],
    *,
    coverage_receipt: Path,
    authorization_root: Path,
    operational_sources: Sequence[Path],
    rotations: Sequence[int] = v8.ROTATIONS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Collect the full campaign, requiring V9 exactly for legacy R2/R3."""
    selected = tuple(rotations)
    if selected != v8.ROTATIONS:
        raise ValueError("The V9 overlay is global-only and requires rotations [2,3,4]")
    manifest, records = v8.collect_rgb_return(
        downstream_runs,
        a5_runs,
        c2f_runs,
        coverage_receipt=coverage_receipt,
        authorization_root=authorization_root,
        operational_sources=operational_sources,
        rotations=selected,
    )
    for rotation in V9_ROTATIONS:
        downstream = downstream_runs[rotation].resolve(strict=True)
        sidecar, _ = _validate_v9(downstream, operational_sources)
        v8._add_record(  # noqa: SLF001 - version overlay extends the closed manifest
            records,
            f"roots/r{rotation}_downstream/{sidecar.name}",
            sidecar,
            downstream,
        )
    r4_v9 = downstream_runs[4] / "RGB_OPERATIONAL_RECOVERY_V9.json"
    if r4_v9.exists():
        raise ValueError("R4 must retain its frozen V8 chain in the global return")
    manifest["active_rgb_context_chain_by_rotation"] = {
        "2": ["V5", "V6", "V7", "V8", "V9"],
        "3": ["V5", "V6", "V7", "V9"],
        "4": ["V5", "V6", "V7", "V8"],
    }
    manifest["operational_v9_rotations"] = list(V9_ROTATIONS)
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["files"] = records
    return manifest, records


def export_if_ready(
    downstream_runs: Mapping[int, Path],
    a5_runs: Mapping[int, Path],
    c2f_runs: Mapping[int, Path],
    *,
    coverage_receipt: Path,
    authorization_root: Path,
    operational_sources: Sequence[Path],
    output: Path,
    rotations: Sequence[int] = v8.ROTATIONS,
    writer_probe: Callable[[Sequence[Path]], list[dict[str, object]]] = v8._writer_fds_linux,
) -> dict[str, Any]:
    """Publish only a quiescent global V9 closure; never mutate a run."""
    selected = tuple(rotations)
    roots = [
        mapping[rotation]
        for rotation in selected
        for mapping in (downstream_runs, a5_runs, c2f_runs)
    ]
    try:
        resolved_output = output.resolve()
        protected = [*roots, authorization_root]
        if any(resolved_output.is_relative_to(root.resolve(strict=True)) for root in protected):
            raise ValueError("RGB archive output must remain outside every live/input root")
        writers = writer_probe(roots)
        if writers:
            return {"status": "NOT_READY", "reason": "ACTIVE_WRITER", "writers": writers}
        manifest, records = collect_rgb_return(
            downstream_runs,
            a5_runs,
            c2f_runs,
            coverage_receipt=coverage_receipt,
            authorization_root=authorization_root,
            operational_sources=operational_sources,
            rotations=selected,
        )
        writers = writer_probe(roots)
        if writers:
            return {"status": "NOT_READY", "reason": "ACTIVE_WRITER", "writers": writers}
        return v8.publish_rgb_return(manifest, records, output)
    except (FileNotFoundError, ValueError) as error:
        return {"status": "NOT_READY", "reason": f"{type(error).__name__}: {error}"}


__all__ = ["V9_POLICY", "V9_ROTATIONS", "V9_SCHEMA", "collect_rgb_return", "export_if_ready"]

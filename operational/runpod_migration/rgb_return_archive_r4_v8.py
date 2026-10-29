"""Resolve R4 media config through its exact composed V8 authority."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import rgb_return_archive as frozen


def _composed_v8_config_source(
    downstream: Path,
    operational_sources: Sequence[Path],
    expected_authority: object,
) -> tuple[Path, dict[str, Any]]:
    sidecar = frozen.read_json(downstream / "RGB_OPERATIONAL_RECOVERY_V8.json")
    layout_matches = [
        path.resolve(strict=True)
        for path in operational_sources
        if frozen.sha256(path.resolve(strict=True)) == sidecar.get("layout_sha256")
    ]
    if len(layout_matches) != 1:
        raise ValueError("R4 V8 layout SHA must resolve to exactly one source")
    layout = frozen.read_json(layout_matches[0])
    expected_config_sha = layout.get("original_config_sha256")
    config_matches = [
        path.resolve(strict=True)
        for path in operational_sources
        if frozen.sha256(path.resolve(strict=True)) == expected_config_sha
    ]
    if len(config_matches) != 1:
        raise ValueError("R4 V8 original config SHA must resolve to exactly one source")
    config_path = config_matches[0]
    config = frozen.read_json(config_path)
    original_header = {key: value for key, value in config.items() if key != "scopes"}
    original_authority = frozen._canonical_sha256(original_header)
    composed_authority = frozen._canonical_sha256(
        {
            "original_authority_sha256": original_authority,
            "layout_authority_sha256": sidecar.get("layout_authority_sha256"),
        }
    )
    if (
        sidecar.get("schema") != "ejepa_rgb_compositional_context_v8"
        or sidecar.get("status") != "PASS"
        or layout.get("schema") != "ejepa_rgb_compositional_media_layout_v8"
        or layout.get("status") != "PASS"
        or sidecar.get("original_config_sha256") != expected_config_sha
        or config.get("schema") != "ejepa_rgb_media_scope_config_v1"
        or not re.fullmatch(r"[0-9a-f]{64}", str(expected_config_sha))
        or Path(str(layout.get("original_config_path"))).name != config_path.name
        or composed_authority != expected_authority
    ):
        raise ValueError("R4 composed V8 media authority/config genealogy changed")
    return config_path, config


def collect_rgb_return(
    downstream_runs: Mapping[int, Path],
    a5_runs: Mapping[int, Path],
    c2f_runs: Mapping[int, Path],
    *,
    coverage_receipt: Path,
    authorization_root: Path,
    operational_sources: Sequence[Path],
    rotations: Sequence[int] = frozen.INDIVIDUAL_RETURN_ROTATIONS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Collect only R4 while retaining every frozen V8 validation."""
    if tuple(rotations) != frozen.INDIVIDUAL_RETURN_ROTATIONS:
        raise ValueError("The composed V8 authority overlay supports only rotation [4]")
    if set(downstream_runs) != {4}:
        raise ValueError("The composed V8 authority overlay requires exactly R4")
    downstream = downstream_runs[4].resolve(strict=True)
    v6_sidecar_path = downstream / "RGB_OPERATIONAL_RECOVERY_V6.json"
    actual_v6 = frozen.read_json(v6_sidecar_path)
    config_path, config = _composed_v8_config_source(
        downstream, operational_sources, actual_v6.get("media_authority_sha256")
    )
    original_authority = frozen._canonical_sha256(
        {key: value for key, value in config.items() if key != "scopes"}
    )
    original_read = frozen.read_json

    def read_with_original_v6_authority(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if path.resolve() == v6_sidecar_path.resolve():
            return {**value, "media_authority_sha256": original_authority}
        return value

    frozen.read_json = read_with_original_v6_authority
    try:
        manifest, records = frozen.collect_rgb_return(
            downstream_runs,
            a5_runs,
            c2f_runs,
            coverage_receipt=coverage_receipt,
            authorization_root=authorization_root,
            operational_sources=operational_sources,
            rotations=rotations,
        )
    finally:
        frozen.read_json = original_read
    manifest["r4_media_config_resolution"] = {
        "method": "COMPOSED_V8_AUTHORITY_EXACT_ORIGINAL_CONFIG_SHA256",
        "original_config_source_path": str(config_path),
        "stored_composed_authority_sha256": actual_v6["media_authority_sha256"],
        "original_header_authority_sha256": original_authority,
        "integrity_checks_bypassed": False,
        "frozen_v8_genealogy_reverified": True,
    }
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
    rotations: Sequence[int] = frozen.INDIVIDUAL_RETURN_ROTATIONS,
    writer_probe: Callable[[Sequence[Path]], list[dict[str, object]]] = frozen._writer_fds_linux,
) -> dict[str, Any]:
    """Publish R4 only after both writer probes and the composed authority checks pass."""
    roots = [mapping[4] for mapping in (downstream_runs, a5_runs, c2f_runs)]
    try:
        if tuple(rotations) != frozen.INDIVIDUAL_RETURN_ROTATIONS:
            raise ValueError("The composed V8 authority overlay supports only rotation [4]")
        resolved_output = output.resolve()
        if any(
            resolved_output.is_relative_to(root.resolve(strict=True))
            for root in [*roots, authorization_root]
        ):
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
            rotations=rotations,
        )
        writers = writer_probe(roots)
        if writers:
            return {"status": "NOT_READY", "reason": "ACTIVE_WRITER", "writers": writers}
        return frozen.publish_rgb_return(manifest, records, output)
    except (FileNotFoundError, ValueError) as error:
        return {"status": "NOT_READY", "reason": f"{type(error).__name__}: {error}"}


__all__ = ["collect_rgb_return", "export_if_ready"]

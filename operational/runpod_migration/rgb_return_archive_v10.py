"""Global R2/R4 compositional V8 plus R2/R3 V9 archive overlay."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import rgb_return_archive as frozen
from . import rgb_return_archive_r4_v8 as authority
from . import rgb_return_archive_v9 as v9

_v9_collect = v9.collect_rgb_return


def _validate_r2_v8(
    downstream: Path,
    rotation: int,
    sources: Sequence[Path],
    config_path: Path,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    if rotation != 2:
        raise ValueError("The global V10 R2 validator received another rotation")
    sidecar = downstream / "RGB_OPERATIONAL_RECOVERY_V8.json"
    value = frozen.read_json(sidecar)
    source = frozen._source_by_sha256(sources, value.get("source_sha256"), label="V8")
    v7_source = frozen._source_by_sha256(
        sources, value.get("context_v7_sha256"), label="V7"
    )
    layout = frozen._source_by_sha256(sources, value.get("layout_sha256"), label="layout")
    layout_value = frozen.read_json(layout)
    if (
        source.name != "runpod_rgb_context_v8.py"
        or v7_source.name != "runpod_rgb_context_v7.py"
        or value.get("schema") != "ejepa_rgb_compositional_context_v8"
        or value.get("status") != "PASS"
        or value.get("original_config_sha256") != frozen.sha256(config_path)
        or value.get("migration_binding_sha256")
        != frozen.sha256(downstream / "MIGRATION_BINDING.json")
        or value.get("allowed_physical_roots")
        != ["/opt/ejepa_rgb_frames", "/workspace/ejepa_rgb_frames"]
        or value.get("physical_sidecar_roles_relabelled") is not False
        or value.get("existing_r3_admissions_changed") is not False
        or value.get("optimizer_updates_inferred") is not False
        or layout_value.get("schema") != "ejepa_rgb_compositional_media_layout_v8"
        or layout_value.get("status") != "PASS"
        or layout_value.get("original_config_sha256") != frozen.sha256(config_path)
        or Path(str(layout_value.get("original_config_path"))).resolve(strict=True)
        != config_path.resolve(strict=True)
        or layout_value.get("allowed_physical_roots")
        != ["/opt/ejepa_rgb_frames", "/workspace/ejepa_rgb_frames"]
        or value.get("layout_authority_sha256")
        != frozen._canonical_sha256(
            {key: item for key, item in layout_value.items() if key != "scopes"}
        )
    ):
        raise ValueError("R2 V8 sidecar/layout binding changed")
    scopes = {
        (item.get("rotation"), item.get("role")): item
        for item in layout_value.get("scopes", [])
        if isinstance(item, dict)
    }
    expected = {
        "refiner": ("REUSE", 3, "generation", "/opt/ejepa_rgb_frames"),
        "generation": ("DIRECT", 2, "generation", "/workspace/ejepa_rgb_frames"),
    }
    original_config = frozen.read_json(config_path)
    original_authority = frozen._canonical_sha256(
        {key: item for key, item in original_config.items() if key != "scopes"}
    )
    combined = frozen._canonical_sha256(
        {
            "original_authority_sha256": original_authority,
            "layout_authority_sha256": value["layout_authority_sha256"],
        }
    )
    evidence: list[dict[str, Any]] = []
    for role, (mode, physical_rotation, physical_role, physical_root) in expected.items():
        item = scopes.get((2, role))
        if not isinstance(item, dict) or (
            item.get("mode"),
            item.get("physical_rotation"),
            item.get("physical_role"),
            item.get("physical_root"),
        ) != (mode, physical_rotation, physical_role, physical_root):
            raise ValueError(f"R2/{role} V8 physical genealogy changed")
        names = ["target_plan", "source_plan", "source_manifest"]
        if mode == "REUSE":
            names.append("reuse_proof")
        paths: dict[str, Path] = {}
        for name in names:
            path = Path(str(item.get(f"{name}_path"))).resolve(strict=True)
            if item.get(f"{name}_sha256") != frozen.sha256(path):
                raise ValueError(f"R2/{role} V8 {name} bytes changed")
            paths[name] = path
        target_plan = frozen.read_json(paths["target_plan"])
        source_plan = frozen.read_json(paths["source_plan"])
        manifest = frozen.read_json(paths["source_manifest"])
        for label, plan in (("target", target_plan), ("source", source_plan)):
            payload = {key: item for key, item in plan.items() if key != "receipt_sha256"}
            if (
                plan.get("schema") != "ejepa_rgb_frame_supplement_plan_v1"
                or plan.get("status") != "PLANNED_HEADERS_VERIFIED"
                or plan.get("receipt_sha256") != frozen._canonical_sha256(payload)
            ):
                raise ValueError(f"R2/{role} V8 {label} plan receipt changed")
        manifest_payload = {
            key: item for key, item in manifest.items() if key != "receipt_sha256"
        }
        if (
            manifest.get("schema") != "ejepa_rgb_frame_supplement_v1"
            or manifest.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
            or manifest.get("plan_sha256") != frozen.sha256(paths["source_plan"])
            or manifest.get("receipt_sha256")
            != frozen._canonical_sha256(manifest_payload)
            or manifest.get("payload_recompressed") is not False
            or manifest.get("decoded_pixel_parity") is not True
        ):
            raise ValueError(f"R2/{role} V8 physical manifest changed")
        admission_path = downstream / f"RGB_MEDIA_SCOPE_ADMISSION_R2_{role}.json"
        admission = frozen.read_json(admission_path)
        physical = admission.get("physical_admission")
        if (
            admission.get("schema") != "ejepa_rgb_compositional_scope_admission_v8"
            or admission.get("status") != "PASS"
            or admission.get("rotation") != 2
            or admission.get("role") != role
            or admission.get("target_plan_sha256") != frozen.sha256(paths["target_plan"])
            or admission.get("physical_rotation") != physical_rotation
            or admission.get("physical_role") != physical_role
            or admission.get("physical_root") != physical_root
            or admission.get("composition_mode") != mode
            or admission.get("authority_sha256") != combined
            or admission.get("physical_sidecar_roles_relabelled") is not False
            or admission.get("optimizer_updates_inferred") is not False
            or not isinstance(physical, dict)
            or physical.get("schema") != "ejepa_rgb_media_scope_admission_v1"
            or physical.get("status") != "PASS"
            or physical.get("rotation") != physical_rotation
            or physical.get("role") != physical_role
            or physical.get("plan_sha256") != frozen.sha256(paths["source_plan"])
            or physical.get("manifest_sha256") != frozen.sha256(paths["source_manifest"])
            or physical.get("media_root") != physical_root
            or physical.get("model_inference_before_admission") is not False
            or physical.get("optimizer_updates_inferred") is not False
        ):
            raise ValueError(f"R2/{role} V8 admission changed")
        receipts: list[Path] = []
        for record in manifest.get("tars", []):
            relative = frozen._safe_relative(str(record.get("relative")))
            receipt = Path(physical_root).joinpath(*relative.parts).with_suffix(
                Path(relative.name).suffix + ".json"
            ).resolve(strict=True)
            record_payload = {
                key: item for key, item in record.items() if key != "receipt_sha256"
            }
            members = record.get("members")
            if (
                frozen.read_json(receipt) != record
                or record.get("schema") != "ejepa_rgb_sparse_tar_receipt_v1"
                or record.get("status") != "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
                or record.get("receipt_sha256")
                != frozen._canonical_sha256(record_payload)
                or not isinstance(members, list)
                or not members
                or any(
                    not isinstance(member, dict)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(member.get("compressed_payload_sha256", "")),
                    )
                    for member in members
                )
            ):
                raise ValueError("R2 V8 physical receipt differs from manifest")
            receipts.append(receipt)
        if mode == "DIRECT":
            if (
                admission.get("reuse_proof_sha256") is not None
                or admission.get("reuse_proof_receipt_sha256") is not None
                or frozen.sha256(paths["target_plan"]) != frozen.sha256(paths["source_plan"])
            ):
                raise ValueError("R2 generation DIRECT genealogy changed")
        else:
            proof = frozen.read_json(paths["reuse_proof"])
            payload = {key: item for key, item in proof.items() if key != "receipt_sha256"}
            if (
                proof.get("schema") != "ejepa_rgb_media_scope_reuse_proof_v1"
                or proof.get("status") != "PASS"
                or proof.get("receipt_sha256") != frozen._canonical_sha256(payload)
                or admission.get("reuse_proof_sha256") != frozen.sha256(paths["reuse_proof"])
                or admission.get("reuse_proof_receipt_sha256") != proof.get("receipt_sha256")
                or proof.get("target", {}).get("rotation") != 2
                or proof.get("target", {}).get("role") != role
                or proof.get("physical", {}).get("rotation") != physical_rotation
                or proof.get("physical", {}).get("role") != physical_role
                or proof.get("target", {}).get("plan_sha256")
                != frozen.sha256(paths["target_plan"])
                or proof.get("physical", {}).get("plan_sha256")
                != frozen.sha256(paths["source_plan"])
                or proof.get("physical", {}).get("manifest_sha256")
                != frozen.sha256(paths["source_manifest"])
                or proof.get("same_exact_member_names_and_payload_bytes") is not True
                or proof.get("same_logical_tar_paths") is not True
                or proof.get("physical_sidecar_roles_relabelled") is not False
                or proof.get("optimizer_updates") != 0
            ):
                raise ValueError("R2 refiner REUSE genealogy changed")
        evidence.append(
            {
                "role": role,
                "admission": admission_path,
                "paths": paths,
                "physical_root": physical_root,
                "physical_rotation": physical_rotation,
                "physical_role": physical_role,
                "mode": mode,
                "tar_receipts": receipts,
            }
        )
    return sidecar, source, layout, evidence


def collect_rgb_return(
    downstream_runs: Mapping[int, Path],
    a5_runs: Mapping[int, Path],
    c2f_runs: Mapping[int, Path],
    *,
    coverage_receipt: Path,
    authorization_root: Path,
    operational_sources: Sequence[Path],
    rotations: Sequence[int] = frozen.ROTATIONS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if tuple(rotations) != frozen.ROTATIONS:
        raise ValueError("V10 is a global [2,3,4] closure only")
    original_validator = frozen._validate_rgb_operational_recovery_v8
    original_read = frozen.read_json
    original_authorities: dict[Path, str] = {}
    for rotation in (2, 4):
        downstream = downstream_runs[rotation].resolve(strict=True)
        actual = original_read(downstream / "RGB_OPERATIONAL_RECOVERY_V6.json")
        _, config = authority._composed_v8_config_source(
            downstream, operational_sources, actual.get("media_authority_sha256")
        )
        original_authorities[downstream] = frozen._canonical_sha256(
            {key: item for key, item in config.items() if key != "scopes"}
        )

    def read(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if path.name == "RGB_OPERATIONAL_RECOVERY_V6.json":
            parent = path.parent.resolve()
            if parent in original_authorities:
                return {**value, "media_authority_sha256": original_authorities[parent]}
        return value

    def validate(
        downstream: Path, rotation: int, sources: Sequence[Path], config: Path
    ) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
        if rotation == 2:
            return _validate_r2_v8(downstream, rotation, sources, config)
        return original_validator(downstream, rotation, sources, config)

    frozen.read_json = read
    frozen._validate_rgb_operational_recovery_v8 = validate
    try:
        manifest, records = _v9_collect(
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
        frozen._validate_rgb_operational_recovery_v8 = original_validator
    if manifest.get("compositional_v8_rotations") != [2, 4]:
        raise ValueError("Global V10 archive did not preserve both V8 rotations")
    manifest["global_context_closure"] = "R2_V8_V9__R3_V7_V9__R4_V8"
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
    rotations: Sequence[int] = frozen.ROTATIONS,
    writer_probe: Callable[[Sequence[Path]], list[dict[str, object]]] = frozen._writer_fds_linux,
) -> dict[str, Any]:
    original = v9.collect_rgb_return
    v9.collect_rgb_return = collect_rgb_return
    try:
        return v9.export_if_ready(
            downstream_runs,
            a5_runs,
            c2f_runs,
            coverage_receipt=coverage_receipt,
            authorization_root=authorization_root,
            operational_sources=operational_sources,
            output=output,
            rotations=rotations,
            writer_probe=writer_probe,
        )
    finally:
        v9.collect_rgb_return = original


__all__ = ["collect_rgb_return", "export_if_ready"]

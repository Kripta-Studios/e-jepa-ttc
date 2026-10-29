"""Pure construction of detection-campaign runtime lineage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from e_jepa_ttc.detection.campaign_training import CampaignConfig, canonical_sha256
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol

_BASE_RUNTIME_RELATIVE_PATHS = (
    "scripts/train_detection_campaign.py",
    "data/manifests/eap_detection_public_files_v1.json",
    "scripts/infer_detection_campaign.py",
    "scripts/smoke_monodgp_portable.py",
    "src/e_jepa_ttc/__init__.py",
    "src/e_jepa_ttc/detection/__init__.py",
    "src/e_jepa_ttc/detection/campaign_training.py",
    "src/e_jepa_ttc/detection/campaign_preregistration.py",
    "src/e_jepa_ttc/detection/backbone_initialization.py",
    "src/e_jepa_ttc/detection/development_metrics.py",
    "src/e_jepa_ttc/detection/diagnostic_export.py",
    "src/e_jepa_ttc/detection/eap_rgb_data.py",
    "src/e_jepa_ttc/detection/evaluation_ap_fast.py",
    "src/e_jepa_ttc/detection/evaluation_protocol.py",
    "src/e_jepa_ttc/detection/evaluation_ground_truth.py",
    "src/e_jepa_ttc/detection/evaluation_v2.py",
    "src/e_jepa_ttc/detection/geometry_contract.py",
    "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
    "src/e_jepa_ttc/detection/portable_attention.py",
    "src/e_jepa_ttc/detection/velocity_extension.py",
)


def base_runtime_relative_paths() -> tuple[str, ...]:
    """Return the sole authoritative ordinary runtime source closure."""
    return _BASE_RUNTIME_RELATIVE_PATHS


def optimized_runtime_relative_paths(performance_config_relative: str) -> tuple[str, ...]:
    """Extend the ordinary closure with the indexed runner dependencies."""
    if not performance_config_relative or Path(performance_config_relative).is_absolute():
        raise ValueError("Performance config path must be nonempty and repository-relative")
    return _BASE_RUNTIME_RELATIVE_PATHS + (
        "scripts/train_detection_campaign_optimized.py",
        "src/e_jepa_ttc/detection/indexed_rgb_data.py",
        "src/e_jepa_ttc/detection/indexed_rgb_store.py",
        "src/e_jepa_ttc/detection/optimized_campaign.py",
        performance_config_relative,
    )


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _hash_mapping(value: Mapping[str, Any], field: str) -> dict[str, str]:
    if not value:
        raise ValueError(f"{field} must be nonempty")
    return {str(path): _digest(digest, f"{field}.{path}") for path, digest in value.items()}


def validate_source_receipt(value: Mapping[str, Any]) -> dict[str, object]:
    """Reject incomplete or invented portable-source receipts."""
    expected = {
        "upstream_revision",
        "variant",
        "patch_sha256",
        "patched_file_sha256",
        "complete_source_file_sha256",
        "external_extension_used",
        "automatic_pretrained_download",
    }
    if set(value) != expected:
        raise ValueError("Source receipt fields differ from the portable source contract")
    if not isinstance(value["upstream_revision"], str) or not value["upstream_revision"]:
        raise ValueError("Source receipt upstream_revision must be nonempty")
    if not isinstance(value["variant"], str) or not value["variant"]:
        raise ValueError("Source receipt variant must be nonempty")
    if value["external_extension_used"] is not False:
        raise ValueError("External detector extensions are forbidden")
    if value["automatic_pretrained_download"] is not False:
        raise ValueError("Automatic pretrained downloads are forbidden")
    return {
        "upstream_revision": value["upstream_revision"],
        "variant": value["variant"],
        "patch_sha256": _digest(value["patch_sha256"], "source.patch_sha256"),
        "patched_file_sha256": _hash_mapping(
            value["patched_file_sha256"], "source.patched_file_sha256"
        ),
        "complete_source_file_sha256": _hash_mapping(
            value["complete_source_file_sha256"],
            "source.complete_source_file_sha256",
        ),
        "external_extension_used": False,
        "automatic_pretrained_download": False,
    }


def build_runtime_preregistration(
    *,
    config: CampaignConfig,
    protocol: EvaluationProtocol,
    source_receipt: Mapping[str, Any],
    effective_model: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    portable_source_sha256: Mapping[str, str],
    dataset_media_sha256: str,
    media_admission_receipt_sha256: str,
    media_admission_population_sha256: str,
    runtime_extension: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build path-independent preregistration bytes without observing hardware."""
    source = validate_source_receipt(source_receipt)
    effective = dict(effective_model)
    code = _hash_mapping(code_sha256, "code_sha256")
    portable = _hash_mapping(portable_source_sha256, "portable_source_sha256")
    return {
        "schema": "eap_detection_campaign_runtime_preregistration_v1",
        "config": asdict(config),
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "source_receipt": source,
        "source_receipt_sha256": canonical_sha256(source),
        "effective_model": effective,
        "effective_model_sha256": canonical_sha256(effective),
        "code_sha256": code,
        "code_sha256_manifest": canonical_sha256(code),
        "portable_source_sha256": portable,
        "portable_source_sha256_manifest": canonical_sha256(portable),
        "dataset_media_sha256": dataset_media_sha256,
        "media_admission_receipt_sha256": media_admission_receipt_sha256,
        "media_admission_population_sha256": media_admission_population_sha256,
        "matmul_precision": config.optimization.matmul_precision,
        "runtime_extension": dict(runtime_extension or {}),
    }


def build_runtime_provenance(
    preregistration_file_sha256: str, prereg: Mapping[str, Any]
) -> dict[str, object]:
    """Derive the strict checkpoint provenance from persisted preregistration."""
    return {
        "runtime_preregistration_sha256": _digest(
            preregistration_file_sha256, "runtime_preregistration_sha256"
        ),
        "source_receipt_sha256": prereg["source_receipt_sha256"],
        "effective_model_sha256": prereg["effective_model_sha256"],
        "code_sha256_manifest": prereg["code_sha256_manifest"],
        "portable_source_sha256_manifest": prereg["portable_source_sha256_manifest"],
        "matmul_precision": prereg["matmul_precision"],
    }


__all__ = [
    "base_runtime_relative_paths",
    "build_runtime_preregistration",
    "build_runtime_provenance",
    "optimized_runtime_relative_paths",
    "validate_source_receipt",
]

"""Build optimized campaign lineage on CPU without constructing a detector."""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_detection_campaign as ordinary
import train_detection_campaign_optimized as optimized
from smoke_monodgp_portable import load_detector_config, prepare_source

from e_jepa_ttc.detection.campaign_preregistration import (
    base_runtime_relative_paths,
    build_runtime_preregistration,
    build_runtime_provenance,
    optimized_runtime_relative_paths,
)
from e_jepa_ttc.detection.campaign_training import (
    load_campaign_config,
    sha256_file,
    validate_campaign_manifest,
    validate_manifest_against_metadata,
)
from e_jepa_ttc.detection.optimized_campaign import PerformanceConfig


def build(config_path: Path, performance_path: Path, output: Path) -> dict[str, object]:
    """Write once-only CPU lineage with the production serialization contract."""
    if output.exists():
        raise FileExistsError(f"Use a new preregistration output directory: {output}")
    config = load_campaign_config(config_path)
    performance, performance_receipt = PerformanceConfig.load(performance_path)
    reader_path = ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py"
    store_path = ROOT / "src/e_jepa_ttc/detection/indexed_rgb_store.py"
    if sha256_file(reader_path) != performance.expected_indexed_reader_sha256:
        raise ValueError("Indexed reader source differs from performance sidecar")
    if sha256_file(store_path) != performance.expected_indexed_store_sha256:
        raise ValueError("Indexed store source differs from performance sidecar")
    manifest = validate_campaign_manifest(config)
    metadata = validate_manifest_against_metadata(manifest, config)
    admission = dict(metadata["media_admission"])
    admission.pop("data_member_sha256")
    protocol = ordinary.load_protocol(config.evaluation.protocol)
    _, _, extension = optimized._build_store_contract(config, performance)
    extension["performance_config"] = performance_receipt
    performance_relative = performance_path.resolve().relative_to(ROOT).as_posix()
    base_count = len(base_runtime_relative_paths())
    extra = tuple(
        ROOT / relative
        for relative in optimized_runtime_relative_paths(performance_relative)[base_count:]
    )
    output.mkdir(parents=True)
    try:
        code_sha256 = ordinary._snapshot(output, ordinary.campaign_owned_paths(extra))
        code_sha256.update(
            ordinary.snapshot_media_admission(output, Path(str(admission["receipt_path"])))
        )
        with tempfile.TemporaryDirectory(prefix="phase_ttc_prereg_") as temporary:
            temporary_root = Path(temporary)
            portable = temporary_root / "portable_source"
            source = prepare_source(
                Path(config.model.source_root),
                portable,
                temporary_root / "upstream.patch",
                predict_velocity=config.model.predict_velocity,
            )
            effective_model = load_detector_config(
                portable,
                config.model.group_num,
                predict_velocity=config.model.predict_velocity,
                velocity_loss_coef=config.model.velocity_loss_coef,
            )
            portable_sha256 = ordinary._snapshot_portable_source(portable, output)
        prereg = build_runtime_preregistration(
            config=config,
            protocol=protocol,
            source_receipt=source,
            effective_model=effective_model,
            code_sha256=code_sha256,
            portable_source_sha256=portable_sha256,
            dataset_media_sha256=str(manifest["source_media_sha256"]),
            media_admission_receipt_sha256=str(admission["receipt_sha256"]),
            media_admission_population_sha256=str(
                admission["train_token_population_sha256"]
            ),
            runtime_extension=extension,
        )
        ordinary.write_json(
            output / "config_resolved.json", asdict(config) | {"sha256": config.sha256}
        )
        ordinary.write_json(output / "runtime_preregistration.json", prereg)
        provenance = build_runtime_provenance(
            sha256_file(output / "runtime_preregistration.json"), prereg
        )
        receipt = {
            "schema": "eap_detection_campaign_cpu_preregistration_v1",
            "status": "PASS",
            "config_sha256": config.sha256,
            "runtime_preregistration_sha256": provenance[
                "runtime_preregistration_sha256"
            ],
            "runtime_provenance": provenance,
            "cuda_or_model_constructed": False,
            "preflight_builder_sha256": sha256_file(Path(__file__)),
        }
        ordinary.write_json(output / "preflight_receipt.json", receipt)
        return receipt
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--performance-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.performance_config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

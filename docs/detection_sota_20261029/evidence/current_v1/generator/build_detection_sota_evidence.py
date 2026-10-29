"""Build the small, public detection-development evidence bundle.

The bundle is an allow-list of receipts and result files.  It deliberately does
not copy checkpoints, datasets, RGB media, or entire output directories.  Large
JSON evidence is gzip-compressed with a fixed header timestamp so byte-for-byte
regeneration is possible.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "docs/detection_sota_20261029/evidence/current_v1"
RESULTS_MD = REPO_ROOT / "docs/detection_sota_20261029/GENERATED_RESULTS.md"
PROTOCOL_SHA256 = "10da30cf8f08247d48d975f3650cb02f1b43375fe0e841a5389dfb3b5d7b4a5e"
TRAINING_PRODUCER_COMMIT = "fcad2299979766f6d175e45b8016ea3b7a11a594"
S2_CODE_COMMIT = "88d607ba"


@dataclass(frozen=True)
class SourceSpec:
    source: str
    packaged: str
    gzip_compress: bool = False
    redact_local_paths: bool = False


SOURCES = (
    # S0 diagnostic, exact GT, and observer parity provenance.
    SourceSpec(
        "output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/DIAGNOSTICS.md",
        "s0/DIAGNOSTICS.md",
    ),
    SourceSpec(
        "output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/"
        "evaluation_protocol_v2.json",
        "s0/evaluation_protocol_v2.json",
    ),
    SourceSpec(
        "output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/"
        "ground_truth_v2.json",
        "s0/ground_truth_v2.json.gz",
        True,
    ),
    SourceSpec(
        "output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/diagnostics.json",
        "s0/diagnostics.json.gz",
        True,
    ),
    SourceSpec(
        "output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/"
        "legacy_recomputed.json",
        "s0/legacy_recomputed.json",
    ),
    SourceSpec(
        "output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/provenance.json",
        "s0/provenance.json",
        redact_local_paths=True,
    ),
    SourceSpec(
        "output/s0_controls/pilot_selection256_v2_r100/controls.json",
        "s0/controls.json.gz",
        True,
    ),
    SourceSpec(
        "output/s0_controls/pilot_selection256_v2_r100/CONTROLS.md",
        "s0/CONTROLS.md",
    ),
    SourceSpec(
        "output/s0_controls/pilot_selection256_v2_r100/preregistration.json",
        "s0/controls_preregistration.json",
    ),
    SourceSpec(
        "output/s0_observer/selection256_predictions.upstream.patch",
        "s0/observer.upstream.patch",
    ),
    # D4096 data coverage and population audit.  The 52 MB row ledger is
    # intentionally omitted; its derived audit and exact selected manifest are included.
    SourceSpec(
        "output/campaign_data/d4096_v4_final/d4096_coverage.json",
        "s1_data/d4096_coverage.json",
    ),
    SourceSpec(
        "output/campaign_data/d4096_v4_final/d4096_manifest.json",
        "s1_data/d4096_manifest.json.gz",
        True,
    ),
    SourceSpec(
        "output/campaign_data/d4096_v4_final/population_audit.json",
        "s1_data/population_audit.json.gz",
        True,
    ),
    SourceSpec(
        "output/campaign_data/d4096_v4_final/execution_provenance.json",
        "s1_data/execution_provenance.json",
    ),
    SourceSpec(
        "output/campaign_data/d4096_v4_final/receipt.json",
        "s1_data/receipt.json",
    ),
    # Actual data-read profile v3, including parity between cache configurations.
    SourceSpec(
        "output/campaign_data/read_profile_v3_final/summary.json",
        "s1_data/read_profile_summary.json.gz",
        True,
        True,
    ),
    SourceSpec(
        "output/campaign_data/read_profile_v3_final/receipt.json",
        "s1_data/read_profile_receipt.json",
    ),
    SourceSpec(
        "output/campaign_data/read_profile_v3_final/cache2.json",
        "s1_data/read_profile_cache2.json.gz",
        True,
    ),
    SourceSpec(
        "output/campaign_data/read_profile_v3_final/cache24.json",
        "s1_data/read_profile_cache24.json.gz",
        True,
    ),
    # Smoke 4 -> 8 admission and executed GPU profile.  Checkpoints are excluded.
    SourceSpec(
        "output/campaign_smoke_admission_v1/admission.json",
        "s1_smoke/admission.json",
    ),
    SourceSpec(
        "output/campaign_smoke_admission_v1/config.yaml",
        "s1_smoke/config.yaml",
    ),
    SourceSpec(
        "output/campaign_smoke_admission_v1/stage1/summary.json",
        "s1_smoke/stage1_summary.json",
        redact_local_paths=True,
    ),
    SourceSpec(
        "output/campaign_smoke_admission_v1/stage2/summary.json",
        "s1_smoke/stage2_summary.json",
        redact_local_paths=True,
    ),
    SourceSpec(
        "output/campaign_smoke_admission_v1/stage2_inference.json",
        "s1_smoke/stage2_inference.json",
    ),
    SourceSpec(
        "output/s1_profile/pilot_velocity_batch4.json",
        "s1_smoke/gpu_profile.json",
        redact_local_paths=True,
    ),
    # Frozen R0 receipts only.  Live metrics and checkpoints must not enter v1.
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v1/preflight_failure.json",
        "s1_r0/v1_preflight_failure.json",
    ),
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/config.yaml",
        "s1_r0/v2_config.yaml",
    ),
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/producer_receipt.json",
        "s1_r0/v2_producer_receipt.json",
    ),
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/stage1/config_resolved.json",
        "s1_r0/v2_stage1_config_resolved.json",
    ),
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/stage1/preregistration.json",
        "s1_r0/v2_stage1_preregistration.json",
        redact_local_paths=True,
    ),
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/stage1/runtime_preregistration.json",
        "s1_r0/v2_stage1_runtime_preregistration.json",
        redact_local_paths=True,
    ),
    # S5 real label-free replay with native PHASE bounds.
    SourceSpec(
        "output/s5_replay/temporal8_gpu_max2_chunk1_phase_bounds_v1.json",
        "s5/phase_bounds_real_replay.json",
        redact_local_paths=True,
    ),
    SourceSpec(
        "output/s5_replay/temporal8_tokens_v1.json",
        "s5/replay_tokens.json",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _load_json(path: str) -> Any:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _gzip_deterministic(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as stream:
        stream.write(data)
    return output.getvalue()


def _redact_local_paths(value: Any) -> Any:
    """Replace machine-local Windows paths while retaining the terminal name."""
    if isinstance(value, dict):
        return {key: _redact_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_local_paths(item) for item in value]
    if isinstance(value, str) and len(value) >= 3 and value[1:3] in (":\\", ":/"):
        terminal = PureWindowsPath(value.replace("/", "\\")).name
        return f"<LOCAL_PATH>/{terminal}" if terminal else "<LOCAL_PATH>"
    return value


def _slim_bootstrap(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "valid_replicates": value["valid_replicates"],
            "unsupported_replicates": value["unsupported_replicates"],
            "median": value["median"],
            "ci95": value["ci95"],
        }
        for key, value in summary.items()
    }


def _extract_results(load_json: Callable[[str], Any] = _load_json) -> dict[str, Any]:
    diagnostic = load_json(SOURCES[3].source)
    provenance = load_json(SOURCES[5].source)
    controls = load_json(SOURCES[6].source)
    coverage = load_json(SOURCES[10].source)
    audit = load_json(SOURCES[12].source)
    read_profile = load_json(SOURCES[15].source)
    read_receipt = load_json(SOURCES[16].source)
    admission = load_json(SOURCES[19].source)
    stage1 = load_json(SOURCES[21].source)
    stage2 = load_json(SOURCES[22].source)
    gpu_profile = load_json(SOURCES[24].source)
    preflight_failure = load_json(SOURCES[25].source)
    r0_receipt = load_json(SOURCES[27].source)
    replay = load_json(SOURCES[31].source)

    evaluation = diagnostic["evaluation"]
    bootstrap = controls["paired_sequence_bootstrap"]
    _require(evaluation["protocol_sha256"] == PROTOCOL_SHA256, "diagnostic protocol mismatch")
    _require(controls["protocol_sha256"] == PROTOCOL_SHA256, "control protocol mismatch")
    _require(audit["evaluation_protocol_sha256"] == PROTOCOL_SHA256, "audit protocol mismatch")
    _require(stage1["evaluation_protocol_sha256"] == PROTOCOL_SHA256, "smoke protocol mismatch")
    _require(stage2["evaluation_protocol_sha256"] == PROTOCOL_SHA256, "resume protocol mismatch")
    _require(
        provenance["query_sidecar"]["prediction_parity"]["status"] == "PASS",
        "observer parity failed",
    )
    _require(read_receipt["parity"] == "PASS", "data read-profile parity failed")
    _require(admission["status"] == "PASS", "smoke admission failed")
    _require(r0_receipt["git_commit"] == TRAINING_PRODUCER_COMMIT, "R0 producer commit mismatch")
    _require(
        replay["targets_read"] is False and replay["optimizer_updates"] == 0,
        "S5 replay was not label-free",
    )

    population = {
        k: v for k, v in evaluation["population"].items() if k != "ground_truth_sha256_by_token"
    }
    profile_workers = {
        str(worker["label_cache_sequences"]): {
            key: worker[key]
            for key in (
                "cold_seconds",
                "warm_repeat_seconds",
                "peak_working_set_bytes",
                "trace_count",
                "trace_sha256",
                "checksums_sha256",
                "counts",
            )
        }
        for worker in read_profile["workers"]
    }
    return {
        "schema": "phase_ttc_detection_sota_evidence_results_v1",
        "scope": "local development evidence; no official eAP or SOTA claim",
        "code_revisions": {
            "r0_training_producer": TRAINING_PRODUCER_COMMIT,
            "s2_code_head": S2_CODE_COMMIT,
        },
        "s0": {
            "protocol_sha256": PROTOCOL_SHA256,
            "official_eap_score": evaluation["official_eap_score"],
            "sample_count": evaluation["sample_count"],
            "population": population,
            "predictions": {
                k: v for k, v in evaluation["predictions"].items() if not isinstance(v, list)
            },
            "supported_class_count": evaluation["supported_class_count"],
            "dev_AP101_supported_classes": evaluation["dev_AP101_supported_classes"],
            "dev_AP101_all_eight_absent_zero": evaluation["dev_AP101_all_eight_absent_zero"],
            "class_agnostic_recall": evaluation["class_agnostic_recall"],
            "observer_prediction_parity": provenance["query_sidecar"]["prediction_parity"],
            "controls": {
                "proposal_budget_per_frame": controls["proposal_budget_per_frame"],
                "observed": controls["observed_model_metrics"]["dev_AP101_supported_classes"],
                "constant_prior": controls["constant_prior"]["metrics"][
                    "dev_AP101_supported_classes"
                ],
                "frame_derangement": controls["frame_derangement"]["metrics"][
                    "dev_AP101_supported_classes"
                ],
                "observed_minus_constant_prior": controls[
                    "dev_AP101_observed_minus_constant_prior"
                ],
                "observed_minus_frame_derangement": controls[
                    "dev_AP101_observed_minus_derangement"
                ],
                "bootstrap": {
                    "unit": bootstrap["unit"],
                    "sequence_count": bootstrap["sequence_count"],
                    "num_replicates": bootstrap["num_replicates"],
                    "seed": bootstrap["seed"],
                    "observed_minus_control": _slim_bootstrap(bootstrap["observed_minus_control"]),
                },
            },
        },
        "s1_data": {
            "manifest_content_sha256": coverage["manifest_sha256"],
            "dataset_revision": audit["dataset_revision"],
            "source_media_sha256": audit["source_media_sha256"],
            "split_policy": audit["split_policy"],
            "train": coverage["train"],
            "selection": coverage["selection"],
            "read_profile": {
                "trace_count": read_profile["trace_count"],
                "seed": read_profile["seed"],
                "max_open_shards": read_profile["max_open_shards"],
                "parity": read_profile["parity"],
                "workers_by_label_cache_sequences": profile_workers,
            },
        },
        "s1_smoke": {
            "status": admission["status"],
            "scope": admission["scope"],
            "updates": admission["updates"],
            "image_exposures": admission["image_exposures"],
            "resume_boundary_update": admission["resume_boundary_update"],
            "sampler_continuation_exact": admission["sampler_continuation_exact"],
            "independent_inference": admission["independent_inference"],
            "stage1": {
                key: stage1[key]
                for key in (
                    "status",
                    "final_update",
                    "stage_image_exposures",
                    "stage_training_seconds",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            },
            "stage2": {
                key: stage2[key]
                for key in (
                    "status",
                    "start_update",
                    "final_update",
                    "cumulative_image_exposures",
                    "cumulative_training_seconds",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            },
            "gpu_profile": {
                "scope": gpu_profile["scope"],
                "device": gpu_profile["device"],
                "batch_size": gpu_profile["batch_size"],
                "warmup_steps": gpu_profile["warmup_steps"],
                "measured_steps": gpu_profile["measured_steps"],
                "checkpoint_load": gpu_profile["checkpoint_load"],
                "optimizer_load": gpu_profile["optimizer_load"],
                "preprocess_batch_seconds": gpu_profile["preprocess_batch_seconds"],
                "median_phase_seconds": gpu_profile["median_phase_seconds"],
            },
        },
        "s1_r0": {
            "v1_preflight": preflight_failure,
            "v2_status_at_bundle_cut": "RUNNING_NO_SCIENTIFIC_RESULT_INCLUDED",
            "producer_git_commit": r0_receipt["git_commit"],
            "nominal_updates": r0_receipt["nominal_updates"],
            "planned_segments": r0_receipt["planned_segments"],
            "live_metrics_excluded": True,
            "checkpoints_excluded": True,
        },
        "s5": {
            "status": replay["status"],
            "scope": replay["scope"],
            "sequence_id": replay["sequence_id"],
            "frame_count": replay["frame_count"],
            "final_detection_count": replay["final_detection_count"],
            "mature_unambiguous_track_count": replay["mature_unambiguous_track_count"],
            "event_packet_count": replay["event_packet_count"],
            "targets_read": replay["targets_read"],
            "optimizer_updates": replay["optimizer_updates"],
            "phase_bounds": replay["phase_bounds"],
            "phase_bounds_semantics": replay["phase_bounds_semantics"],
            "cuda_resources": replay["cuda_resources"],
        },
        "gates": {
            "S0": (
                "PARTIAL: local v2 diagnostics and controls closed; official scorer, broader "
                "independent population, and rare-class support remain open"
            ),
            "S1": (
                "RUNNING: D4096/data/smoke gates passed; R0 seed 7 v2 has no scientific "
                "result in this bundle"
            ),
            "S2": (
                "NOT_EVALUATED: code revision recorded; no frozen comparable result in current_v1"
            ),
            "S3": "NOT_STARTED: causal RGB history experiment absent",
            "S4": "NOT_STARTED: event contribution matrix absent",
            "S5": (
                "PARTIAL_ENGINEERING: real label-free detection-to-PHASE replay executed; "
                "no TTC accuracy/generalization claim"
            ),
            "S6": (
                "NOT_STARTED: multi-seed confirmation, calibration, and final independent "
                "comparison absent"
            ),
        },
        "exclusions": [
            "datasets and raw RGB/event media",
            "all checkpoints and optimizer states",
            "live R0 metrics while the run is incomplete",
            (
                "52 MB frame_population row ledger (derived audit and exact selected manifest "
                "retained)"
            ),
            (
                "large observer predictions/sidecar (their hashes and exact parity receipt are "
                "retained in S0 provenance)"
            ),
        ],
    }


def _format_float(value: float) -> str:
    return f"{value:.9f}"


def _render_results_markdown(results: dict[str, Any]) -> bytes:
    s0 = results["s0"]
    controls = s0["controls"]
    train = results["s1_data"]["train"]
    selection = results["s1_data"]["selection"]
    smoke = results["s1_smoke"]
    s5 = results["s5"]
    population = s0["population"]
    predictions = s0["predictions"]
    gt_counts = (
        f"{population['raw_ground_truth_count']} / {population['eligible_ground_truth_count']} / "
        f"{population['ignored_ground_truth_count']}"
    )
    prediction_counts = f"{predictions['input_count']} / {predictions['scored_count']}"
    train_evaluable = train["evaluable_annotations_v2"]
    selection_evaluable = selection["evaluable_annotations_v2"]
    trainable = train["trainable_annotations_legacy_profile"]
    selection_trainable = selection["trainable_annotations_legacy_profile"]
    evaluable_counts = f"{sum(train_evaluable.values())} / {sum(selection_evaluable.values())}"
    trainable_counts = f"{sum(trainable.values())} / {sum(selection_trainable.values())}"
    coverage_rows = [
        "",
        "| Clase | GT evaluable v2 en train | Target entrenable legacy en train | "
        "GT evaluable v2 en selección |",
        "|---|---:|---:|---:|",
        *(
            f"| {class_name} | {train_evaluable[class_name]} | {trainable[class_name]} | "
            f"{selection_evaluable[class_name]} |"
            for class_name in sorted(train_evaluable)
        ),
    ]
    targets_read = str(s5["targets_read"]).lower()
    lines = [
        "# Resultados generados de la continuación de detección",
        "",
        "> Archivo generado por `scripts/build_detection_sota_evidence.py`. No editar a mano.",
        "",
        "Estas cifras se extraen de artefactos canónicos con hashes conservados en "
        "`evidence/current_v1/manifest.json`. Son evidencia local de desarrollo; no son una "
        "puntuación eAP oficial ni una afirmación SOTA.",
        "",
        "## S0: evaluación v2 y controles",
        "",
        "| Magnitud | Valor |",
        "|---|---:|",
        f"| Frames | {s0['sample_count']} |",
        f"| GT raw / elegible / ignorado | {gt_counts} |",
        f"| Predicciones input / evaluadas | {prediction_counts} |",
        f"| Clases con soporte | {s0['supported_class_count']} / 8 |",
        (
            "| AP101 desarrollo, clases con soporte | "
            f"{_format_float(s0['dev_AP101_supported_classes'])} |"
        ),
        f"| Control constante train-only | {_format_float(controls['constant_prior'])} |",
        (
            "| Control por permutación sin puntos fijos | "
            f"{_format_float(controls['frame_derangement'])} |"
        ),
        (
            "| Delta frente a constante | "
            f"{_format_float(controls['observed_minus_constant_prior'])} |"
        ),
        (
            "| Delta frente a permutación | "
            f"{_format_float(controls['observed_minus_frame_derangement'])} |"
        ),
        "",
        f"La paridad del observador fue `{s0['observer_prediction_parity']['status']}` con error "
        f"máximo {s0['observer_prediction_parity']['maximum_absolute_error']}. El bootstrap usa "
        f"{controls['bootstrap']['num_replicates']} réplicas pareadas por secuencia, pero solo "
        f"{controls['bootstrap']['sequence_count']} clusters; sus intervalos son por ello débiles.",
        "",
        "## S1: datos, admisión y ejecución",
        "",
        "| Evidencia | Valor |",
        "|---|---:|",
        f"| D4096 train frames | {train['frame_count']} |",
        f"| D4096 selección frames | {selection['frame_count']} |",
        f"| GT evaluable v2 en D4096 train / selección | {evaluable_counts} |",
        f"| Targets entrenables legacy en D4096 train / selección | {trainable_counts} |",
        f"| Smoke updates / exposiciones | {smoke['updates']} / {smoke['image_exposures']} |",
        f"| Resume exacto en update | {smoke['resume_boundary_update']} |",
        f"| Inferencia independiente | {smoke['independent_inference']['status']} |",
        f"| R0 v2 | {results['s1_r0']['v2_status_at_bundle_cut']} |",
        *coverage_rows,
        "",
        "El R0 congelado usa el productor `fcad2299`; no se publica una métrica científica "
        "mientras sigue incompleto. El fallo v1 ocurrió antes de cualquier update y se conserva "
        "como recibo de fail-closed.",
        "",
        "## S5: replay real hasta PHASE",
        "",
        f"Se ejecutaron {s5['frame_count']} frames, con {s5['final_detection_count']} detecciones "
        f"finales, {s5['mature_unambiguous_track_count']} tracks maduros no ambiguos y "
        f"{s5['event_packet_count']} eventos ingeridos. `targets_read={targets_read}` "
        f"y `optimizer_updates={s5['optimizer_updates']}`. Los límites conservados son cuantiles "
        "nativos de fase, no un intervalo TTC continuo. Esta es evidencia de integración, no de "
        "precisión ni generalización TTC.",
        "",
        "## Estado de gates",
        "",
        "| Gate | Estado exacto en current_v1 |",
        "|---|---|",
    ]
    lines.extend(f"| {gate} | {state} |" for gate, state in results["gates"].items())
    lines.extend(
        [
            "",
            "## Regeneración",
            "",
            "```powershell",
            "python scripts/build_detection_sota_evidence.py",
            "python scripts/build_detection_sota_evidence.py --check",
            "python scripts/build_detection_sota_evidence.py --from-bundle --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _render_readme() -> bytes:
    return (
        "# Detection SOTA evidence — current_v1\n\n"
        "Bundle público, pequeño y regenerable de evidencia local. `results.json` contiene "
        "los escalares extraídos; `manifest.json` enlaza cada archivo empaquetado con el path, "
        "tamaño y SHA-256 de su fuente. Los JSON grandes usan gzip determinista (`mtime=0`).\n\n"
        "No contiene datasets, RGB/eventos crudos, checkpoints ni estados de optimizer. "
        "No acredita una puntuación eAP oficial, SOTA ni generalización TTC.\n\n"
        "`source_sha256` acredita los bytes locales originales. Cuando `encoding` declara "
        "redacción de rutas, esa transformación no es reversible; la verificación pública "
        "recalcula `packaged_sha256` y deriva resultados del payload redactado.\n\n"
        "Regenerar y verificar:\n\n"
        "```powershell\n"
        "python scripts/build_detection_sota_evidence.py\n"
        "python scripts/build_detection_sota_evidence.py --check\n"
        "python scripts/build_detection_sota_evidence.py --from-bundle --check\n"
        "```\n"
    ).encode()


def _build_expected() -> tuple[dict[Path, bytes], dict[str, Any]]:
    results = _extract_results()
    files: dict[Path, bytes] = {}
    source_entries: list[dict[str, Any]] = []
    for spec in SOURCES:
        source_path = REPO_ROOT / spec.source
        _require(source_path.is_file(), f"missing source: {spec.source}")
        source_bytes = source_path.read_bytes()
        transformed_bytes = source_bytes
        if spec.redact_local_paths:
            source_json = json.loads(source_bytes.decode("utf-8"))
            transformed_bytes = _json_bytes(_redact_local_paths(source_json))
        packaged_bytes = (
            _gzip_deterministic(transformed_bytes) if spec.gzip_compress else transformed_bytes
        )
        files[EVIDENCE_DIR / spec.packaged] = packaged_bytes
        source_entries.append(
            {
                "source": spec.source,
                "source_bytes": len(source_bytes),
                "source_sha256": _sha256(source_bytes),
                "packaged": spec.packaged,
                "packaged_bytes": len(packaged_bytes),
                "packaged_sha256": _sha256(packaged_bytes),
                "encoding": "+".join(
                    part
                    for part in (
                        "json-local-path-redaction-v1" if spec.redact_local_paths else "identity",
                        "gzip-mtime0" if spec.gzip_compress else "",
                    )
                    if part
                ),
            }
        )

    results_bytes = _json_bytes(results)
    generated_results_bytes = _render_results_markdown(results)
    readme_bytes = _render_readme()
    script_bytes = Path(__file__).read_bytes()
    files[EVIDENCE_DIR / "results.json"] = results_bytes
    files[EVIDENCE_DIR / "GENERATED_RESULTS.md"] = generated_results_bytes
    files[EVIDENCE_DIR / "README.md"] = readme_bytes
    files[EVIDENCE_DIR / "generator/build_detection_sota_evidence.py"] = script_bytes
    generated_entries = [
        {"packaged": name, "packaged_bytes": len(data), "packaged_sha256": _sha256(data)}
        for name, data in (
            ("results.json", results_bytes),
            ("GENERATED_RESULTS.md", generated_results_bytes),
            ("README.md", readme_bytes),
            ("generator/build_detection_sota_evidence.py", script_bytes),
        )
    ]
    manifest = {
        "schema": "phase_ttc_detection_sota_evidence_manifest_v1",
        "bundle": "current_v1",
        "deterministic": True,
        "generator_sha256": _sha256(script_bytes),
        "source_sha256_semantics": (
            "attestation of original local bytes; path-redacted packages are not reversible"
        ),
        "sources": source_entries,
        "generated": generated_entries,
        "excluded_classes": results["exclusions"],
    }
    files[EVIDENCE_DIR / "manifest.json"] = _json_bytes(manifest)
    files[RESULTS_MD] = generated_results_bytes
    return files, manifest


def _safe_bundle_path(relative_path: str) -> Path:
    _require(not Path(relative_path).is_absolute(), f"absolute packaged path: {relative_path}")
    candidate = EVIDENCE_DIR / relative_path
    _require(
        candidate.resolve().is_relative_to(EVIDENCE_DIR.resolve()),
        f"packaged path escapes bundle: {relative_path}",
    )
    return candidate


def _verify_packaged_entry(entry: dict[str, Any]) -> bytes:
    path = _safe_bundle_path(entry["packaged"])
    _require(path.is_file(), f"missing packaged file: {entry['packaged']}")
    data = path.read_bytes()
    _require(len(data) == entry["packaged_bytes"], f"size mismatch: {entry['packaged']}")
    _require(_sha256(data) == entry["packaged_sha256"], f"hash mismatch: {entry['packaged']}")
    return data


def _decode_packaged_source(entry: dict[str, Any], data: bytes) -> bytes:
    encodings = entry["encoding"].split("+")
    if "gzip-mtime0" in encodings:
        try:
            data = gzip.decompress(data)
        except (EOFError, gzip.BadGzipFile) as error:
            raise RuntimeError(f"invalid gzip: {entry['packaged']}") from error
    supported = {"identity", "json-local-path-redaction-v1", "gzip-mtime0"}
    _require(set(encodings) <= supported, f"unsupported encoding: {entry['encoding']}")
    if "json-local-path-redaction-v1" not in encodings:
        _require(len(data) == entry["source_bytes"], f"source size mismatch: {entry['source']}")
        _require(
            _sha256(data) == entry["source_sha256"], f"source hash mismatch: {entry['source']}"
        )
    return data


def _build_from_bundle() -> tuple[dict[Path, bytes], int]:
    """Derive public tables solely from checked packaged payloads."""
    manifest_path = EVIDENCE_DIR / "manifest.json"
    _require(manifest_path.is_file(), "missing bundle manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == "phase_ttc_detection_sota_evidence_manifest_v1",
        "unsupported bundle manifest schema",
    )
    source_payloads: dict[str, bytes] = {}
    for entry in manifest["sources"]:
        packaged = _verify_packaged_entry(entry)
        source_payloads[entry["source"]] = _decode_packaged_source(entry, packaged)
    for entry in manifest["generated"]:
        _verify_packaged_entry(entry)

    def load_bundle_json(source: str) -> Any:
        _require(source in source_payloads, f"source not packaged: {source}")
        return json.loads(source_payloads[source].decode("utf-8"))

    results = _extract_results(load_bundle_json)
    expected_results = _json_bytes(results)
    expected_markdown = _render_results_markdown(results)
    _require(
        (EVIDENCE_DIR / "results.json").read_bytes() == expected_results,
        "derived results differ from packaged results.json",
    )
    _require(
        (EVIDENCE_DIR / "GENERATED_RESULTS.md").read_bytes() == expected_markdown,
        "derived table differs from packaged GENERATED_RESULTS.md",
    )
    return {RESULTS_MD: expected_markdown}, len(manifest["sources"])


def _check(files: dict[Path, bytes]) -> int:
    expected_evidence = {
        path.relative_to(EVIDENCE_DIR).as_posix()
        for path in files
        if path.is_relative_to(EVIDENCE_DIR)
    }
    actual_evidence = (
        {
            path.relative_to(EVIDENCE_DIR).as_posix()
            for path in EVIDENCE_DIR.rglob("*")
            if path.is_file()
        }
        if EVIDENCE_DIR.exists()
        else set()
    )
    errors: list[str] = []
    if actual_evidence != expected_evidence:
        errors.append(
            f"evidence file set differs; missing={sorted(expected_evidence - actual_evidence)}, "
            f"unexpected={sorted(actual_evidence - expected_evidence)}"
        )
    for path, expected in files.items():
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(REPO_ROOT)}")
        elif path.read_bytes() != expected:
            errors.append(f"byte mismatch: {path.relative_to(REPO_ROOT)}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"PASS: {len(files)} files regenerate byte-for-byte")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify exact regenerated bytes without writing"
    )
    parser.add_argument(
        "--from-bundle",
        action="store_true",
        help="derive and verify public tables without private output files",
    )
    args = parser.parse_args()
    if args.from_bundle:
        files, source_count = _build_from_bundle()
        if args.check:
            print(f"PASS: public bundle verified; tables derived from {source_count} sources")
            return 0
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f"WROTE: {RESULTS_MD.relative_to(REPO_ROOT)} from {source_count} bundled sources")
        return 0
    files, manifest = _build_expected()
    if args.check:
        return _check(files)
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(
        f"WROTE: {len(files)} files; {len(manifest['sources'])} explicit sources; "
        f"bundle={EVIDENCE_DIR.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

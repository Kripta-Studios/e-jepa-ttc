#!/usr/bin/env python
"""Analyze a frozen label-free detector output against independent raw eAP labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.development_metrics import evaluate_detections  # noqa: E402
from e_jepa_ttc.detection.error_diagnostics import (  # noqa: E402
    analyze_query_sidecar,
    build_error_diagnostics,
)
from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationGroundTruth,
    build_evaluation_ground_truth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402

EXPECTED_MEDIA_SHA256 = "69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314"
EXPECTED_CHECKPOINT_SHA256 = "dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338"
EXPECTED_REVISION = "a814a946e60d37f7242358af5f344542177e50c5"
EXPECTED_PUBLIC_FILES_MANIFEST_SHA256 = (
    "2e65cd7f2600eea55f37f59e9a22d55b5ba0aa50445756fd7aaf01332c2b3ff1"
)


def sha256(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_public_files_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load the exact checked-in public file manifest and bind its revision."""
    manifest_sha256 = sha256(path)
    if manifest_sha256 != EXPECTED_PUBLIC_FILES_MANIFEST_SHA256:
        raise ValueError("Public-files manifest differs from the pinned manifest")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "eap_detection_public_files_v1":
        raise ValueError("Unexpected public-files manifest schema")
    if value.get("repository") != "NAIL-HNU/eAP-dataset":
        raise ValueError("Public-files manifest identifies another repository")
    if value.get("revision") != EXPECTED_REVISION:
        raise ValueError("Public-files manifest revision differs from the frozen release")
    if not isinstance(value.get("files"), dict):
        raise ValueError("Public-files manifest lacks its file mapping")
    return value, manifest_sha256


def verify_pinned_file(data_root: Path, relative: str, public_files: Mapping[str, object]) -> str:
    """Fail closed unless one dataset file matches the pinned size and SHA-256."""
    files = public_files["files"]
    if not isinstance(files, Mapping):
        raise ValueError("Public-files manifest lacks its file mapping")
    receipt = files.get(relative)
    if not isinstance(receipt, dict):
        raise ValueError(f"Dataset file is absent from the pinned manifest: {relative}")
    expected_sha256 = receipt.get("sha256")
    expected_size = receipt.get("size_bytes")
    if not isinstance(expected_sha256, str) or not isinstance(expected_size, int):
        raise ValueError(f"Pinned manifest lacks content identity for: {relative}")
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError("Dataset path escapes its root")
    if path.stat().st_size != expected_size:
        raise ValueError(f"Dataset file size differs from pinned manifest: {relative}")
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"Dataset file hash differs from pinned manifest: {relative}")
    return observed


def load_ground_truth(
    data_root: Path,
    tokens: list[str],
    protocol: EvaluationProtocol,
    public_files: Mapping[str, object],
) -> tuple[dict[str, EvaluationGroundTruth], dict[str, str]]:
    """Read only selected metadata/labels and build detector-independent GT."""
    media_path = data_root / "data/train.parquet"
    media_sha256 = verify_pinned_file(data_root, "data/train.parquet", public_files)
    if media_sha256 != EXPECTED_MEDIA_SHA256:
        raise ValueError("train.parquet differs from the frozen public release")
    media = pd.read_parquet(media_path, filters=[("sample_token", "in", tokens)])
    if set(media.sample_token) != set(tokens) or len(media) != len(tokens):
        raise ValueError("Frozen prediction tokens are missing or duplicated in train.parquet")
    if not media.split.eq("train").all():
        raise ValueError("Only public train metadata is admissible")
    label_hashes: dict[str, str] = {}
    labels_by_token: dict[str, list[dict[str, object]]] = {token: [] for token in tokens}
    for relative, rows in media.groupby("labels_path", sort=True):
        label_path = (data_root / str(relative)).resolve()
        if not label_path.is_relative_to(data_root.resolve()):
            raise ValueError("Label path escapes the dataset root")
        label_hashes[str(relative)] = verify_pinned_file(data_root, str(relative), public_files)
        requested = rows.sample_token.tolist()
        labels = pd.read_parquet(label_path, filters=[("sample_token", "in", requested)])
        if not labels.empty and (
            not labels.split.eq("train").all()
            or not set(labels.sample_token).issubset(set(requested))
        ):
            raise ValueError("Label file returned rows outside the requested train population")
        for token, frame in labels.groupby("sample_token", sort=False):
            labels_by_token[str(token)] = frame.to_dict("records")
    ground_truth: dict[str, EvaluationGroundTruth] = {}
    for row in media.to_dict("records"):
        token = str(row["sample_token"])
        ground_truth[token] = build_evaluation_ground_truth(
            labels_by_token[token],
            intrinsic=np.stack([np.asarray(value, dtype=np.float64) for value in row["K_event"]]),
            event_from_ego=np.stack(
                [np.asarray(value, dtype=np.float64) for value in row["T_event_ego"]]
            ),
            image_wh=(1280, 720),
            protocol=protocol,
        )
    return ground_truth, label_hashes


def render_report(result: dict[str, Any], provenance: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    assert isinstance(evaluation, dict)
    population = evaluation["population"]
    recall = evaluation["class_agnostic_recall"]
    assert isinstance(population, dict) and isinstance(recall, dict)
    threshold = recall["thresholds"]["2.0"]
    geometry = result["geometry_errors_on_class_agnostic_pairs"]
    oracles = result["oracles"]
    lines = [
        "# Diagnóstico S0 del detector congelado",
        "",
        (
            "Este informe usa exactamente las predicciones label-free del piloto y "
            "etiquetas raw leídas después, en un proceso separado. El perfil v2 es "
            "local y no oficial."
        ),
        "",
        f"- Checkpoint: `{provenance['checkpoint_sha256']}`.",
        f"- Predicciones: `{provenance['prediction_sha256']}`.",
        f"- Protocolo: `{provenance['protocol_sha256']}`.",
        (
            "- Población raw/elegible/ignorada: "
            f"{population['raw_ground_truth_count']} / "
            f"{population['eligible_ground_truth_count']} / "
            f"{population['ignored_ground_truth_count']}."
        ),
        (
            f"- Recall class-agnostic a 2 m: {threshold['recall']:.6f} "
            f"({threshold['TP']}/{recall['eligible_ground_truth_count']})."
        ),
        f"- AP101 v2 sobre clases con soporte: {evaluation['dev_AP101_supported_classes']:.8f}.",
        (
            "- Error medio de profundidad de cámara en correspondencias class-agnostic: "
            f"{geometry['camera_depth_error_m']['mean']:.4f} m."
        ),
        (
            "- Error vertical ego Z-up medio en las mismas correspondencias: "
            f"{geometry['ego_z_vertical_error_m']['mean']:.4f} m."
        ),
        "",
        "## Oráculos de desarrollo (pool fijo, rematching completo)",
        "",
        "| Sustitución | AP101 clases con soporte |",
        "|---|---:|",
    ]
    for name, row in oracles.items():
        lines.append(f"| {name} | {row['dev_AP101_supported_classes']:.8f} |")
    lines.append("")
    sidecar = result.get("query_sidecar")
    if isinstance(sidecar, dict):
        uncertainty = sidecar["depth_uncertainty"]
        assert isinstance(uncertainty, dict)
        lines.extend(
            [
                "## Observador externo con paridad exacta",
                "",
                (
                    "El sidecar aporta logits completos, query ID y el segundo componente "
                    "de `pred_depth`, entrenado como log-varianza, sin cambiar las "
                    "predicciones. Su correlación con el error absoluto de profundidad "
                    "óptica de cámara es "
                    f"{uncertainty['pearson_log_variance_vs_camera_depth_error']:.6f}; "
                    "no demuestra calibración de incertidumbre de profundidad."
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                "La box2D decodificada por query y la descomposición interna del modelo "
                "entre prior y residuo siguen sin estar disponibles. El diagnóstico "
                "geométrico `fy·H_GT/h_proyectada` es externo, usa GT y no se atribuye "
                "al modelo. No se infieren ni se fabrican internos ausentes."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("Use a new output directory; refusing to overwrite diagnostics")
    inference_path = args.predictions.resolve()
    frozen = json.loads(inference_path.read_text(encoding="utf-8"))
    if frozen.get("schema") != "monodgp_eap_label_free_inference_v1":
        raise ValueError("Unexpected frozen inference schema")
    if frozen.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen inference does not identify the pilot checkpoint")
    if (
        frozen.get("ground_truth_in_output") is not False
        or frozen.get("annotation_loader") != "override_returns_empty_without_path_access"
    ):
        raise ValueError("Inference receipt does not prove label-free production")
    tokens = list(frozen["sample_tokens"])
    predictions = cast(dict[str, list[dict[str, object]]], frozen["predictions"])
    if set(tokens) != set(predictions):
        raise ValueError("Prediction token population differs from its receipt")
    protocol = EvaluationProtocol.default()
    public_files, public_files_sha256 = load_public_files_manifest(
        args.public_files_manifest.resolve()
    )
    ground_truth, label_hashes = load_ground_truth(
        args.data_root.resolve(), tokens, protocol, public_files
    )
    diagnostics = build_error_diagnostics(predictions, ground_truth, protocol=protocol)
    sidecar_provenance = None
    if (args.sidecar is None) != (args.sidecar_predictions is None):
        raise ValueError("Sidecar and its prediction output must be supplied together")
    if args.sidecar is not None and args.sidecar_predictions is not None:
        sidecar_path = args.sidecar.resolve()
        sidecar_predictions_path = args.sidecar_predictions.resolve()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        observed_predictions = json.loads(sidecar_predictions_path.read_text(encoding="utf-8"))
        if sidecar.get("schema") != "monodgp_external_diagnostic_sidecar_v1":
            raise ValueError("Unexpected diagnostic sidecar schema")
        if sidecar.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("Sidecar checkpoint differs from the pilot")
        prediction_output_sha256 = sha256(sidecar_predictions_path)
        if sidecar.get("prediction_output_sha256") != prediction_output_sha256:
            raise ValueError("Sidecar is not bound to its prediction output")
        if observed_predictions.get("predictions") != predictions:
            raise ValueError("Sidecar prediction output differs from frozen pilot predictions")
        claims = sidecar.get("claims")
        if not isinstance(claims, Mapping) or claims != {
            "changes_predictions": False,
            "ground_truth_access": False,
            "internal_unexposed_values_reconstructed": False,
        }:
            raise ValueError("Sidecar claims do not preserve external observation scope")
        diagnostics["query_sidecar"] = analyze_query_sidecar(
            sidecar, predictions, ground_truth, protocol=protocol
        )
        sidecar_provenance = {
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256(sidecar_path),
            "prediction_output_path": str(sidecar_predictions_path),
            "prediction_output_sha256": prediction_output_sha256,
            "observer_sha256": sidecar.get("observer_sha256"),
            "prediction_parity": observed_predictions.get("prediction_parity"),
        }
    legacy = json.loads(args.legacy.read_text(encoding="utf-8"))
    if legacy["predictions"] != predictions:
        raise ValueError("Legacy evaluation predictions differ from label-free inference")
    legacy_recomputed = evaluate_detections(predictions, legacy["ground_truth"])
    if any(legacy["metrics"].get(key) != value for key, value in legacy_recomputed.items()):
        raise ValueError("Legacy detection metrics do not regenerate exactly")
    provenance = {
        "schema": "phase_ttc_detection_s0_provenance_v1",
        "dataset_revision": EXPECTED_REVISION,
        "public_files_manifest_path": str(args.public_files_manifest.resolve()),
        "public_files_manifest_sha256": public_files_sha256,
        "public_files_manifest_revision": public_files["revision"],
        "source_media_sha256": EXPECTED_MEDIA_SHA256,
        "label_file_sha256": label_hashes,
        "prediction_path": str(inference_path),
        "prediction_sha256": sha256(inference_path),
        "legacy_evaluation_sha256": sha256(args.legacy),
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "checkpoint_verified_by_inference_receipt": frozen["checkpoint_load"],
        "label_free_production": True,
        "query_sidecar": sidecar_provenance,
        "protocol_sha256": protocol.sha256,
        "analysis_code_sha256": {
            "scripts/analyze_detection_errors.py": sha256(Path(__file__)),
            "src/e_jepa_ttc/detection/error_diagnostics.py": sha256(
                ROOT / "src/e_jepa_ttc/detection/error_diagnostics.py"
            ),
            "src/e_jepa_ttc/detection/evaluation_ground_truth.py": sha256(
                ROOT / "src/e_jepa_ttc/detection/evaluation_ground_truth.py"
            ),
            "src/e_jepa_ttc/detection/evaluation_ap_fast.py": sha256(
                ROOT / "src/e_jepa_ttc/detection/evaluation_ap_fast.py"
            ),
            "src/e_jepa_ttc/detection/evaluation_protocol.py": sha256(
                ROOT / "src/e_jepa_ttc/detection/evaluation_protocol.py"
            ),
            "src/e_jepa_ttc/detection/evaluation_v2.py": sha256(
                ROOT / "src/e_jepa_ttc/detection/evaluation_v2.py"
            ),
        },
        "ground_truth_sha256_by_token": diagnostics["evaluation"]["population"][
            "ground_truth_sha256_by_token"
        ],
        "legacy_detection_metrics_exactly_regenerated": True,
        "legacy_supplemental_metrics_not_recomputed": sorted(
            set(legacy["metrics"]) - set(legacy_recomputed)
        ),
        "official_score": False,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "evaluation_protocol_v2.json", protocol.to_dict())
    write_json(
        args.output / "ground_truth_v2.json",
        {
            "schema": "phase_ttc_detection_ground_truth_bundle_v2",
            "protocol_sha256": protocol.sha256,
            "sequence_id_by_token": {
                token: token.split(":", 1)[0] for token in sorted(ground_truth)
            },
            "ground_truth_by_token": {
                token: ground_truth[token].to_dict() for token in sorted(ground_truth)
            },
        },
    )
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "legacy_recomputed.json", legacy_recomputed)
    write_json(args.output / "diagnostics.json", diagnostics)
    (args.output / "DIAGNOSTICS.md").write_text(
        render_report(diagnostics, provenance), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--sidecar-predictions", type=Path)
    parser.add_argument(
        "--public-files-manifest",
        type=Path,
        default=ROOT / "data/manifests/eap_detection_public_files_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

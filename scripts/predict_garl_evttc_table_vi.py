"""Run frozen, label-free model inference for the EvTTC Table VI protocol.

The primary mode accepts a checkpoint, inference config and frozen protocol,
reads only label-free NPZ inputs, and emits the stable payload consumed by
``evaluate_garl_evttc_table_vi.py``.  The legacy ``--input`` mode remains an
explicit normalization route for already-computed predictions.  Neither mode
opens TTC targets or performs model/checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e_jepa_ttc.utils.io import write_structured

if __package__:
    from scripts.table_vi_label_free import (
        canonical_token_schema,
        infer_predictions,
        load_label_free_inputs,
        load_model,
        load_normalization,
        parse_settings,
        read_mapping,
        reject_target_fields,
        resolve_device,
        sha256_file,
    )
else:
    from table_vi_label_free import (  # pyright: ignore[reportMissingImports]
        canonical_token_schema,
        infer_predictions,
        load_label_free_inputs,
        load_model,
        load_normalization,
        parse_settings,
        read_mapping,
        reject_target_fields,
        resolve_device,
        sha256_file,
    )

_FORBIDDEN_KEYS = frozenset(
    {
        "ttc",
        "ttc_s",
        "frame_ttc",
        "target_ttc",
        "target_ttc_s",
        "gt_ttc",
        "future_labels",
        "labels",
        "targets",
        "depth",
        "depth_history",
        "category",
        "category_index",
        "class_id",
        "foreground_mask",
        "geometry_target",
        "mask",
        "visible_heights_px",
    }
)
_IDENTITY_KEYS = ("sequence_id", "sample_token", "track_id", "timestamp_us")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _reject_forbidden(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            # ``predicted_ttc_s`` is the model output, not a target.  All
            # target-like spellings remain forbidden at every nesting level.
            if key_text in _FORBIDDEN_KEYS:
                raise ValueError(f"{path} contains forbidden target field: {key}")
            _reject_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, path=f"{path}[{index}]")


def _input_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _reject_forbidden(payload)
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input payload must contain a non-empty predictions list.")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Prediction row {index} is not an object.")
        missing = [key for key in _IDENTITY_KEYS if key not in raw]
        if missing:
            raise ValueError(f"Prediction row {index} is missing identity fields: {missing}")
        if "predicted_ttc_s" not in raw:
            raise ValueError(f"Prediction row {index} is missing predicted_ttc_s.")
        identity = (
            str(raw["sequence_id"]),
            str(raw["sample_token"]),
            str(raw["track_id"]),
            int(raw["timestamp_us"]),
        )
        if identity in seen:
            raise ValueError(f"Duplicate prediction identity: {identity}")
        seen.add(identity)
        value = float(raw["predicted_ttc_s"])
        if not (-float("inf") < value < float("inf")):
            raise ValueError(f"Prediction row {index} has a non-finite predicted_ttc_s.")
        normalized.append(
            {
                **{key: raw[key] for key in _IDENTITY_KEYS},
                "predicted_ttc_s": value,
            }
        )
    return normalized


def predict(input_path: Path, output_path: Path, *, protocol_id: str) -> dict[str, Any]:
    """Normalize explicit label-free predictions without reading targets."""

    payload = _read(input_path)
    rows = _input_rows(payload)
    result: dict[str, Any] = {
        "artifact_type": "garl_evttc_table_vi_predictions_v1",
        "schema_version": "v1",
        "evidence_type": "label_free_inference_payload",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": protocol_id,
        "selection_uses_evttc": False,
        "target_labels_opened": False,
        "training_updates_on_target_dataset": 0,
        "source_payload_sha256": _sha256(input_path),
        "sample_count": len(rows),
        "predictions": rows,
    }
    write_structured(output_path, result)
    return result


def run_label_free_inference(
    checkpoint_path: Path,
    config_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    normalization_path: Path | None = None,
) -> dict[str, Any]:
    """Run frozen inference after schema and protocol coverage validation."""

    config = read_mapping(config_path)
    protocol = read_mapping(protocol_path)
    reject_target_fields(protocol, location="protocol")
    canonical_token_schema(protocol, location="protocol")
    protocol_id = str(protocol.get("protocol_id", "")).strip()
    if not protocol_id:
        raise ValueError("protocol.protocol_id must be a non-empty string.")
    if protocol.get("labels_used_by_predict") is not False:
        raise ValueError("Protocol must explicitly declare labels_used_by_predict=false.")
    if protocol.get("selection_uses_evttc") is not False:
        raise ValueError("Protocol must explicitly declare selection_uses_evttc=false.")
    if protocol.get("predict_score_separation") is not True:
        raise ValueError("Protocol must explicitly declare predict_score_separation=true.")
    bbox_protocol = str(protocol.get("bbox_protocol", "")).strip()
    if bbox_protocol not in {
        "P0_oracle_bbox_roi",
        "P1_predicted_bbox_roi",
        "P2_raw_fullframe",
    }:
        raise ValueError(
            "Protocol must explicitly declare bbox_protocol as "
            "P0_oracle_bbox_roi, P1_predicted_bbox_roi or P2_raw_fullframe."
        )

    settings = parse_settings(
        config,
        config_path=config_path,
        normalization_override=normalization_path,
    )
    arrays, identities, coverage, shard_evidence = load_label_free_inputs(
        settings.input_manifest,
        protocol,
    )
    normalization, normalization_sha256 = load_normalization(settings.normalization_path)

    # Input schema, target-field rejection, token uniqueness and exact coverage
    # are deliberately complete before checkpoint deserialization or GPU use.
    device = resolve_device(settings.device)
    model = load_model(checkpoint_path, settings, device)
    rows = infer_predictions(
        model,
        arrays,
        identities,
        batch_size=settings.batch_size,
        device=device,
        normalization=normalization,
    )
    result: dict[str, Any] = {
        "artifact_type": "garl_evttc_table_vi_predictions_v1",
        "schema_version": "v1",
        "evidence_type": "frozen_label_free_checkpoint_inference",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": protocol_id,
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "bbox_protocol": bbox_protocol,
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_manifest_path": settings.input_manifest.as_posix(),
        "input_manifest_sha256": sha256_file(settings.input_manifest),
        "normalization_path": (
            settings.normalization_path.as_posix()
            if settings.normalization_path is not None
            else None
        ),
        "normalization_sha256": normalization_sha256,
        "normalization_explicit": settings.normalization_path is not None,
        "model_architecture": settings.architecture,
        "device": str(device),
        "token_schema": list(_IDENTITY_KEYS),
        "coverage_by_sequence": coverage,
        "input_shards": shard_evidence,
        "selection_uses_evttc": False,
        "target_labels_opened": False,
        "training_updates_on_target_dataset": 0,
        "sample_count": len(rows),
        "predictions": rows,
    }
    write_structured(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--checkpoint",
        type=Path,
        help="Frozen checkpoint used for label-free inference.",
    )
    mode.add_argument(
        "--input",
        type=Path,
        help="Explicit legacy normalization payload containing predictions.",
    )
    parser.add_argument("--config", type=Path, help="Inference-only JSON/YAML config.")
    parser.add_argument("--protocol", type=Path, help="Frozen Table VI protocol JSON/YAML.")
    parser.add_argument(
        "--normalization",
        type=Path,
        help="Optional explicit normalization JSON/YAML; overrides config.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", default="garl_evttc_table_vi_v1")
    args = parser.parse_args()
    if args.input is not None:
        if args.config is not None or args.protocol is not None or args.normalization is not None:
            parser.error("--input normalization mode cannot be combined with inference options.")
        result = predict(args.input, args.output, protocol_id=args.protocol_id)
    else:
        if args.config is None or args.protocol is None:
            parser.error("--checkpoint requires both --config and --protocol.")
        result = run_label_free_inference(
            args.checkpoint,
            args.config,
            args.protocol,
            args.output,
            normalization_path=args.normalization,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

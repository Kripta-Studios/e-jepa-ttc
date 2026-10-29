"""Zero-update RGB-PORT transfer on the already-exposed EvTTC Dev32 cohort.

Prediction is deliberately separated from scoring.  ``predict-dev32`` never opens
ground-truth or historical prediction CSVs; ``score-dev32`` requires a sealed
prediction artifact before joining the public development labels/comparators.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.features import (
    EVENT_PHASE17_SHA256,
    RGB_PHASE17_SHA256,
    ProducerObservation,
    event_phase17,
    event_statistics_from_channels12,
    raw_rgb_statistics,
    rgb_phase17,
)
from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from e_jepa_ttc.simplex_t.phase import emitted_phase, phase_to_ttc
from operational.evttc_transfer.score import metrics as native_metrics
from operational.rgb_port.accounting import (
    atomic_write_bytes,
    atomic_write_json,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port.evaluate import common_cap60_metrics, score_signed_ttc
from operational.rgb_port.infer_experts import Fragments
from operational.rgb_port.prepare_features import ENDPOINT_IDS
from operational.rgb_port.recipe import canonical_sha256, resolved_recipe
from operational.rgb_port.train_heads import load_head_endpoint, pair_point_phase
from operational.rgb_port.train_producers import load_producer_endpoint, producer_features
from operational.rgb_port_revision.transfer_inputs import (
    EVENT_LAGS_US,
    prepare_event,
    raw_rgb_observations,
)

PRODUCERS = ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F")
HEADS = (
    "PAIR_E_MATCHED",
    "PAIR_R",
    "E_H1_MATCHED",
    "E_CTX_MATCHED",
    "R_H1",
    "R_CTX",
    "F_TRUE",
    "F_ZERO",
)
PREDICTION_FIELDS = (
    "query_id",
    "sequence_id",
    "anchor_us",
    "prediction_status",
    "rgb_available",
    "rgb_unavailable_reason",
    "E_H1_MATCHED",
    "E_CTX_MATCHED",
    "R_H1",
    "R_CTX",
    "F_TRUE",
    "F_ZERO",
)
RGBHistoryItem = tuple[int, int, int, Tensor, Tensor]

OWN_RGB_PREPROCESSING = {
    "contract": "rgb_port_own_shared_roi_v1",
    "camera": "blackflys/left",
    "frame_selection": (
        "actual_causal_sensor_triplets_within_650ms; T2_only_at_true_sensor_cold_start"
    ),
    "history": "up_to_8_actual_sensor_observations_not_sparse_evaluation_queries",
    "event_history": "4_observations_at_100ms_cadence_full_span_at_most_600ms",
    "adaptation": (
        "retrospective_shared_query_ROI_available_at_query; differs_from_native_per_anchor_ROI"
    ),
    "timestamps": "blackflys/left/ts_actual_sensor_clock",
    "roi": "union_of_last_two_visible_rgb_boxes_common_square_margin_fraction_0.25",
    "crop": "crop_uint8_clipped_shared_roi_PIL_RGB_bilinear_128x128",
    "quantization": "resize_uint8_then_float32_div_255",
    "target_fields_used": False,
}

PUBLISHED_GARL_PREPROCESSING = {
    "contract": "published_garl_native_historical_preprocessing",
    "use": "read_only_precomputed_comparator_joined_after_own_predictions_are_sealed",
    "reused_for_own_rgb_port_predictions": False,
    "rerun": False,
}


def _source_bindings() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative = (
        "operational/rgb_port/transfer.py",
        "operational/rgb_port_revision/transfer_inputs.py",
        "operational/rgb_port/accounting.py",
        "operational/rgb_port/infer_experts.py",
        "operational/rgb_port/recipe.py",
        "operational/evttc_transfer/inputs.py",
        "operational/evttc_rgb_transfer/inputs.py",
        "src/e_jepa_ttc/data/event_v4_geometry.py",
        "src/e_jepa_ttc/rgb_port/contracts.py",
        "src/e_jepa_ttc/rgb_port/data.py",
        "src/e_jepa_ttc/rgb_port/features.py",
        "src/e_jepa_ttc/rgb_port/fusion.py",
    )
    return {name: sha256_file(root / name) for name in relative}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode())


def _manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = read_json_shared(path)
    rows = value.get("rows")
    forbidden = {"ttc", "target_ttc", "truth_ttc_seconds", "distance", "depth"}
    if value.get("status") != "LABEL_FREE_MANIFEST" or not isinstance(rows, list) or not rows:
        raise ValueError("Dev32 source must be the frozen label-free query manifest")
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError("prediction manifest contains a forbidden target field")
    ids = [str(row["query_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Dev32 query identities are not unique")
    return value, rows


def _select_rgb_history(
    timeline: Sequence[RGBHistoryItem], prediction_cutoff: int
) -> list[RGBHistoryItem]:
    """Select at most eight observations whose complete RGB pairs fit in 650 ms."""
    selected = [
        item
        for item in timeline
        if item[2] <= prediction_cutoff and 0 <= prediction_cutoff - item[0] <= 650_000
    ][-8:]
    if selected and any(
        selected[index][1] >= selected[index + 1][1] for index in range(len(selected) - 1)
    ):
        raise ValueError("RGB history anchors are not strictly chronological")
    return selected


def _own_rgb_crop(
    frames: Sequence[np.ndarray], boxes: Sequence[Sequence[float]]
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Shared last-two-box ROI for genuine T2 or T3 sensor observations."""
    from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
    from e_jepa_ttc.rgb_port.data import crop_uint8

    if len(frames) not in {2, 3} or len(boxes) != len(frames):
        raise ValueError("RGB observations require two or three aligned frames and boxes")
    square = common_square_from_boxes(boxes, (len(boxes) - 2, len(boxes) - 1), margin_fraction=0.25)
    uint8 = np.stack([crop_uint8(frame, square) for frame in frames])
    return np.ascontiguousarray(uint8.astype(np.float32) / np.float32(255.0)), square


def _fragment_values(
    prediction: Mapping[str, Any], history: Sequence[RGBHistoryItem]
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in PREDICTION_FIELDS:
        value = prediction[name]
        if name in {"query_id", "sequence_id", "prediction_status", "rgb_unavailable_reason"}:
            values[f"row_{name}"] = np.asarray(str(value))
        elif name == "rgb_available":
            values[f"row_{name}"] = np.asarray(bool(value), dtype=np.bool_)
        elif name == "anchor_us":
            values[f"row_{name}"] = np.asarray(int(value), dtype=np.int64)
        else:
            values[f"row_{name}"] = np.asarray(float(value), dtype=np.float64)
    values["history_first_sensor_ts_us"] = np.asarray([item[0] for item in history], np.int64)
    values["history_anchor_us"] = np.asarray([item[1] for item in history], np.int64)
    values["history_available_us"] = np.asarray([item[2] for item in history], np.int64)
    values["history_features"] = (
        torch.stack([item[3] for item in history]).detach().cpu().numpy().astype(np.float32)
        if history
        else np.empty((0, 17), np.float32)
    )
    values["history_expert_phase"] = (
        torch.stack([item[4] for item in history]).detach().cpu().numpy().astype(np.float32)
        if history
        else np.empty((0, 3), np.float32)
    )
    return values


def _restore_fragment(
    values: Mapping[str, np.ndarray], row: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, Any], list[RGBHistoryItem]]:
    required = {f"row_{name}" for name in PREDICTION_FIELDS} | {
        "history_first_sensor_ts_us",
        "history_anchor_us",
        "history_available_us",
        "history_features",
        "history_expert_phase",
    }
    if set(values) != required:
        raise ValueError("Dev32 transfer fragment schema differs")
    prediction: dict[str, Any] = {}
    for name in PREDICTION_FIELDS:
        value = np.asarray(values[f"row_{name}"]).item()
        prediction[name] = value
    if (
        str(prediction["query_id"]) != str(row["query_id"])
        or str(prediction["sequence_id"]) != str(row["sequence_id"])
        or int(prediction["anchor_us"]) != int(row["anchor_us"])
    ):
        raise ValueError("Dev32 fragment row identity differs")
    first = np.asarray(values["history_first_sensor_ts_us"], np.int64)
    anchors = np.asarray(values["history_anchor_us"], np.int64)
    available = np.asarray(values["history_available_us"], np.int64)
    features = np.asarray(values["history_features"], np.float32)
    experts = np.asarray(values["history_expert_phase"], np.float32)
    count = len(first)
    if (
        anchors.shape != (count,)
        or available.shape != (count,)
        or features.shape != (count, 17)
        or experts.shape != (count, 3)
        or count > 8
        or np.any(anchors <= first)
        or np.any(available < anchors)
        or not np.isfinite(features).all()
        or not np.isfinite(experts).all()
    ):
        raise ValueError("Dev32 fragment RGB history differs")
    history = [
        (
            int(first[index]),
            int(anchors[index]),
            int(available[index]),
            torch.from_numpy(features[index]).to(device),
            torch.from_numpy(experts[index]).to(device),
        )
        for index in range(count)
    ]
    return prediction, history


def _fragment_inventory(store: Fragments, count: int) -> str:
    records: list[dict[str, str]] = []
    for index in range(count):
        path = store.root / f"part_{index:06d}.npz"
        receipt_path = path.with_suffix(".json")
        receipt = read_json_shared(receipt_path)
        fragment_sha = sha256_file(path)
        if receipt.get("binding_sha256") != store.digest or receipt.get("sha256") != fragment_sha:
            raise ValueError("Dev32 fragment inventory changed before sealing")
        records.append(
            {
                "fragment_sha256": fragment_sha,
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
    return canonical_sha256({"binding_sha256": store.digest, "records": records})


def _recipes(config: Path, p_manifest: Path) -> dict[str, Any]:
    manifest = read_json_shared(p_manifest)
    population = int(manifest["population_size"])
    digest = sha256_file(p_manifest)
    return {
        fit_id: resolved_recipe(
            config, fit_id=fit_id, producer_population=population, role_manifest_sha256=digest
        )
        for fit_id in PRODUCERS
    }


def _models(run: Path, config: Path, p_manifest: Path, device: str) -> dict[str, nn.Module]:
    recipes = _recipes(config, p_manifest)
    result: dict[str, nn.Module] = {
        fit_id: load_producer_endpoint(run / "fits" / fit_id, recipes[fit_id], device=device)
        for fit_id in PRODUCERS
    }
    target = torch.device(device)
    result.update(
        {fit_id: load_head_endpoint(run / "fits" / fit_id, fit_id, target) for fit_id in HEADS}
    )
    return result


def _endpoint_bindings(run: Path) -> dict[str, str]:
    """Bind every head consumed by transfer, including both controlled fusions."""
    result: dict[str, str] = {}
    for fit_id in HEADS:
        receipt = read_json_shared(run / "fits" / fit_id / "CHECKPOINT_RECEIPT.json")
        checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
        if (
            receipt.get("fit_id") != fit_id
            or receipt.get("status") != "COMPLETE"
            or not receipt.get("scientific_endpoint")
            or receipt.get("checkpoint_sha256") != sha256_file(checkpoint)
        ):
            raise ValueError(f"{fit_id} transfer endpoint is incomplete or changed")
        result[fit_id] = str(receipt["checkpoint_sha256"])
    return result


def _validate_endpoint_freeze(path: Path) -> dict[str, str]:
    """Read the live freeze with FILE_SHARE_DELETE and verify all six immutable parents."""
    payload = read_json_shared(path)
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != ENDPOINT_IDS:
        raise ValueError("feature freeze must bind all four producers and both PAIR endpoints")
    result: dict[str, str] = {}
    for fit_id, raw in endpoints.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{fit_id} freeze entry is invalid")
        checkpoint = Path(raw["checkpoint_path"]).resolve(strict=True)
        if (
            raw.get("status") != "COMPLETE"
            or not raw.get("scientific_endpoint")
            or raw.get("checkpoint_sha256") != sha256_file(checkpoint)
        ):
            raise ValueError(f"{fit_id} is not a frozen endpoint")
        result[fit_id] = str(raw["checkpoint_sha256"])
    return result


def _normalizer(path: Path, modality: str, freeze_sha: str) -> tuple[Tensor, Tensor]:
    value = FrozenNormalizer.load(path)
    # Endpoint ancestry is checked by the six-endpoint freeze.  Here the transform itself is
    # bound by file hash in the seal and must retain its modality and H-only fit role.
    schema = EVENT_PHASE17_SHA256 if modality == "event" else RGB_PHASE17_SHA256
    if (
        value.modality != modality
        or value.fit_role != "H"
        or value.schema_sha256 != schema
        or value.producer_sha256 != freeze_sha
        or value.mean.shape != (17,)
    ):
        raise ValueError(f"{modality} normalizer contract differs")
    return torch.from_numpy(value.mean).float(), torch.from_numpy(value.scale).float()


def _clock(anchors: np.ndarray, available: np.ndarray, query: int) -> np.ndarray:
    result = np.zeros((len(anchors), 4), np.float32)
    for i, (anchor, arrival) in enumerate(zip(anchors, available, strict=True)):
        if anchor > query or arrival > query or arrival < anchor:
            raise ValueError("transfer observation is not causally available")
        result[i] = (
            (query - anchor) / 1e6,
            (query - arrival) / 1e6,
            0 if i == 0 else (anchor - anchors[i - 1]) / 1e6,
            (arrival - anchor) / 1e6,
        )
    return result


def _expert_block(
    models: Mapping[str, nn.Module], prefix: str, sensor: Tensor, delta: Tensor
) -> tuple[Tensor, Tensor]:
    ids = (
        ("R_A5", "R_C2F", "PAIR_R")
        if prefix == "R"
        else ("E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED")
    )
    outputs = [
        producer_features(cast(CausalScaleTTC, models[name]), sensor, delta) for name in ids[:2]
    ]
    pair_features = ProducerObservation.from_output(outputs[0]).pair_input(delta[:, -1])
    pair_phase = emitted_phase(pair_point_phase(models[ids[2]], pair_features))
    expert = torch.stack((outputs[0]["point_phase"], outputs[1]["point_phase"], pair_phase), -1)
    d0 = torch.stack((outputs[0]["flow"], outputs[0]["margin"], outputs[0]["log_variance"]), -1)
    d1 = torch.stack((outputs[1]["flow"], outputs[1]["margin"], outputs[1]["log_variance"]), -1)
    base = (
        raw_rgb_statistics(sensor[:, -1])
        if prefix == "R"
        else event_statistics_from_channels12(sensor[:, -1])
    )
    features = (
        rgb_phase17(sensor[:, -1], d0, d1, expert)
        if prefix == "R"
        else event_phase17(base, d0, d1, expert)
    )
    return features, expert


def _head_ttc(
    model: nn.Module, features: Tensor, timing: Tensor, valid: Tensor, expert: Tensor
) -> float:
    output = model(features, timing, valid, expert)
    return float(phase_to_ttc(output["point_phase"]).item())


def predict_dev32(
    *,
    run: Path,
    config: Path,
    p_manifest: Path,
    endpoint_freeze: Path,
    query_manifest: Path,
    event_normalizer: Path,
    rgb_normalizer: Path,
    output: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run own frozen endpoints on pixels/events only and seal target-free predictions."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    endpoint_hashes = _validate_endpoint_freeze(endpoint_freeze)
    head_hashes = _endpoint_bindings(run)
    manifest, rows = _manifest(query_manifest)
    completed_path = output / "PREDICTIONS_SEALED.json"
    expected = {
        "query_manifest_sha256": sha256_file(query_manifest),
        "config_sha256": sha256_file(config),
        "p_manifest_sha256": sha256_file(p_manifest),
        "endpoint_freeze_sha256": sha256_file(endpoint_freeze),
        "endpoint_hashes": endpoint_hashes,
        "head_endpoint_hashes": head_hashes,
        "normalizers": {
            "event": sha256_file(event_normalizer),
            "rgb": sha256_file(rgb_normalizer),
        },
        "source_sha256": _source_bindings(),
        "own_rgb_preprocessing": OWN_RGB_PREPROCESSING,
        "device": device,
        "precision": "float32",
    }
    fragment_binding = {
        "schema": "rgb_port_dev32_transfer_fragments_v1",
        **expected,
        "population": len(rows),
        "query_rows_metadata_sha256": manifest.get("rows_metadata_sha256"),
        "optimizer_updates": 0,
        "targets_are_model_inputs": False,
    }
    fragments = Fragments(output / "fragments", fragment_binding)
    if completed_path.is_file():
        completed = read_json_shared(completed_path)
        predictions = Path(completed["predictions_path"])
        if (
            completed.get("status") != "COMPLETE"
            or any(completed.get(key) != value for key, value in expected.items())
            or completed.get("predictions_sha256") != sha256_file(predictions)
            or completed.get("fragment_binding_sha256") != fragments.digest
            or completed.get("fragment_inventory_sha256")
            != _fragment_inventory(fragments, len(rows))
        ):
            raise ValueError("sealed Dev32 transfer differs from current immutable inputs")
        return completed
    target = torch.device(device)
    event_mean, event_scale = [
        x.to(target) for x in _normalizer(event_normalizer, "event", sha256_file(endpoint_freeze))
    ]
    rgb_mean, rgb_scale = [
        x.to(target) for x in _normalizer(rgb_normalizer, "rgb", sha256_file(endpoint_freeze))
    ]
    prediction_rows: list[dict[str, Any]] = []
    rgb_histories: dict[str, list[RGBHistoryItem]] = {}
    models: dict[str, nn.Module] | None = None
    with torch.inference_mode():
        for index, row in enumerate(rows):
            sequence_id = str(row["sequence_id"])
            saved = fragments.read(index)
            if saved is not None:
                prediction, history = _restore_fragment(saved, row, target)
                rgb_histories[sequence_id] = history
                prediction_rows.append(prediction)
                continue
            if models is None:
                models = _models(run, config, p_manifest, device)
            prepared = prepare_event(row)
            event_sensor = torch.from_numpy(prepared["own_events"]).to(target)
            event_delta = torch.full(
                (len(EVENT_LAGS_US), 2), 0.1, dtype=torch.float32, device=target
            )
            event_raw, event_expert = _expert_block(models, "E", event_sensor, event_delta)
            event_features = ((event_raw - event_mean) / event_scale).unsqueeze(0)
            query = int(row["anchor_us"])
            event_anchors = query - np.asarray(EVENT_LAGS_US, dtype=np.int64)
            # The common ROI is selected causally for this query, so every retrospective
            # observation becomes available only at the query boundary.
            event_available = np.full(len(EVENT_LAGS_US), query, np.int64)
            event_timing = (
                torch.from_numpy(_clock(event_anchors, event_available, query))
                .to(target)
                .unsqueeze(0)
            )
            event_valid = torch.ones((1, len(EVENT_LAGS_US)), dtype=torch.bool, device=target)
            current_event = event_expert[-1:].float()
            e_h1 = _head_ttc(
                models["E_H1_MATCHED"],
                event_features[:, -1:],
                event_timing[:, -1:],
                event_valid[:, -1:],
                current_event,
            )
            e_ctx = _head_ttc(
                models["E_CTX_MATCHED"], event_features, event_timing, event_valid, current_event
            )

            observations, unavailable_reason = raw_rgb_observations(row)
            rgb_available = bool(observations)
            rgb_meta = {"reason": unavailable_reason}
            values: dict[str, Any] = {"E_H1_MATCHED": e_h1, "E_CTX_MATCHED": e_ctx}
            if rgb_available:
                prediction_cutoff = query
                timeline = []
                for rgb, meta in observations:
                    rgb_sensor = torch.from_numpy(rgb).to(target).unsqueeze(0)
                    rgb_delta = torch.tensor(
                        [meta["delta_t_s"]], dtype=torch.float32, device=target
                    )
                    rgb_raw, rgb_expert = _expert_block(models, "R", rgb_sensor, rgb_delta)
                    timeline.append(
                        (
                            int(meta["first_sensor_ts_us"]),
                            int(meta["anchor_us"]),
                            int(meta["available_us"]),
                            ((rgb_raw - rgb_mean) / rgb_scale)[0].detach(),
                            rgb_expert[0].detach(),
                        )
                    )
                history = _select_rgb_history(timeline, prediction_cutoff)
                rgb_histories[sequence_id] = history
                rgb_features = torch.stack([item[3] for item in history]).unsqueeze(0)
                rgb_expert_current = history[-1][4].unsqueeze(0)
                rgb_timing = (
                    torch.from_numpy(
                        _clock(
                            np.asarray([item[1] for item in history], np.int64),
                            np.asarray([item[2] for item in history], np.int64),
                            prediction_cutoff,
                        )
                    )
                    .to(target)
                    .unsqueeze(0)
                )
                rgb_valid = torch.ones((1, len(history)), dtype=torch.bool, device=target)
                values["R_H1"] = _head_ttc(
                    models["R_H1"],
                    rgb_features[:, -1:],
                    rgb_timing[:, -1:],
                    rgb_valid[:, -1:],
                    rgb_expert_current,
                )
                values["R_CTX"] = _head_ttc(
                    models["R_CTX"],
                    rgb_features,
                    rgb_timing,
                    rgb_valid,
                    rgb_expert_current,
                )
                for fit_id in ("F_TRUE", "F_ZERO"):
                    fusion_event_timing = (
                        torch.from_numpy(_clock(event_anchors, event_available, prediction_cutoff))
                        .to(target)
                        .unsqueeze(0)
                    )
                    fused = models[fit_id](
                        event_features,
                        fusion_event_timing,
                        event_valid,
                        current_event,
                        rgb_features,
                        rgb_timing,
                        rgb_valid,
                    )
                    values[fit_id] = float(phase_to_ttc(fused["point_phase"]).item())
                reason = ""
            else:
                history = _select_rgb_history(rgb_histories.get(sequence_id, []), query)
                rgb_histories[sequence_id] = history
                values.update(
                    {"R_H1": float("nan"), "R_CTX": float("nan"), "F_TRUE": e_ctx, "F_ZERO": e_ctx}
                )
                reason = str(rgb_meta.get("reason") or "RGB_UNAVAILABLE")
            prediction = {
                "query_id": row["query_id"],
                "sequence_id": row["sequence_id"],
                "anchor_us": query,
                "prediction_status": "PREDICTED",
                "rgb_available": rgb_available,
                "rgb_unavailable_reason": reason,
                **values,
            }
            fragments.write(index, _fragment_values(prediction, history))
            prediction_rows.append(prediction)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "PREDICTIONS.csv"
    _write_csv(csv_path, prediction_rows, PREDICTION_FIELDS)
    seal = {
        "schema": "rgb_port_dev32_prediction_seal_v1",
        "status": "COMPLETE",
        "protocol_scope": "historically_exposed_exploratory_transfer_not_holdout_or_sota",
        "labels_read": False,
        "optimizer_updates": 0,
        "population": len(rows),
        "query_manifest_sha256": expected["query_manifest_sha256"],
        "config_sha256": expected["config_sha256"],
        "p_manifest_sha256": expected["p_manifest_sha256"],
        "query_rows_metadata_sha256": manifest.get("rows_metadata_sha256"),
        "endpoint_freeze_sha256": expected["endpoint_freeze_sha256"],
        "endpoint_hashes": endpoint_hashes,
        "head_endpoint_hashes": head_hashes,
        "normalizers": expected["normalizers"],
        "predictions_path": str(csv_path.resolve()),
        "predictions_sha256": sha256_file(csv_path),
        "source_sha256": expected["source_sha256"],
        "own_rgb_preprocessing": expected["own_rgb_preprocessing"],
        "device": expected["device"],
        "precision": expected["precision"],
        "fragment_binding_sha256": fragments.digest,
        "fragment_inventory_sha256": _fragment_inventory(fragments, len(rows)),
        "fragment_count": len(rows),
    }
    atomic_write_json(output / "PREDICTIONS_SEALED.json", seal)
    return seal


def score_dev32(*, seal_path: Path, comparator_csv: Path, output: Path) -> dict[str, Any]:
    """Join public development targets/comparators only after own predictions are sealed."""
    seal = read_json_shared(seal_path)
    predictions = Path(seal["predictions_path"])
    if (
        seal.get("status") != "COMPLETE"
        or seal.get("labels_read") is not False
        or seal.get("predictions_sha256") != sha256_file(predictions)
    ):
        raise ValueError("own predictions are not immutably sealed")
    with predictions.open("r", encoding="utf-8", newline="") as stream:
        own = {row["query_id"]: row for row in csv.DictReader(stream)}
    with comparator_csv.open("r", encoding="utf-8", newline="") as stream:
        public = list(csv.DictReader(stream))
    if set(own) != {row["query_id"] for row in public}:
        raise ValueError("sealed predictions and public comparator population differ")
    fields = list(PREDICTION_FIELDS) + [name for name in public[0] if name not in PREDICTION_FIELDS]
    joined = [{**own[row["query_id"]], **row} for row in public]
    output.mkdir(parents=True, exist_ok=True)
    scored = output / "SCORED_PREDICTIONS.csv"
    _write_csv(scored, joined, fields)
    truth = np.asarray([float(row["truth_ttc_seconds"]) for row in public], np.float64)
    groups = np.asarray([row["sequence_id"] for row in public]).astype(str)
    metrics: dict[str, Any] = {}
    for name in (
        "E_H1_MATCHED",
        "E_CTX_MATCHED",
        "R_H1",
        "R_CTX",
        "F_TRUE",
        "F_ZERO",
        "public_Garl_event_lhr",
        "public_Garl_rgb_event_full",
    ):
        if name not in joined[0]:
            continue
        prediction = np.asarray(
            [float(row[name]) if row[name] not in {"", "nan"} else np.nan for row in joined]
        )
        legacy = native_metrics(prediction, truth)
        native = score_signed_ttc(truth, prediction, groups)
        common = common_cap60_metrics(truth, prediction, groups)
        metrics[name] = {
            "native": native,
            "common_cap60": common,
            "legacy_native": legacy,
            # Stable aliases for compact queue summaries.
            "rows": legacy["finite_predictions_on_labels"],
            "native_mae_s": legacy["mae_seconds"],
        }
    prediction_status = Counter(str(row["prediction_status"]) for row in joined)
    rgb_reasons = Counter(
        str(row["rgb_unavailable_reason"]) for row in joined if str(row["rgb_unavailable_reason"])
    )
    row_accounting = {
        "population": len(joined),
        "prediction_status_counts": dict(sorted(prediction_status.items())),
        "rgb_available_count": sum(
            str(row["rgb_available"]).lower() in {"true", "1"} for row in joined
        ),
        "rgb_unavailable_count": sum(
            str(row["rgb_available"]).lower() not in {"true", "1"} for row in joined
        ),
        "rgb_unavailable_reason_counts": dict(sorted(rgb_reasons.items())),
        "rows_omitted": 0,
    }
    result = {
        "schema": "rgb_port_dev32_score_v1",
        "status": "COMPLETE",
        "scope": "historically_exposed_exploratory_transfer_not_holdout_or_sota",
        "no_recipe_adjustment_from_scores": True,
        "optimizer_updates": 0,
        "prediction_seal_sha256": sha256_file(seal_path),
        "comparator_csv_sha256": sha256_file(comparator_csv),
        "scored_csv": str(scored.resolve()),
        "scored_csv_sha256": sha256_file(scored),
        "metrics": metrics,
        "row_accounting": row_accounting,
        "metric_views": {
            "native": "unclipped predictions",
            "common_cap60": "same rows with finite predictions clipped to [-60,60]",
        },
        "preprocessing_contracts": {
            "own_rgb_port": seal.get("own_rgb_preprocessing", OWN_RGB_PREPROCESSING),
            "published_garl_comparator": PUBLISHED_GARL_PREPROCESSING,
        },
        "official_comparison_admitted": False,
    }
    atomic_write_json(output / "RESULT.json", result)
    return result


def block_external(*, output: Path, branch: str, reason: str) -> dict[str, Any]:
    """Record a specific external-input block without affecting independent P/H/V work."""
    result = {
        "schema": "rgb_port_transfer_external_status_v1",
        "status": "BLOCKED_EXTERNAL",
        "branch": branch,
        "reason": reason,
        "optimizer_updates": 0,
        "independent_p_h_v_unaffected": True,
    }
    atomic_write_json(output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict-dev32")
    for name in (
        "run",
        "config",
        "p-manifest",
        "endpoint-freeze",
        "query-manifest",
        "event-normalizer",
        "rgb-normalizer",
        "output",
    ):
        predict.add_argument(f"--{name}", type=Path, required=True)
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score = commands.add_parser("score-dev32")
    score.add_argument("--seal", type=Path, required=True)
    score.add_argument("--comparator-csv", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    blocked = commands.add_parser("block-external")
    blocked.add_argument("--output", type=Path, required=True)
    blocked.add_argument("--branch", required=True)
    blocked.add_argument("--reason", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "predict-dev32":
        result = predict_dev32(
            run=args.run,
            config=args.config,
            p_manifest=args.p_manifest,
            endpoint_freeze=args.endpoint_freeze,
            query_manifest=args.query_manifest,
            event_normalizer=args.event_normalizer,
            rgb_normalizer=args.rgb_normalizer,
            output=args.output,
            device=args.device,
        )
    elif args.command == "score-dev32":
        result = score_dev32(
            seal_path=args.seal, comparator_csv=args.comparator_csv, output=args.output
        )
    else:
        result = block_external(output=args.output, branch=args.branch, reason=args.reason)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["block_external", "main", "parser", "predict_dev32", "score_dev32"]

"""Run frozen V13 endpoints with separately sealed FCWD RGB registration."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_union
from e_jepa_ttc.rgb_port.data import crop_uint8
from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from operational.rgb_port import transfer as engine
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_external_fix.contracts import validate
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward
from operational.rgb_port_revision.transfer_inputs import EVENT_LAGS_US, sensor_triplets
from operational.sota_eval.fcwd_inputs import FCWDEventReader, _validate_row
from operational.sota_eval.fcwd_run import validate_manifest

from .prepare import source_bindings


def prepare_event(row: dict[str, Any]) -> dict[str, np.ndarray]:
    """Preserve V13's four observations and native three-window producer input."""
    path = _validate_row(row)
    windows = np.asarray(row["windows_us"], np.int64)
    if windows.shape != (3, 2) or not np.all(windows[:, 1] - windows[:, 0] == 100_000):
        raise ValueError("Three 100ms windows required")
    if not np.all(np.diff(windows[:, 1]) == 100_000) or windows[-1, 1] != row["anchor_us"]:
        raise ValueError("FCWD windows must end at the query on native cadence")
    with FCWDEventReader(path) as reader:
        value = encode_union(
            reader,  # pyright: ignore[reportArgumentType] -- same bounded reader interface
            windows,
            np.arange(15, -1, -1, dtype=np.int64) * 100_000,
            np.arange(16) >= 12,
            tuple(row["square_xyxy"]),
            sequence_id=row["sequence_id"],
            roi_size=128,
            event_pixel_diff=0.0,
            retained_bytes_max=1024 * 1024**2,
        )[-len(EVENT_LAGS_US) :]
    return {"own_events": value.numpy().astype(np.float32, copy=False)}


def rgb_observations(
    row: dict[str, Any], registration: dict[str, Any]
) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], str]:
    """Use one fixed mapped query ROI for every actual historical RGB frame."""
    path = _validate_row(row)
    if registration["query_id"] != row["query_id"]:
        raise ValueError("Wrong query registration")
    if not registration["accepted"]:
        return [], str(registration["reason"])
    query = int(row["anchor_us"])
    if registration["available_us"] > query or registration["rgb_sensor_us"] > query:
        raise ValueError("Noncausal RGB registration")
    square = registration["rgb_square_xyxy"]
    result = []
    with h5py.File(path, "r") as handle:
        data = handle["blackflys/right/data"]
        if not isinstance(data, h5py.Dataset):
            raise ValueError("RGB data is not a dataset")
        clock = np.asarray(handle["blackflys/right/ts"], np.int64)
        groups = sensor_triplets(clock, query)
        if not groups:
            return [], "INSUFFICIENT_CAUSAL_RGB_HISTORY"
        frames = {
            int(i): crop_uint8(np.asarray(data[int(i)], np.uint8), square)
            for i in np.unique(np.concatenate(groups))
        }
        for indices in groups:
            timestamps = clock[indices]
            tensor = np.ascontiguousarray(
                np.stack([frames[int(i)] for i in indices]).astype(np.float32) / 255
            )
            result.append(
                (
                    tensor,
                    {
                        "first_sensor_ts_us": int(timestamps[0]),
                        "anchor_us": int(timestamps[-1]),
                        "available_us": query,
                        "timestamps_us": timestamps.tolist(),
                        "delta_t_s": (np.diff(timestamps) / 1e6).tolist(),
                        "square_xyxy": square,
                        "registration": "query_local_fixed_across_history",
                    },
                )
            )
    return result, ""


def run(
    run_path: Path, manifest_path: Path, registration_path: Path, output: Path, device: str
) -> dict[str, Any]:
    """Reuse the exact sealed endpoint engine through scoped input adapters."""
    validate(run_path)
    manifest = read_json_shared(manifest_path)
    rows = validate_manifest(manifest)
    binding_path = registration_path / "BINDING.json"
    binding = read_json_shared(binding_path)
    seal_path = registration_path / "REGISTRATION_SEALED.json"
    registration_seal = read_json_shared(seal_path)
    if (
        binding["manifest_sha256"] != sha256_file(manifest_path)
        or binding["sources"] != source_bindings()
        or registration_seal["binding_sha256"] != sha256_file(binding_path)
        or registration_seal["population"] != len(rows)
        or binding["selected_ids"] != [r["query_id"] for r in rows]
    ):
        raise ValueError("Registration is not sealed for the complete current input population")
    registrations = {}
    for name, digest in registration_seal["query_sha256"].items():
        path = registration_path / "queries" / name
        if sha256_file(path) != digest:
            raise ValueError("Sealed query registration changed")
        value = read_json_shared(path)
        if value["binding_sha256"] != sha256_file(binding_path):
            raise ValueError("Query registration belongs to another binding")
        registrations[value["query_id"]] = value
    if set(registrations) != {r["query_id"] for r in rows}:
        raise ValueError("Registration identities differ from inference population")
    source = engine._source_bindings()
    source.update(source_bindings())
    contract = {
        "contract": "fcwd_query_local_rgb_registration_v1",
        "registration_seal_sha256": sha256_file(seal_path),
        "calibration_source_sha256": binding["calibration"]["source_sha256"],
        "camera": "blackflys/right",
        "rgb_available_policy": "accepted_registration_and_causal_history",
        "timestamps": "actual_sensor_timestamps_100ms_cadence_with_1ms_tolerance",
        "roi": "event_boxes_projected_to_RGB_then_shared_square_25pct_margin",
        "registration": "same_query_translation_for_all_history_frames",
        "spatial_model": "published_intrinsics_plus_image_estimated_local_translation",
        "metric_extrinsics": False,
        "labels_read": False,
        "scope": "exploratory_input_adaptation; oracle_event_ROI; no_official_Garl_or_SOTA_claim",
        "unavailable": "keep_query; RGB_heads_NaN; both_fusion_routes_fallback_to_E_CTX",
    }
    with ExitStack() as stack:
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        stack.enter_context(patch.object(engine, "_manifest", lambda _: (manifest, rows)))
        stack.enter_context(patch.object(engine, "prepare_event", prepare_event))
        stack.enter_context(
            patch.object(
                engine,
                "raw_rgb_observations",
                lambda row: rgb_observations(row, registrations[row["query_id"]]),
            )
        )
        stack.enter_context(patch.object(engine, "_source_bindings", lambda: source))
        stack.enter_context(patch.object(engine, "OWN_RGB_PREPROCESSING", contract))
        result = engine.predict_dev32(
            run=run_path,
            config=Path("configs/rgb_port/producers.json"),
            p_manifest=run_path / "P_MANIFEST.json",
            endpoint_freeze=run_path / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json",
            query_manifest=manifest_path,
            event_normalizer=run_path / "features/H/E_NORMALIZER.json",
            rgb_normalizer=run_path / "features/H/R_NORMALIZER.json",
            output=output,
            device=device,
        )
    result.update(
        schema="rgb_port_fcwd_registered_prediction_seal_v1",
        protocol_scope=contract["scope"],
        registration_seal_sha256=sha256_file(seal_path),
    )
    atomic_write_json(output / "PREDICTIONS_SEALED.json", result)
    return result


def main() -> None:
    """Predict only; scoring cannot adjust this registration or model recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    run(args.run, args.manifest, args.registration, args.output, args.device)


if __name__ == "__main__":
    main()

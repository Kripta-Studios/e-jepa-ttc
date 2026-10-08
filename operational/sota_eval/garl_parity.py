"""Audit Garl event-only/full adapters against pinned upstream eAP behavior.

The parity population is selected from three distinct TRAIN40 eAP sequences
without opening TTC, 3-D geometry, or other target columns.  Upstream
``preprocess_data`` is the reference; the comparison path uses the local frozen
helpers.  EvTTC-specific differences are recorded as transfer adaptations and
are not tuned from metric errors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from operational.efficient_context.common import digest
from operational.evttc_rgb_transfer.model import FullGarl
from operational.evttc_transfer.models import _PUBLIC_GARL_SHA256
from operational.train40_system.garl_predictions import native_model, release_ttc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "artifacts/train40_system_20261005"
DEFAULT_FULL = ROOT / "artifacts/evttc_rgb_transfer_20261008/public_garl"
DEFAULT_OUTPUT = ROOT / "artifacts/sota_campaign_20261008/garl_parity"
UPSTREAM_COMMIT = "256661242b8a7f5e56aa3c1c02348b30f6e89de6"
HF_REVISION = "b676fcdaf26c04bcf896cdb2b208c9c424e8462a"
LOCAL_EXECUTED_HELPERS = (
    ROOT / "src/e_jepa_ttc/data/garl_official_preprocessing.py",
    ROOT / "src/e_jepa_ttc/efficient_context/garl_input.py",
    ROOT / "src/e_jepa_ttc/simplex_t/cached_event_reader.py",
    ROOT / "operational/evttc_rgb_transfer/inputs.py",
)


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value without formatting dependence."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes(code_root: Path, campaign: Path) -> dict[str, str]:
    freeze = json.loads((campaign / "DELIVERY_FREEZE.json").read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for item in freeze["native_source_files"]}
    for relative, sha256 in expected.items():
        if digest(code_root / relative) != sha256:
            raise ValueError(f"Pinned upstream source changed: {relative}")
    return expected


def _local_source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): digest(path) for path in LOCAL_EXECUTED_HELPERS}


def _reproduction_environment() -> dict[str, Any]:
    return {
        "opencv_python_headless": "4.13.0.92",
        "isolated_dependency_target": (
            r"C:\Users\Álvaro Schwiedop\AppData\Local\Temp\garl_parity_cv2_cp311"
        ),
        "install_command": (
            "uv pip install --python '..\\e-jepa-ttc\\.venv\\Scripts\\python.exe' "
            "--target 'C:\\Users\\Álvaro Schwiedop\\AppData\\Local\\Temp\\"
            "garl_parity_cv2_cp311' --no-deps opencv-python-headless==4.13.0.92"
        ),
        "run_command_powershell": (
            "$env:PYTHONPATH='C:\\Users\\Álvaro Schwiedop\\AppData\\Local\\Temp\\"
            "garl_parity_cv2_cp311;src;.'; "
            "& '..\\e-jepa-ttc\\.venv\\Scripts\\python.exe' -m "
            "operational.sota_eval.garl_parity --run-models"
        ),
        "shared_environment_modified": False,
    }


def _write_audit_binding(
    output: Path, report: Mapping[str, Any], *, source_before_binding_sha256: str | None
) -> dict[str, Any]:
    records = report["records"]
    binding = {
        "artifact_type": "garl_parity_audit_binding_v1",
        "audit_source_sha256": digest(Path(__file__)),
        "source_before_binding_sha256": source_before_binding_sha256,
        "result_sha256": digest(output / "RESULT.json"),
        "executed_local_helpers": _local_source_hashes(),
        "passed_evidence": {
            "status": report["status"],
            "samples": report["samples"],
            "preprocessing_max_abs": report["preprocessing_max_abs"],
            "record_sensor_hashes": [record["reference_sensor_sha256"] for record in records],
            "all_records_bit_exact": all(record["bit_exact"] for record in records),
            "event_checkpoint_sha256": report["identities"]["event_checkpoint_sha256"],
            "full_checkpoint_sha256": report["identities"]["full_checkpoint_sha256"],
            "full_strict_state_load": report["models"]["full"]["strict_state_load"],
        },
        "reproduction_environment": _reproduction_environment(),
        "models_rerun_for_binding_only": False,
    }
    (output / "AUDIT_BINDING.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return binding


def bind_existing(output: Path) -> dict[str, Any]:
    """Add complete source lineage to passed evidence without repeating inference."""
    result_path = output / "RESULT.json"
    before = digest(result_path)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASSED"
        or int(report.get("samples", 0)) < 3
        or float(report.get("preprocessing_max_abs", float("inf"))) != 0.0
        or not all(record.get("bit_exact") for record in report.get("records", []))
        or report.get("models", {}).get("full", {}).get("strict_state_load") is not True
    ):
        raise ValueError("Only passed, bit-exact, strict-load evidence can be source-bound")
    report["identities"]["local_executed_helpers"] = _local_source_hashes()
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    binding = _write_audit_binding(output, report, source_before_binding_sha256=before)
    names = ("RESULT.json", "REPORT.md", "AUDIT_BINDING.json")
    (output / "SHA256.json").write_text(
        json.dumps({name: digest(output / name) for name in names}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return binding


def _select_rows(campaign: Path, eap_root: Path, count: int) -> list[dict[str, Any]]:
    """Select one deterministic label-free row from each available sequence."""
    columns = [
        "sequence_id",
        "sample_token",
        "public_track_id",
        "frame_timestamps_us",
        "rgb_shard_paths",
        "rgb_member_paths",
        "events_path",
        "event_windows_us",
        "boxes_xyxy",
    ]
    frame = pd.read_parquet(campaign / "TRAIN40_ROWS.parquet", columns=columns)
    rows: list[dict[str, Any]] = []
    for sequence in sorted(frame["sequence_id"].unique()):
        subset = frame[frame["sequence_id"] == sequence]
        raw_records = cast(Any, subset).to_dict("records")
        records = sorted(raw_records, key=lambda row: str(row["sample_token"]))
        for row in records:
            if (eap_root / str(row["events_path"])).is_file():
                rows.append(row)
                break
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"Only {len(rows)} of {count} distinct eAP sequences are available")
    return rows


def _list(value: object) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"Expected a list-like parquet value, got {type(value).__name__}")


def _box4(value: object) -> tuple[float, float, float, float]:
    items = _list(value)
    if len(items) != 4:
        raise ValueError(f"Expected four box coordinates, got {len(items)}")
    return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))


def _decode_rgb(eap_root: Path, shard: object, member: object) -> np.ndarray:
    """Mirror upstream HF image decode: OpenCV BGR followed by BGR->RGB."""
    import cv2  # pyright: ignore[reportMissingImports]

    with tarfile.open(eap_root / str(shard), "r") as archive:
        extracted = archive.extractfile(str(member))
        if extracted is None:
            raise FileNotFoundError(f"{member} in {shard}")
        encoded = np.frombuffer(extracted.read(), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to decode {member}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _reference_sensor(row: dict[str, Any], eap_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Call the pinned upstream dataset preprocessing as the reference path."""
    import torchvision.transforms as transforms
    from garl_ttc.datasets.ttc_dataset import preprocess_data
    from garl_ttc.utils.events import extract_from_h5_by_timewindow

    boxes = [tuple(int(v) for v in box) for box in _list(row["boxes_xyxy"])]
    windows = [tuple(int(v) for v in pair) for pair in _list(row["event_windows_us"])]
    images = [
        _decode_rgb(eap_root, shard, member)
        for shard, member in zip(
            _list(row["rgb_shard_paths"]), _list(row["rgb_member_paths"]), strict=True
        )
    ]
    height, width = images[-1].shape[:2]
    events = extract_from_h5_by_timewindow(
        eap_root / str(row["events_path"]),
        [window[0] for window in windows],
        [window[1] for window in windows],
        5,
        [height, width],
    )
    max_edge = max(max(box[2] - box[0], box[3] - box[1]) for box in boxes)
    timestamps = [int(value) for value in _list(row["frame_timestamps_us"])]
    tracks = [
        {
            "public_track_id": str(row["public_track_id"]),
            "timestamp": timestamp,
            "box": list(box),
            "max_edge": max_edge,
            "seq_name": "",
        }
        for timestamp, box in zip(timestamps, boxes, strict=True)
    ]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    result = preprocess_data(
        {"images": images, "events": events, "tracks": tracks, "masks": [None, None]},
        [128, 128],
        transform,
        fy=1694.1323524131867,
        data_mode="image_event",
        logger=logging.getLogger("garl_parity"),
        require_labels=False,
        sample_token=str(row["sample_token"]),
    )
    if result is None:
        raise RuntimeError("Pinned upstream preprocessing rejected parity row")
    sensor = result["data"][0].numpy().astype(np.float32, copy=False)
    if sensor.shape != (46, 128, 128):
        raise ValueError(f"Unexpected upstream sensor shape: {sensor.shape}")
    return sensor, {
        "boxes": boxes,
        "windows_us": windows,
        "event_counts": [len(event["t"]) for event in events],
        "rgb_shapes": [list(image.shape) for image in images],
    }


def _local_sensor(
    row: dict[str, Any], eap_root: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use frozen local helpers, independently of upstream preprocess_data."""
    from e_jepa_ttc.data.garl_official_preprocessing import (
        official_resize_roi,
        official_square_box,
    )
    from e_jepa_ttc.efficient_context.garl_input import inference_record
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from operational.evttc_rgb_transfer.inputs import _normalized_rgb

    boxes = [_box4(box) for box in _list(row["boxes_xyxy"])]
    images = [
        _decode_rgb(eap_root, shard, member)
        for shard, member in zip(
            _list(row["rgb_shard_paths"]), _list(row["rgb_member_paths"]), strict=True
        )
    ]
    normalized = np.concatenate([_normalized_rgb(image) for image in images], axis=0)
    rgb_square = official_square_box(boxes, 1)
    rgb = official_resize_roi(normalized, rgb_square, (128, 128)).numpy()
    pool = ReaderPool()
    try:
        event = inference_record(
            {
                "sequence_id": str(row["sequence_id"]),
                "event_windows_us": _list(row["event_windows_us"]),
                "boxes_xyxy": _list(row["boxes_xyxy"]),
            },
            pool,
            eap_root / "data/train",
        ).numpy()
    finally:
        pool.close()
    full = np.concatenate((rgb, event), axis=0).astype(np.float32, copy=False)
    return full, event.astype(np.float32, copy=False), {"rgb_square": list(rgb_square)}


def _evttc_diagnostic(manifest_path: Path) -> dict[str, Any]:
    """Describe transfer-only timing differences without choosing from metrics."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", manifest.get("queries"))
    if not isinstance(rows, list):
        raise ValueError("EvTTC query manifest must contain a rows list")
    endpoints = [pair for row in rows for pair in row["windows_us"][-2:]]
    non_ms = sum(int(start) % 1000 != 0 or int(end) % 1000 != 0 for start, end in endpoints)
    start_shifts = [-(int(start) % 1000) for start, _ in endpoints]
    end_shifts = [-(int(end) % 1000) for _, end in endpoints]
    duration_changes = [
        ((int(end) // 1000) - (int(start) // 1000)) * 1000 - (int(end) - int(start))
        for start, end in endpoints
    ]
    return {
        "queries": len(rows),
        "endpoint_windows": len(endpoints),
        "non_millisecond_aligned_endpoint_windows": non_ms,
        "current_transfer_reader": "exact half-open [start_us,end_us)",
        "eap_upstream_reader": "ms_to_idx[floor(start/1000):floor(end/1000)]",
        "millisecond_floor_quantization_us": {
            "start_shift_range": [min(start_shifts), max(start_shifts)],
            "end_shift_range": [min(end_shifts), max(end_shifts)],
            "mean_start_shift": float(np.mean(start_shifts)),
            "mean_end_shift": float(np.mean(end_shifts)),
            "duration_change_range": [min(duration_changes), max(duration_changes)],
        },
        "event_pixel_diff": {
            "eap_upstream": 5,
            "evttc_transfer": 0,
            "interpretation": (
                "dataset-specific sensor registration; +5 is not transferred because EvTTC uses "
                "its explicit RGB-to-event calibration"
            ),
        },
        "not_selected_from_metric_errors": True,
    }


def _model_predictions(
    sensors: list[np.ndarray], campaign: Path, full_public: Path, code_root: Path
) -> dict[str, Any]:
    """Exercise each strict native endpoint sequentially with at most two CPU threads."""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    event_model = native_model(campaign, code_root)
    event_heights = []
    with torch.inference_mode():
        for sensor in sensors:
            heights, _ = event_model.forward_test(torch.from_numpy(sensor[None, 6:]))
            event_heights.append(heights.float().numpy()[0])
    event_array = np.asarray(event_heights, np.float32)
    event_ttc = release_ttc(event_array, float(event_model.dT))
    del event_model
    gc.collect()

    full_model = FullGarl(full_public, code_root, "cpu")
    full_rows = [full_model.predict_sensor(sensor) for sensor in sensors]
    result = {
        "event_only": {
            "checkpoint_sha256": _PUBLIC_GARL_SHA256,
            "dT_seconds": 0.1,
            "heights": event_array.tolist(),
            "ttc_seconds": event_ttc.tolist(),
        },
        "full": {
            "checkpoint_sha256": full_model.bindings["checkpoint_sha256"],
            "dT_seconds": full_model.bindings["native_delta_t_s"],
            "outputs": full_rows,
            "strict_state_load": full_model.bindings["strict_state_load"],
        },
        "output_order": ["visible_height_t0", "visible_height_t1"],
        "ttc_equation": "dT / (1 - height_t0 / height_t1)",
        "ttc_units": "seconds",
        "clipping": "none",
        "sign": "signed; denominator sign is preserved",
    }
    del full_model
    gc.collect()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded parity audit and publish reproducible evidence."""
    code_root = args.code_root.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    campaign = args.campaign.resolve()
    full_public = args.full_public.resolve()
    sources = _source_hashes(code_root, campaign)
    rows = _select_rows(campaign, args.eap_root.resolve(), args.samples)
    records, sensors = [], []
    for row in rows:
        reference, reference_meta = _reference_sensor(row, args.eap_root.resolve())
        local, event_only, local_meta = _local_sensor(row, args.eap_root.resolve())
        rgb_error = float(np.max(np.abs(reference[:6] - local[:6])))
        event_error = float(np.max(np.abs(reference[6:] - event_only)))
        full_error = float(np.max(np.abs(reference - local)))
        records.append(
            {
                "sequence_id": str(row["sequence_id"]),
                "sample_token": str(row["sample_token"]),
                "reference_sensor_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "local_sensor_sha256": hashlib.sha256(local.tobytes()).hexdigest(),
                "rgb_max_abs": rgb_error,
                "event_max_abs": event_error,
                "full_max_abs": full_error,
                "bit_exact": bool(np.array_equal(reference, local)),
                "reference": reference_meta,
                "local": local_meta,
            }
        )
        sensors.append(local)
    max_error = max(record["full_max_abs"] for record in records)
    models = (
        _model_predictions(sensors, campaign, full_public, code_root)
        if args.run_models
        else {"status": "NOT_RUN"}
    )
    evttc = _evttc_diagnostic(args.evttc_manifest.resolve())
    report: dict[str, Any] = {
        "artifact_type": "sota_garl_native_parity_v1",
        "status": "PASSED" if max_error == 0.0 else "FAILED",
        "samples": len(records),
        "sequence_coverage": [record["sequence_id"] for record in records],
        "preprocessing_reference": "pinned upstream garl_ttc.datasets.ttc_dataset.preprocess_data",
        "comparison_path": "frozen local normalization/ROI/timevolume helpers",
        "preprocessing_max_abs": max_error,
        "records": records,
        "models": models,
        "contracts": {
            "input_channels": "RGB0[3],RGB1[3],EVENT0[20],EVENT1[20]",
            "rgb": "RGB uint8 -> /255 -> ImageNet normalize -> shared square crop",
            "events": (
                "x+5 eAP registration, full-sensor clip, 20-plane polarity-agnostic "
                "timevolume per endpoint, endpoint square crop"
            ),
            "polarity": "not read or consumed by the published Garl timevolume",
            "crop": "integer center, max edge shared over both endpoints, ceil boundaries",
            "resize": "bilinear grid_sample align_corners=True with endpoint-inclusive linspace",
            "targets_read": False,
            "optimizer_updates": 0,
        },
        "evttc_transfer_diagnostic": evttc,
        "findings": [
            {
                "severity": "INFO",
                "finding": "eAP local adapters are bit-exact with upstream preprocessing",
                "supported": max_error == 0.0,
            },
            {
                "severity": "INFO",
                "finding": "published event representation is polarity-agnostic",
                "impact": "polarity is intentionally absent from both EO and full Garl inputs",
            },
            {
                "severity": "LIMITATION",
                "finding": (
                    "EvTTC exact timing and calibrated sensor registration are transfer "
                    "adaptations, "
                    "not an upstream EvTTC protocol"
                ),
                "action": "retain declaration; do not choose +5 or ms-flooring from scored errors",
            },
        ],
        "identities": {
            "upstream_commit": UPSTREAM_COMMIT,
            "hf_revision": HF_REVISION,
            "native_sources": sources,
            "campaign_index_sha256": digest(campaign / "TRAIN40_INDEX.npz"),
            "campaign_rows_sha256": digest(campaign / "TRAIN40_ROWS.parquet"),
            "event_checkpoint_sha256": _PUBLIC_GARL_SHA256,
            "full_checkpoint_sha256": digest(full_public / "paper_ours_full.pth"),
            "full_config_sha256": digest(full_public / "configs/ablation/ours_full.yaml"),
            "script_sha256": digest(Path(__file__)),
            "local_executed_helpers": _local_source_hashes(),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "RESULT.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = (
        "# Garl native parity\n\n"
        f"Status: **{report['status']}**. Samples: {len(records)} across "
        f"{len(report['sequence_coverage'])} eAP sequences. Maximum preprocessing error: "
        f"`{max_error}`.\n\n"
        "The event-only and full adapters use the published height order `(t0,t1)`, "
        "`dT=0.1 s`, and the unclipped signed equation "
        "`TTC=dT/(1-height_t0/height_t1)`. The published timevolume ignores polarity.\n\n"
        "EvTTC uses exact microsecond windows and explicit cross-camera calibration. These are "
        "declared transfer adaptations; the audit does not select eAP's `x+5` or millisecond "
        "flooring based on EvTTC errors.\n"
    )
    (args.output / "REPORT.md").write_text(markdown, encoding="utf-8")
    artifact_names = ["RESULT.json", "REPORT.md"]
    if args.run_models:
        _write_audit_binding(args.output, report, source_before_binding_sha256=None)
        artifact_names.append("AUDIT_BINDING.json")
    manifest = {name: digest(args.output / name) for name in artifact_names}
    (args.output / "SHA256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--eap-root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--code-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--full-public", type=Path, default=DEFAULT_FULL)
    parser.add_argument(
        "--evttc-manifest",
        type=Path,
        default=ROOT / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--bind-existing", action="store_true")
    args = parser.parse_args()
    if args.bind_existing:
        binding = bind_existing(args.output.resolve())
        print(json.dumps(binding, indent=2))
        return 0
    if args.samples < 3:
        raise ValueError("At least three distinct eAP sequences are required")
    report = run(args)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

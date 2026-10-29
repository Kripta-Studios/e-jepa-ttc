"""Audit a release-semantic Garl cache against the frozen release functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garl_official_preprocessing import official_square_box  # noqa: E402
from e_jepa_ttc.data.garl_release_cache import (  # noqa: E402
    GarlReleaseCacheDataset,
)
from e_jepa_ttc.data.garlttc_eap import (  # noqa: E402
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.data.garlttc_lhr_cache import select_temporal_indices  # noqa: E402


def _as_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _nested_float(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_nested_float(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_nested_float(item) for item in value]
    return float(cast(Any, value)) if value is not None else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_release_rgb(
    eap_root: Path,
    shard: object,
    member: object,
    square: tuple[int, int, int, int],
    target_size: int,
    get_target_roi_from_feature_torch: Callable[..., torch.Tensor],
) -> np.ndarray:
    with tarfile.open(eap_root / str(shard), "r") as archive:
        extracted = archive.extractfile(str(member))
        if extracted is None:
            raise FileNotFoundError(f"{member} in {shard}")
        encoded = np.frombuffer(extracted.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV failed to decode {member} in {shard}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    feature = image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
    normalized = torch.from_numpy((feature - mean) / std)[None]
    height, width = feature.shape[1:]
    cropped = get_target_roi_from_feature_torch(
        normalized,
        list(square),
        [target_size, target_size],
    )
    del height, width
    return cropped[0].numpy()


def _read_release_rgb_shape(
    eap_root: Path,
    shard: object,
    member: object,
) -> tuple[int, int]:
    """Return the official RGB ``(width, height)`` sensor shape."""

    with tarfile.open(eap_root / str(shard), "r") as archive:
        extracted = archive.extractfile(str(member))
        if extracted is None:
            raise FileNotFoundError(f"{member} in {shard}")
        encoded = np.frombuffer(extracted.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV failed to decode {member} in {shard}")
    height, width = image_bgr.shape[:2]
    return int(width), int(height)


def _release_input(
    row: dict[str, object],
    *,
    eap_root: Path,
    extract_from_h5_by_timewindow: Callable[..., list[dict[str, np.ndarray]]],
    get_timevolume_roi_np: Callable[..., tuple[np.ndarray, np.ndarray]],
    get_target_roi_from_feature_torch: Callable[..., torch.Tensor],
) -> dict[str, np.ndarray | float]:
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    frame_timestamps = [int(cast(Any, value)) for value in _as_list(row["frame_timestamps_us"])]
    first, second, _ = select_temporal_indices(
        frame_timestamps,
        anchor_timestamp_us=int(cast(Any, row["timestamp_us"])),
        target_delta_t_s=0.1,
        tolerance_s=0.025,
        context_delta_t_s=0.1,
        context_tolerance_s=0.05,
    )
    endpoint_indices = (first, second)
    squares = [official_square_box(boxes, index) for index in endpoint_indices]
    shards = _as_list(row["rgb_shard_paths"])
    members = _as_list(row["rgb_member_paths"])
    sensor_width, sensor_height = _read_release_rgb_shape(eap_root, shards[-1], members[-1])
    event_path = resolve_eap_events_path(eap_root, str(row["events_path"]))
    event_list = extract_from_h5_by_timewindow(
        str(event_path),
        [int(windows[index][0]) for index in endpoint_indices],
        [int(windows[index][1]) for index in endpoint_indices],
        5,
        [sensor_height, sensor_width],
    )
    event_planes: list[np.ndarray] = []
    for event, square in zip(event_list, squares, strict=True):
        feature, _ = get_timevolume_roi_np(
            expand_box=list(square),
            x=event["x"],
            y=event["y"],
            tus=event["t"],
            number_of_planes=20,
        )
        source = torch.from_numpy(feature)[None]
        _, _, height, width = source.shape
        event_planes.append(
            get_target_roi_from_feature_torch(
                source,
                [0, 0, width, height],
                [128, 128],
            )[0].numpy()
        )

    frame_ttc = _as_list(row["frame_ttc"])
    corners = _as_list(row["box3d_Fcam"])
    box3d_h = float(cast(Any, row["box3d_h"]))
    max_edge = max(
        official_square_box(boxes, index)[3] - official_square_box(boxes, index)[1]
        for index in range(len(boxes))
    )
    visible: list[float] = []
    for index in endpoint_indices:
        points = np.asarray(_nested_float(corners[index]), dtype=np.float64).reshape(-1, 3)
        visible.append(1694.1323524131867 * box3d_h / float(points[:, 2].min()) * (128 / max_edge))

    visual_square = squares[-1]
    rgb_pair = np.stack(
        [
            _read_release_rgb(
                eap_root,
                shards[index],
                members[index],
                visual_square,
                128,
                get_target_roi_from_feature_torch,
            )
            for index in endpoint_indices
        ]
    )
    return {
        "event_roi": np.concatenate(event_planes, axis=0),
        "rgb_pair": rgb_pair,
        "ttc_s": float(cast(Any, frame_ttc[second])),
        "visible_height": np.asarray(visible, dtype=np.float32),
        "delta_t_s": (frame_timestamps[second] - frame_timestamps[first]) * 1e-6,
        "sensor_width": sensor_width,
        "sensor_height": sensor_height,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_split <= 0:
        raise ValueError("--samples-per-split must be positive.")

    release_root = args.release_root.resolve()
    sys.path.insert(0, str(release_root))
    from garl_ttc.datasets.event_representation import (  # type: ignore[reportMissingImports]  # noqa: PLC0415
        get_timevolume_roi_np,
    )
    from garl_ttc.datasets.ttc_dataset import (  # type: ignore[reportMissingImports]  # noqa: PLC0415
        get_target_roi_from_feature_torch,
    )
    from garl_ttc.utils.events import (  # type: ignore[reportMissingImports]  # noqa: PLC0415
        extract_from_h5_by_timewindow,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    assignments = split_payload["assignments"]
    sequences = sorted(
        str(sequence) for role in ("train", "validation") for sequence in assignments[role]
    )
    index = load_garlttc_train_index(args.garlttc_root, sequences)
    rows = cast(Any, index.merged).to_dict("records")
    row_by_token = {str(row["sample_token"]): row for row in rows}

    tolerances = {
        "event_roi": 1.0 / 65535.0 + 1e-5,
        "rgb_pair": 2e-3,
        "ttc_s": 1e-6,
        "visible_height": 1e-5,
        "delta_t_s": 1e-6,
        "sensor_width": 0.0,
        "sensor_height": 0.0,
    }
    stats = {name: {"max_abs": 0.0, "mean_abs_sum": 0.0, "count": 0} for name in tolerances}
    failures: list[dict[str, object]] = []
    checked = 0
    for split in ("train", "validation"):
        dataset = GarlReleaseCacheDataset(args.manifest, split=split)
        for dataset_index in range(min(len(dataset), args.samples_per_split)):
            cached = dataset[dataset_index]
            token = str(cached["sample_token"])
            row = row_by_token[token]
            expected = _release_input(
                row,
                eap_root=args.eap_root,
                extract_from_h5_by_timewindow=extract_from_h5_by_timewindow,
                get_timevolume_roi_np=get_timevolume_roi_np,
                get_target_roi_from_feature_torch=get_target_roi_from_feature_torch,
            )
            for name, tolerance in tolerances.items():
                actual_value = cached[name] if name in cached else None
                if actual_value is None:
                    raise ValueError(f"Cache row {token} has no required field {name!r}.")
                actual = (
                    actual_value.detach().cpu().numpy()
                    if isinstance(actual_value, torch.Tensor)
                    else np.asarray(actual_value)
                )
                difference = np.abs(actual.astype(np.float64) - np.asarray(expected[name]))
                row_stats = stats[name]
                row_stats["max_abs"] = max(float(row_stats["max_abs"]), float(difference.max()))
                row_stats["mean_abs_sum"] += float(difference.mean())
                row_stats["count"] += 1
                if float(difference.max()) > tolerance:
                    failures.append(
                        {
                            "split": split,
                            "sample_token": token,
                            "field": name,
                            "max_abs": float(difference.max()),
                            "tolerance": tolerance,
                        }
                    )
            checked += 1

    for _name, row_stats in stats.items():
        row_stats["mean_abs"] = row_stats["mean_abs_sum"] / max(int(row_stats["count"]), 1)
        del row_stats["mean_abs_sum"]
    payload: dict[str, object] = {
        "artifact_type": "garl_release_cache_parity_v1",
        "status": "pass" if not failures else "fail",
        "manifest": args.manifest.resolve().as_posix(),
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "release_root": release_root.as_posix(),
        "release_semantics": manifest.get("release_semantics"),
        "samples_per_split": args.samples_per_split,
        "checked_samples": checked,
        "tolerances": tolerances,
        "fields": stats,
        "failures": failures[:100],
        "failure_count": len(failures),
        "negative_result_preserved": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

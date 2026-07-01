"""EvTTC dataset discovery, manifests, and lazy HDF5 event access."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.targets import load_ttc_csv
from e_jepa_ttc.data.types import DatasetSequence, EventBatch
from e_jepa_ttc.data.validation import normalize_polarity, validate_event_batch
from e_jepa_ttc.utils.io import read_structured, write_structured


@dataclass(frozen=True)
class HDF5DatasetInfo:
    """Small HDF5 dataset descriptor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None


@dataclass(frozen=True)
class HDF5EventLayout:
    """Discovered HDF5 event field layout."""

    kind: str
    x: str | None = None
    y: str | None = None
    t: str | None = None
    p: str | None = None
    ms_map_idx: str | None = None
    compound: str | None = None
    width: int | None = None
    height: int | None = None


NAVIGATION_FEATURE_NAMES = (
    "ego_speed",
    "ego_velocity_x",
    "ego_velocity_y",
    "ego_velocity_z",
    "ego_acceleration_x",
    "ego_acceleration_y",
    "ego_acceleration_z",
    "ego_yaw_rate",
    "ego_navigation_valid",
)


def _sequence_id_from_path(root: Path, sequence_dir: Path) -> str:
    parts = sequence_dir.relative_to(root).parts
    return "-".join(part.replace("%", "") for part in parts)


def _speed_bucket(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        lower = part.lower()
        if "low" in lower:
            return "low"
        if "medium" in lower:
            return "medium"
        if "high" in lower:
            return "high"
    return None


def _target_type(parts: tuple[str, ...]) -> str | None:
    if not parts:
        return None
    family = parts[0]
    if family.startswith("CC"):
        return "car"
    if family.startswith("CP"):
        return "pedestrian"
    return None


def scan_evttc_root(root: str | Path) -> list[DatasetSequence]:
    """Scan an EvTTC root for local sequence folders."""

    root_path = Path(root)
    if not root_path.exists():
        msg = f"EvTTC root does not exist: {root_path}"
        raise FileNotFoundError(msg)

    sequences: list[DatasetSequence] = []
    for ttc_csv in sorted(root_path.rglob("ttc.csv")):
        sequence_dir = ttc_csv.parent
        hdf5_files = sorted(sequence_dir.glob("*.hdf5"))
        event_candidates = [path for path in hdf5_files if path.name.lower() != "gt.hdf5"]
        if not event_candidates:
            continue
        event_hdf5 = max(event_candidates, key=lambda path: path.stat().st_size)
        gt_hdf5 = sequence_dir / "gt.hdf5"
        left_label_dir = sequence_dir / "leftlabel"
        bbox_label_dir = sequence_dir / "bbox_segmentation"
        label_dir = left_label_dir if left_label_dir.exists() else bbox_label_dir
        rel_parts = sequence_dir.relative_to(root_path).parts
        sequence_id = _sequence_id_from_path(root_path, sequence_dir)
        size_bytes = sum(path.stat().st_size for path in hdf5_files)
        sequences.append(
            DatasetSequence(
                dataset="EvTTC",
                sequence_id=sequence_id,
                local_path=sequence_dir.as_posix(),
                event_hdf5=event_hdf5.name,
                gt_hdf5=gt_hdf5.name if gt_hdf5.exists() else None,
                ttc_csv=ttc_csv.name,
                label_dir=label_dir.name if label_dir.exists() else None,
                scenario_family=rel_parts[0] if rel_parts else None,
                speed_bucket=_speed_bucket(rel_parts),
                target_type=_target_type(rel_parts),
                split_group=sequence_id,
                size_bytes=size_bytes,
                original_filename=event_hdf5.name,
                remote_name=sequence_id,
                extra={
                    "relative_parts": list(rel_parts),
                    "label_count": len(list(label_dir.glob("*.json"))) if label_dir.exists() else 0,
                },
            )
        )
    return sequences


def manifest_to_dict(sequences: list[DatasetSequence]) -> dict[str, Any]:
    """Serialize a sequence manifest."""

    return {
        "dataset": "EvTTC",
        "version": "local-handoff-2026-06-30",
        "notes": "Local smoke subset discovered from datasets/evttc.",
        "sequences": [sequence.to_dict() for sequence in sequences],
    }


def write_manifest(path: str | Path, sequences: list[DatasetSequence]) -> None:
    """Write a dataset manifest."""

    write_structured(path, manifest_to_dict(sequences))


def read_manifest(path: str | Path) -> list[DatasetSequence]:
    """Read a dataset manifest."""

    data = read_structured(path)
    raw_sequences = data.get("sequences")
    if not isinstance(raw_sequences, list):
        msg = f"Manifest {path} does not contain a 'sequences' list."
        raise ValueError(msg)
    return [DatasetSequence.from_dict(item) for item in raw_sequences]


def describe_hdf5(path: str | Path) -> list[HDF5DatasetInfo]:
    """Return dataset names, shapes and dtypes without loading arrays."""

    import h5py

    infos: list[HDF5DatasetInfo] = []
    with h5py.File(path, "r") as h5:

        def visit(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                infos.append(
                    HDF5DatasetInfo(
                        name=name,
                        shape=tuple(int(value) for value in obj.shape),
                        dtype=str(obj.dtype),
                        chunks=tuple(int(value) for value in obj.chunks) if obj.chunks else None,
                    )
                )

        h5.visititems(visit)
    return infos


def _infer_resolution(
    attrs: dict[str, object],
    infos: list[HDF5DatasetInfo],
) -> tuple[int | None, int | None]:
    width = attrs.get("width") or attrs.get("sensor_width") or attrs.get("W")
    height = attrs.get("height") or attrs.get("sensor_height") or attrs.get("H")
    if width is not None and height is not None:
        return int(width), int(height)
    for info in infos:
        lower = info.name.lower()
        if lower.endswith("image") and len(info.shape) >= 2:
            return int(info.shape[-1]), int(info.shape[-2])
    return None, None


def _read_resolution_dataset(h5: object, parent: str) -> tuple[int | None, int | None]:
    resolution_path = f"{parent}/calib/resolution" if parent else "calib/resolution"
    if resolution_path not in h5:
        return None, None
    values = h5[resolution_path][:]
    if len(values) != 2:
        return None, None
    return int(values[0]), int(values[1])


def discover_event_layout(path: str | Path) -> HDF5EventLayout | None:
    """Discover common event layouts in an HDF5 file."""

    import h5py

    infos = describe_hdf5(path)
    if not infos:
        return None

    with h5py.File(path, "r") as h5:
        root_attrs = dict(h5.attrs)
        attrs_by_parent: dict[str, dict[str, Any]] = {}
        for info in infos:
            parent = str(Path(info.name).parent).replace("\\", "/")
            parent = "" if parent == "." else parent
            if parent in h5:
                attrs_by_parent[parent] = dict(h5[parent].attrs)

        for info in infos:
            dtype = h5[info.name].dtype
            if dtype.fields:
                fields = {field.lower(): field for field in dtype.fields}
                x_key = fields.get("x")
                y_key = fields.get("y")
                t_key = fields.get("t") or fields.get("ts") or fields.get("timestamp")
                p_key = fields.get("p") or fields.get("polarity")
                if x_key and y_key and t_key and p_key:
                    width, height = _infer_resolution(root_attrs, infos)
                    return HDF5EventLayout(
                        kind="compound",
                        compound=info.name,
                        x=x_key,
                        y=y_key,
                        t=t_key,
                        p=p_key,
                        width=width,
                        height=height,
                    )

    by_parent: dict[str, dict[str, str]] = {}
    for info in infos:
        parts = info.name.split("/")
        parent = "/".join(parts[:-1])
        basename = parts[-1].lower()
        by_parent.setdefault(parent, {})[basename] = info.name

    for parent, names in by_parent.items():
        x = names.get("x") or names.get("xs")
        y = names.get("y") or names.get("ys")
        t = names.get("t") or names.get("ts") or names.get("timestamp") or names.get("timestamps")
        p = names.get("p") or names.get("polarity") or names.get("polarities")
        ms_map_idx = names.get("ms_map_idx")
        if x and y and t and p:
            import h5py

            with h5py.File(path, "r") as h5:
                attrs = dict(h5[parent].attrs) if parent in h5 else dict(h5.attrs)
                width, height = _read_resolution_dataset(h5, parent)
            if width is None or height is None:
                width, height = _infer_resolution(attrs, infos)
            return HDF5EventLayout(
                kind="separate",
                x=x,
                y=y,
                t=t,
                p=p,
                ms_map_idx=ms_map_idx,
                width=width,
                height=height,
            )

    return None


def _slice_bounds_from_ms_map(
    ms_map_idx: object,
    *,
    event_count: int,
    t_start_us: int,
    t_end_us: int,
) -> tuple[int, int]:
    if len(ms_map_idx) == 0:
        return 0, event_count
    start_lookup = max(0, min(int(t_start_us // 1000), len(ms_map_idx) - 1))
    end_lookup_raw = int(t_end_us // 1000) + 2
    if end_lookup_raw >= len(ms_map_idx):
        return int(ms_map_idx[start_lookup]), event_count
    return int(ms_map_idx[start_lookup]), int(ms_map_idx[end_lookup_raw])


def _refine_bounds(
    timestamps: np.ndarray,
    rough_start: int,
    t_start_us: int,
    t_end_us: int,
) -> tuple[int, int]:
    local_start, local_end = np.searchsorted(timestamps, [t_start_us, t_end_us], side="left")
    return int(rough_start + local_start), int(rough_start + local_end)


def read_events_window(
    path: str | Path,
    *,
    t_start_us: int,
    t_end_us: int,
    sequence_id: str,
    width: int | None = None,
    height: int | None = None,
) -> EventBatch:
    """Read events in `[t_start_us, t_end_us]` from a supported HDF5 layout.

    When `ms_map_idx` is available, the reader uses it to restrict timestamp
    reads to the relevant millisecond range before refining bounds.
    """

    import h5py

    layout = discover_event_layout(path)
    if layout is None:
        msg = f"Could not discover event fields in {path}."
        raise ValueError(msg)

    with h5py.File(path, "r") as h5:
        if layout.kind == "compound":
            assert layout.compound and layout.x and layout.y and layout.t and layout.p
            dataset = h5[layout.compound]
            rough_start = 0
            rough_end = int(dataset.shape[0])
            timestamps = dataset[layout.t][rough_start:rough_end]
            start, end = _refine_bounds(timestamps, rough_start, t_start_us, t_end_us)
            rows = dataset[start:end]
            x = rows[layout.x].astype(np.int32)
            y = rows[layout.y].astype(np.int32)
            t = rows[layout.t].astype(np.int64)
            p = normalize_polarity(rows[layout.p])
        else:
            assert layout.x and layout.y and layout.t and layout.p
            event_count = int(h5[layout.t].shape[0])
            if layout.ms_map_idx and layout.ms_map_idx in h5:
                rough_start, rough_end = _slice_bounds_from_ms_map(
                    h5[layout.ms_map_idx],
                    event_count=event_count,
                    t_start_us=t_start_us,
                    t_end_us=t_end_us,
                )
            else:
                rough_start, rough_end = 0, event_count
            timestamps = h5[layout.t][rough_start:rough_end]
            start, end = _refine_bounds(timestamps, rough_start, t_start_us, t_end_us)
            x = h5[layout.x][start:end].astype(np.int32)
            y = h5[layout.y][start:end].astype(np.int32)
            t = h5[layout.t][start:end].astype(np.int64)
            p = normalize_polarity(h5[layout.p][start:end])

    batch = EventBatch(
        x=x,
        y=y,
        t_us=t,
        polarity=p,
        width=int(width or layout.width or 346),
        height=int(height or layout.height or 260),
        sequence_id=sequence_id,
        t_start_us=t_start_us,
        t_end_us=t_end_us,
    )
    validate_event_batch(batch, allow_empty=True)
    return batch


def count_events_window(
    path: str | Path,
    *,
    t_start_us: int,
    t_end_us: int,
) -> int:
    """Count events in ``[t_start_us, t_end_us]`` without reading all event fields."""

    import h5py

    layout = discover_event_layout(path)
    if layout is None or layout.t is None:
        msg = f"Could not discover timestamp field in {path}."
        raise ValueError(msg)

    with h5py.File(path, "r") as h5:
        event_count = int(h5[layout.t].shape[0])
        if layout.ms_map_idx and layout.ms_map_idx in h5:
            rough_start, rough_end = _slice_bounds_from_ms_map(
                h5[layout.ms_map_idx],
                event_count=event_count,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
            )
        else:
            rough_start, rough_end = 0, event_count
        timestamps = h5[layout.t][rough_start:rough_end]
    start, end = _refine_bounds(timestamps, rough_start, t_start_us, t_end_us)
    return max(0, int(end - start))


def read_navigation_window_features(
    path: str | Path,
    *,
    t_start_us: int,
    t_end_us: int,
) -> np.ndarray:
    """Read causal integrated-navigation features for a context window.

    The feature vector uses only navigation samples within
    ``[t_start_us, t_end_us]``. If the HDF5 file does not contain EvTTC
    integrated-navigation datasets, a zero vector is returned so synthetic
    fixtures and older caches remain supported.
    """

    import h5py

    features = np.zeros((len(NAVIGATION_FEATURE_NAMES),), dtype=np.float32)
    with h5py.File(path, "r") as h5:
        base = "integratedNavigation/data"
        required = [f"{base}/ts", f"{base}/velocity", f"{base}/attitude"]
        if any(name not in h5 for name in required):
            return features
        ts = h5[f"{base}/ts"]
        if len(ts) == 0:
            return features
        start = int(np.searchsorted(ts, int(t_start_us), side="left"))
        end = int(np.searchsorted(ts, int(t_end_us), side="right"))
        if end <= start:
            causal_end = int(np.searchsorted(ts, int(t_end_us), side="right"))
            if causal_end <= 0:
                return features
            start = max(0, causal_end - 1)
            end = causal_end
        times = ts[start:end].astype(np.int64)
        velocity = h5[f"{base}/velocity"][start:end].astype(np.float32)
        attitude = h5[f"{base}/attitude"][start:end].astype(np.float32)

    if velocity.size == 0:
        return features
    last_velocity = velocity[-1]
    features[0] = np.float32(np.linalg.norm(last_velocity))
    features[1:4] = last_velocity
    if velocity.shape[0] >= 2:
        duration_s = max(float(times[-1] - times[0]) / 1_000_000.0, 1e-6)
        features[4:7] = (velocity[-1] - velocity[0]) / duration_s
        features[7] = (attitude[-1, 2] - attitude[0, 2]) / duration_s
    features[8] = 1.0
    return features


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate manifest paths and target files."""

    sequences = read_manifest(path)
    report: dict[str, Any] = {
        "manifest": str(path),
        "sequence_count": len(sequences),
        "sequences": [],
    }
    for sequence in sequences:
        event_hdf5 = sequence.resolve("event_hdf5")
        ttc_csv = sequence.resolve("ttc_csv")
        gt_hdf5 = sequence.resolve("gt_hdf5")
        label_dir = sequence.resolve("label_dir")
        if event_hdf5 is None or not event_hdf5.exists():
            msg = f"Missing event_hdf5 for {sequence.sequence_id}: {event_hdf5}"
            raise FileNotFoundError(msg)
        if ttc_csv is None or not ttc_csv.exists():
            msg = f"Missing ttc_csv for {sequence.sequence_id}: {ttc_csv}"
            raise FileNotFoundError(msg)
        table = load_ttc_csv(ttc_csv)
        hdf5_datasets = describe_hdf5(event_hdf5)
        layout = discover_event_layout(event_hdf5)
        report["sequences"].append(
            {
                "sequence_id": sequence.sequence_id,
                "event_hdf5": str(event_hdf5),
                "event_size_bytes": event_hdf5.stat().st_size,
                "gt_hdf5_exists": bool(gt_hdf5 and gt_hdf5.exists()),
                "label_count": len(list(label_dir.glob("*.json")))
                if label_dir and label_dir.exists()
                else 0,
                "ttc_rows": int(table["ttc_s"].shape[0]),
                "ttc_min_s": float(np.min(table["ttc_s"])),
                "ttc_max_s": float(np.max(table["ttc_s"])),
                "hdf5_dataset_count": len(hdf5_datasets),
                "event_layout": asdict(layout) if layout else None,
            }
        )
    return report

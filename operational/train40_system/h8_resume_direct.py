"""Resume H8 generation at its durable cursor without scanning its prefix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

TOTAL_ROWS = 88_744
ASSEMBLY_GUARD_ROWS = 256


def _resume_cursor(
    output: Path,
    directory: Path,
    binding_path: Path,
    expected_binding: dict[str, Any],
    *,
    kernel: Any,  # noqa: ANN401 - injected frozen module surface
    total_rows: int = TOTAL_ROWS,
) -> int:
    progress = kernel.read(output / "H8_FEATURE_PROGRESS.json")
    cursor = progress.get("completed_rows")
    if (
        progress.get("status") not in {"RUNNING", "PAUSED_PRESERVED", "COMPLETE"}
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or not 0 <= cursor <= total_rows
        or progress.get("total_rows") != total_rows
    ):
        raise ValueError("Invalid durable H8 generation cursor")
    if not binding_path.is_file() or kernel.read(binding_path) != expected_binding:
        raise ValueError("H8 feature lineage changed")
    if cursor:
        path = directory / f"query_{cursor - 1:05d}.npz"
        receipt_path = path.with_suffix(".json")
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError("Last durable H8 fragment is missing")
        receipt = kernel.read(receipt_path)
        if (
            receipt.get("binding_sha256") != kernel.digest(binding_path)
            or receipt.get("sha256") != kernel.digest(path)
        ):
            raise ValueError("Last durable H8 fragment changed")
    return cursor


def _assemble_fragments(
    output: Path,
    directory: Path,
    histories: list[list[str | None]],
    jobs: list[dict[str, Any]],
    *,
    kernel: Any,  # noqa: ANN401 - injected frozen module surface
) -> tuple[Any, Any, Any, Any] | None:
    """Load fragments once in canonical row/slot order without hashing them."""
    np = kernel.np
    positions: dict[str, int] = {}
    values: list[Any] = []
    anchors: list[int] = []
    availability: list[int] = []
    total = len(jobs)
    if len(histories) != total:
        raise ValueError("H8 histories and jobs differ")
    for row, (keys, job) in enumerate(zip(histories, jobs, strict=True)):
        if row % ASSEMBLY_GUARD_ROWS == 0:
            allowed, resources = kernel.resource_guard(output)
            if not allowed:
                kernel.atomic_json(output / "H8_FEATURE_PAUSE.json", resources)
                kernel.atomic_json(
                    output / "H8_ASSEMBLY_PROGRESS.json",
                    {"status": "PAUSED", "completed_rows": row, "total_rows": total},
                )
                return None
            kernel.atomic_json(
                output / "H8_ASSEMBLY_PROGRESS.json",
                {"status": "RUNNING", "completed_rows": row, "total_rows": total},
            )
        path = directory / f"query_{row:05d}.npz"
        with np.load(path, allow_pickle=False) as stored:
            features = stored["features"]
            valid = stored["valid"]
        expected_valid = np.asarray(job["valid"], dtype=bool)
        if (
            features.ndim != 2
            or features.shape[0] != 8
            or features.dtype != np.float32
            or valid.shape != (8,)
            or valid.dtype != np.bool_
            or not np.array_equal(valid, expected_valid)
            or len(keys) != 8
        ):
            raise ValueError(f"H8 fragment shape or validity changed at row {row}")
        for slot, key in enumerate(keys):
            if key is None:
                continue
            if key in positions:
                if not np.array_equal(values[positions[key]], features[slot]):
                    raise ValueError("Identical H8 observations have differing expert features")
            else:
                positions[key] = len(values)
                values.append(features[slot].copy())
                anchors.append(int(job["anchor"]) - (7 - slot) * 50_000)
                availability.append(int(job["available"]))
        if (row + 1) % ASSEMBLY_GUARD_ROWS == 0 or row + 1 == total:
            kernel.atomic_json(
                output / "H8_ASSEMBLY_PROGRESS.json",
                {"status": "RUNNING", "completed_rows": row + 1, "total_rows": total},
            )
    history = np.asarray(
        [[positions[key] if key is not None else -1 for key in keys] for keys in histories],
        np.int64,
    )
    return (
        np.asarray(values, np.float32),
        history,
        np.asarray(anchors, np.int64),
        np.asarray(availability, np.int64),
    )


def history_cache(output: Path, raw_root: Path) -> None:
    """Generate only the suffix, then assemble every fragment in canonical order."""
    import hashlib
    import json

    import h5py
    import pyarrow.dataset as ds

    from e_jepa_ttc.data.eap import _require_hdf5plugin
    from e_jepa_ttc.simplex_t.cache import fit_normalizer, training_mass
    from e_jepa_ttc.simplex_t.context_dedup import ContextSource, observation_key
    from e_jepa_ttc.simplex_t.query_context import QueryContextInput
    from operational.simplex_t_shared_route.adapter import extract
    from operational.train40_system import history_resources8 as kernel

    np, torch = kernel.np, kernel.torch
    models, parents = kernel.load_producers(output, pair=True)
    _require_hdf5plugin()
    audit = kernel.read(output / "DATA_AUDIT.json")
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        names = (
            "tokens",
            "sequences",
            "windows_us",
            "square_xyxy",
            "delta_t_s",
            "rgb_members",
            "phase",
            "ttc_s",
        )
        index = {key: stored[key] for key in names}
    if len(index["tokens"]) != TOTAL_ROWS:
        raise ValueError("Canonical H8 row count changed")
    media_path = raw_root / "data/train.parquet"
    media = (
        ds.dataset(media_path)
        .to_table(
            columns=[
                "sequence_id",
                "rgb_member_path",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
            filter=ds.field("sequence_id").isin(audit["sequences"]),
        )
        .to_pylist()
    )
    frames = {(row["sequence_id"], row["rgb_member_path"]): row for row in media}
    if len(frames) != len(media):
        raise ValueError("Ambiguous TRAIN exposure reference")
    raw_pins = {
        item["sequence_id"]: item for item in kernel.read(output / "RAW_PLAN.json")["files"]
    }
    bounds = {}
    for sequence in audit["sequences"]:
        with h5py.File(raw_root / "data/train" / sequence / "events.h5", "r") as handle:
            timestamps = handle["events/t"]
            if not isinstance(timestamps, h5py.Dataset):
                raise ValueError("Raw timestamps must be a dataset")
            bounds[sequence] = (int(timestamps[0]), int(timestamps[-1]))
    family_sha = hashlib.sha256(json.dumps(parents, sort_keys=True).encode()).hexdigest()
    preprocessing_sha = kernel.digest(output / "PREPARE_FREEZE.json")
    directory = output / "h8_feature_fragments"
    directory.mkdir(exist_ok=True)
    binding = {
        "parents": parents,
        "index_sha256": audit["index_sha256"],
        "exposure_metadata_sha256": kernel.digest(media_path),
        "source_sha256": kernel.digest(Path(kernel.__file__)),
        "lags_us": list(range(350000, -1, -50000)),
        "precision": "float32",
    }
    binding_path = directory / "BINDING.json"
    cursor = _resume_cursor(output, directory, binding_path, binding, kernel=kernel)

    jobs: list[dict[str, Any]] = []
    histories: list[list[str | None]] = []
    for row in range(TOTAL_ROWS):
        sequence = str(index["sequences"][row])
        windows = index["windows_us"][row]
        start_bound, end_bound = bounds[sequence]
        endpoints = [frames[(sequence, str(member))] for member in index["rgb_members"][row]]
        if [int(frame["rgb_exposure_start_timestamp_us"]) for frame in endpoints] != windows[
            1:, 1
        ].tolist():
            raise ValueError("TRAIN exposure and event clock mapping differs")
        available = max(int(frame["rgb_exposure_end_timestamp_us"]) for frame in endpoints)
        if available < windows[-1, 1] or windows.max() > end_bound:
            raise ValueError("Current exposure or complete raw support missing")
        current = QueryContextInput(
            query_token=f"{row}:{index['tokens'][row]}",
            producer_family_sha256=family_sha,
            anchor_us=int(windows[-1, 1]),
            roi_available_us=available,
            allowed_cutoff_us=available,
            source_start_us=start_bound,
            windows_us=(
                (int(windows[0, 0]), int(windows[0, 1])),
                (int(windows[1, 0]), int(windows[1, 1])),
                (int(windows[2, 0]), int(windows[2, 1])),
            ),
            square_xyxy=(
                float(index["square_xyxy"][row, 0]),
                float(index["square_xyxy"][row, 1]),
                float(index["square_xyxy"][row, 2]),
                float(index["square_xyxy"][row, 3]),
            ),
        )
        source = ContextSource(sequence, raw_pins[sequence]["sha256"], preprocessing_sha, current)
        valid = windows.min() - np.arange(7, -1, -1) * 50_000 >= start_bound
        if not valid[-1]:
            raise ValueError("Current TRAIN context lacks causal raw support")
        histories.append(
            [
                observation_key(source, int(lag)) if present else None
                for lag, present in zip(range(350000, -1, -50000), valid, strict=True)
            ]
        )
        jobs.append(
            {
                "path": str(raw_root / "data/train" / sequence / "events.h5"),
                "sequence": sequence,
                "windows": windows,
                "square": index["square_xyxy"][row],
                "valid": valid,
                "delta": float(index["delta_t_s"][row]),
                "available": available,
                "anchor": int(windows[-1, 1]),
            }
        )

    kernel.atomic_json(
        output / "H8_FEATURE_PROGRESS.json",
        {"status": "RUNNING", "completed_rows": cursor, "total_rows": TOTAL_ROWS},
    )
    with kernel.ProcessPoolExecutor(max_workers=8, initializer=kernel.worker_init) as pool:
        pending: dict[int, Any] = {}
        with torch.inference_mode():
            for row in range(cursor, TOTAL_ROWS):
                job = jobs[row]
                allowed, resources = kernel.resource_guard(output)
                if not allowed:
                    kernel.atomic_json(output / "H8_FEATURE_PAUSE.json", resources)
                    return
                path = directory / f"query_{row:05d}.npz"
                receipt_path = path.with_suffix(".json")
                if receipt_path.exists():
                    receipt = kernel.read(receipt_path)
                    if (
                        receipt["binding_sha256"] != kernel.digest(binding_path)
                        or kernel.digest(path) != receipt["sha256"]
                    ):
                        raise ValueError("H8 feature fragment changed")
                else:
                    for next_row in range(row, min(row + 8, TOTAL_ROWS)):
                        if next_row not in pending and not (
                            directory / f"query_{next_row:05d}.json"
                        ).exists():
                            pending[next_row] = pool.submit(kernel.prepare, jobs[next_row])
                    started = kernel.time.perf_counter()
                    tensor = torch.from_numpy(pending.pop(row).result()).to("cuda")
                    delta = torch.full((16, 2), job["delta"], dtype=torch.float32, device="cuda")
                    features = extract("H8_SEED7", models, tensor, delta)[-8:]
                    features[~job["valid"]] = 0
                    temporary = path.with_suffix(".pending.npz")
                    np.savez_compressed(temporary, features=features, valid=job["valid"])
                    kernel.os.replace(temporary, path)
                    kernel.atomic_json(
                        receipt_path,
                        {
                            "sha256": kernel.digest(path),
                            "binding_sha256": kernel.digest(binding_path),
                            "seconds": kernel.time.perf_counter() - started,
                            "optimizer_updates": 0,
                        },
                    )
                kernel.atomic_json(
                    output / "H8_FEATURE_PROGRESS.json",
                    {"status": "RUNNING", "completed_rows": row + 1, "total_rows": TOTAL_ROWS},
                )

    assembled = _assemble_fragments(output, directory, histories, jobs, kernel=kernel)
    if assembled is None:
        return
    features, history, anchors, availability = assembled
    normalizer = fit_normalizer(features, history, np.ones(len(features), bool))
    mass = training_mass(index["ttc_s"], index["sequences"])
    path = output / "H8_FEATURES.npz"
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(
        temporary,
        features=features,
        history=history,
        anchor_us=anchors,
        available_us=availability,
        mean=normalizer.mean,
        scale=normalizer.scale,
        mass=mass,
        target_phase=index["phase"].astype(np.float32),
    )
    kernel.os.replace(temporary, path)
    kernel.atomic_json(
        output / "H8_FEATURE_MANIFEST.json",
        {
            "status": "COMPLETE_VERIFIED",
            "row_count": TOTAL_ROWS,
            "parents": parents,
            "files": [{"path": path.name, "sha256": kernel.digest(path)}],
            "binding_sha256": kernel.digest(binding_path),
            "consumed_ids_sha256": normalizer.consumed_ids_sha256,
            "unique_observations": len(features),
            "optimizer_updates": 0,
            "resume_direct": {
                "generation_cursor": cursor,
                "historical_prefix_fragment_hashes_waived": True,
                "last_prefix_fragment_verified": cursor > 0,
                "assembly_loaded_all_rows_in_chronological_order": True,
            },
        },
    )
    kernel.atomic_json(
        output / "H8_ASSEMBLY_PROGRESS.json",
        {"status": "COMPLETE", "completed_rows": TOTAL_ROWS, "total_rows": TOTAL_ROWS},
    )


__all__ = ["history_cache"]

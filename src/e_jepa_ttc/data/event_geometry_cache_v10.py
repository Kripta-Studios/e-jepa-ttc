"""Bounded, integer-clock timestamp surfaces and fixed normal-flow routing features."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, digest, verify
from e_jepa_ttc.data.raw_event_binding import select_hash_probe_tokens
from e_jepa_ttc.models.normal_flow_routing_v10 import fit_affine_normal_flow, fit_time_plane


def geometry_window(
    chunks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], row: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Latest timestamps at original sensor pixel resolution, separate polarity."""
    start, end = int(row["window_start_us"]), int(row["window_end_us"])
    x0, y0, x1, y1 = [float(row[k]) for k in ("roi_x0", "roi_y0", "roi_x1", "roi_y1")]
    off = float(row["event_x_offset_px"])
    size = x1 - x0
    if end <= start or size <= 0 or not np.isclose(size, y1 - y0, atol=1e-6):
        raise ValueError("invalid common ROI/time")
    # A sparse sensor-resolution crop is equivalent to a full invalid surface outside ROI.
    left = max(0, int(np.ceil(x0 - off)))
    top = max(0, int(np.ceil(y0)))
    right = max(left + 1, int(np.ceil(x1 - off)))
    bottom = max(top + 1, int(np.ceil(y1)))
    width, height = right - left, bottom - top
    if width * height > 16000000:
        raise RuntimeError("sensor ROI exceeds bounded surface allocation")
    surface = np.full((2, height, width), np.iinfo(np.int64).min, dtype=np.int64)
    centers = []
    for iy in range(16):
        for ix in range(16):
            cx = int(np.floor(x0 + (ix + 0.5) * size / 16 - off))
            cy = int(np.floor(y0 + (iy + 0.5) * size / 16))
            centers.append((cx, cy))
    centers = sorted(set(centers))
    snapshots = [start + (end - start) * k // 4 for k in range(1, 5)]
    results = []
    position = 0
    previous = start

    def observe(snapshot: int) -> None:
        locations = []
        normals = []
        speeds = []
        weights = []
        plane_records = []
        for pol in range(2):
            for cx, cy in centers:
                xs = np.arange(max(left, cx - 4), min(right, cx + 5))
                ys = np.arange(max(top, cy - 4), min(bottom, cy + 5))
                xx, yy = np.meshgrid(xs, ys)
                times = surface[pol, yy - top, xx - left].ravel()
                good = (times >= max(start, snapshot - 10000)) & (times < snapshot)
                if good.sum() < 9:
                    continue
                xy = np.column_stack(
                    (
                        (xx.ravel()[good] + off - (x0 + x1) / 2) / size,
                        (yy.ravel()[good] - (y0 + y1) / 2) / size,
                    )
                )
                relative = (times[good] - np.int64(snapshot)).astype(np.float64) * 1e-6
                plane = fit_time_plane(xy, relative)
                if not plane.valid:
                    continue
                locations.append([(cx + off - (x0 + x1) / 2) / size, (cy - (y0 + y1) / 2) / size])
                normals.append(plane.normal)
                speeds.append(plane.speed)
                weights.append(max(plane.r_squared, 0) * max(0, 1 - plane.residual_fraction / 0.2))
                plane_records.append(
                    [
                        *locations[-1],
                        *plane.normal.tolist(),
                        plane.speed,
                        pol,
                        plane.r_squared,
                        plane.residual_fraction,
                        float(np.ptp(relative)),
                    ]
                )
        fit = fit_affine_normal_flow(
            np.asarray(locations).reshape(-1, 2),
            np.asarray(normals).reshape(-1, 2),
            np.asarray(speeds),
            np.asarray(weights),
        )
        features = np.zeros(12, dtype=np.float64)
        features[6] = len(locations) / (2 * len(centers))
        features[8] = 6
        features[9] = 10
        raw = np.zeros(6)
        if fit.valid:
            a, b, c, d, tx, ty = fit.beta
            raw = np.array([(a + d) / 2, (c - b) / 2, (a - d) / 2, (b + c) / 2, tx, ty])
            features[:6] = np.clip(np.arcsinh(raw), -20, 20)
            features[7:] = [
                fit.inlier_fraction,
                np.clip(np.log10(max(fit.condition, 1)), 0, 6),
                np.clip(fit.residual, 0, 10),
                float(fit.mode == "affine"),
                1,
            ]
        results.append(
            (
                features,
                dict(
                    snapshot_us=snapshot,
                    mode=fit.mode,
                    valid=bool(fit.valid),
                    raw=raw.tolist(),
                    plane_count=len(locations),
                    candidate_slots=2 * len(centers),
                    orientation_balance=fit.orientation_balance,
                    planes=plane_records,
                ),
            )
        )

    for xs, ys, ts, ps in chunks:
        ts = np.asarray(ts, dtype=np.int64)
        if len(ts) and (
            ts[0] < previous or ts[0] < start or ts[-1] >= end or np.any(np.diff(ts) < 0)
        ):
            raise ValueError("raw timestamp/order/half-open violation")
        if len(ts):
            previous = int(ts[-1])
        if not np.isin(ps, ([0, 1] if row["polarity_encoding"] == "zero_one" else [-1, 1])).all():
            raise ValueError("polarity encoding mismatch")
        keep = (xs + off >= x0) & (xs + off < x1) & (ys >= y0) & (ys < y1)
        x = np.asarray(xs[keep], dtype=np.int64) - left
        y = np.asarray(ys[keep], dtype=np.int64) - top
        t = ts[keep]
        p = (ps[keep] > 0).astype(np.int64)
        begin = 0
        while position < 4:
            cut = int(np.searchsorted(t, snapshots[position], side="left"))
            np.maximum.at(surface, (p[begin:cut], y[begin:cut], x[begin:cut]), t[begin:cut])
            begin = cut
            if len(ts) and ts[-1] >= snapshots[position]:
                observe(snapshots[position])
                position += 1
            else:
                break
        if begin < len(t):
            np.maximum.at(surface, (p[begin:], y[begin:], x[begin:]), t[begin:])
    while position < 4:
        observe(snapshots[position])
        position += 1
    chosen = next((v for v in reversed(results) if v[1]["valid"]), results[-1])
    chosen[1]["snapshots"] = [
        {k: v for k, v in item[1].items() if k != "planes"} for item in results
    ]
    return chosen


def combine_windows(
    first: tuple[np.ndarray, dict[str, Any]], second: tuple[np.ndarray, dict[str, Any]]
) -> tuple[np.ndarray, str]:
    a, b = first[1], second[1]
    cross = np.zeros(4)
    reason = "invalid_window"
    if a["valid"] and b["valid"]:
        x, y = np.asarray(a["raw"]), np.asarray(b["raw"])
        dt = (int(b["snapshot_us"]) - int(a["snapshot_us"])) * 1e-6
        if dt <= 0:
            raise ValueError("window snapshot ordering invalid")
        denominator = 1 - x[0] * dt
        reason = "propagation_singularity"
        if denominator > 1e-6:
            cross = np.array(
                [
                    y[0] - x[0],
                    y[1] - x[1],
                    np.linalg.norm(y[4:] - x[4:]),
                    abs(y[0] - x[0] / denominator),
                ]
            )
            cross = np.clip(np.arcsinh(cross), -20, 20)
            reason = "valid"
    return np.concatenate((first[0], second[0], cross)), reason


def build_geometry(
    binding_csv: Path, raw_root: Path, output: Path, check: Callable[[], None]
) -> dict[str, Any]:
    """Extract only the fixed nine sources/windows after the Stage68 branch gate."""
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(binding_csv)
    tokens = sorted(rows.sample_token.unique())
    if (
        len(rows) != 16384
        or len(tokens) != 8192
        or rows.sequence_id.nunique() != 9
        or rows.duplicated(["sample_token", "window_id"]).any()
    ):
        raise ValueError("geometry binding universe differs")
    identities = []
    for sequence, group in rows.groupby("sequence_id"):
        check()
        p = (raw_root / str(sequence) / "events.h5").resolve(strict=True)
        if not p.is_relative_to(raw_root.resolve(strict=True)):
            raise PermissionError("raw source escaped root")
        expected = group.h5_file_sha256.unique()
        if len(expected) != 1 or digest(p) != expected[0]:
            raise ValueError("raw source identity mismatch")
        identities.append(
            dict(
                path=str(p),
                bytes=p.stat().st_size,
                sha256=expected[0],
                mtime_ns=p.stat().st_mtime_ns,
            )
        )
    atomic_json(output / "SOURCE_HASHES.json", identities)
    probe = select_hash_probe_tokens(rows, 64)
    values: dict[str, list[float]] = {}
    diagnostics: dict[str, Any] = {}

    def extract(token: str) -> None:
        check()
        pair = rows.loc[rows.sample_token == token].sort_values("window_id")
        windows = []
        if pair.roi_transform_sha256.nunique() != 1:
            raise ValueError("per-window crop geometry forbidden")
        for row in pair.to_dict("records"):
            p = (raw_root / row["sequence_id"] / "events.h5").resolve(strict=True)
            stamp = p.stat()
            expected = next(r for r in identities if r["path"] == str(p))
            if stamp.st_size != expected["bytes"] or stamp.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError("source changed during extraction")

            def chunks(
                p: Path = p, row: dict = row, token: str = token
            ) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
                with h5py.File(p, "r") as h:
                    for k in ("events/x", "events/y", "events/t", "events/p"):
                        if isinstance(h.get(k, getlink=True), h5py.ExternalLink):
                            raise PermissionError("external HDF5 link")
                    lo, hi = int(row["start_event_index"]), int(row["stop_event_index"])
                    t = h["events/t"]
                    if not isinstance(t, h5py.Dataset):
                        raise ValueError("timestamp source must be a dataset")
                    if (lo and int(t[lo - 1]) >= int(row["window_start_us"])) or (
                        hi < len(t) and int(t[hi]) < int(row["window_end_us"])
                    ):
                        raise ValueError("binding indices are not exact boundaries")
                    for begin in range(lo, hi, 2000000):
                        check()
                        end = min(begin + 2000000, hi)
                        with (output / "RAW_ACCESS.jsonl").open("a") as log:
                            log.write(
                                json.dumps(
                                    dict(
                                        path=str(p),
                                        token=token,
                                        window=row["window_id"],
                                        start=begin,
                                        stop=end,
                                    )
                                )
                                + "\n"
                            )
                        arrays = []
                        for k in "xytp":
                            dataset = h["events/" + k]
                            if not isinstance(dataset, h5py.Dataset):
                                raise ValueError("event source must be a dataset")
                            arrays.append(np.asarray(dataset[begin:end]))
                        yield cast(
                            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple(arrays)
                        )

            windows.append(geometry_window(chunks(), row))
        vector, reason = combine_windows(windows[0], windows[1])
        values[token] = vector.tolist()
        diagnostics[token] = dict(windows=[w[1] for w in windows], cross_reason=reason)

    progress = output / "PROGRESS.json"
    completed = {}
    if progress.exists():
        completed = json.loads(progress.read_text())
        for token, record in completed.items():
            data = json.loads(verify(record).read_text())
            values[token] = data["features"]
            diagnostics[token] = data["diagnostics"]
    (output / "tokens").mkdir(exist_ok=True)
    for token in list(probe) + [t for t in tokens if t not in probe]:
        if token not in values:
            extract(token)
            token_path = output / "tokens" / f"{tokens.index(token):05d}.json"
            atomic_json(token_path, dict(features=values[token], diagnostics=diagnostics[token]))
            completed[token] = binding(token_path)
            atomic_json(progress, completed)
        if set(probe) <= values.keys() and not (output / "PHYSICAL_PROBE.json").exists():
            atomic_json(output / "PHYSICAL_PROBE.json", {t: diagnostics[t] for t in probe})
    matrix = np.asarray([values[t] for t in tokens], dtype=np.float64)
    np.save(output / "geometry28.npy", matrix, allow_pickle=False)
    atomic_json(output / "tokens.json", tokens)
    atomic_json(output / "GEOMETRY_DIAGNOSTICS.json", diagnostics)
    return dict(
        array=binding(output / "geometry28.npy"),
        tokens=binding(output / "tokens.json"),
        diagnostics=binding(output / "GEOMETRY_DIAGNOSTICS.json"),
        sources=identities,
    )

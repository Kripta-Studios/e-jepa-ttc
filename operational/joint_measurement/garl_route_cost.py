"""Measure pinned upstream and local Garl full pipelines on identical P queries.

Registration reads sensor metadata only. Measurement is CPU FP32, batch one,
after training/evaluation writers stop. Both routes use the complete published
model: the difference is input preparation, not a reduced comparator or a new
model. The upstream representation already uses vectorized time volumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from operational.evttc_rgb_transfer.model import FullGarl
from operational.range_transfer.common import csv_write, freeze, read, sha, utc, write
from operational.sota_eval import garl_parity

from .common import ROOT, RUN, gpu_ownership, owner, require_resources

OUT = RUN / "cost/garl_routes"
COLUMNS = (
    "sequence_id",
    "sample_token",
    "public_track_id",
    "frame_timestamps_us",
    "rgb_shard_paths",
    "rgb_member_paths",
    "events_path",
    "event_windows_us",
    "boxes_xyxy",
)
ROUTES = ("UPSTREAM", "LOCAL")


def select_rows(frame: pd.DataFrame, count: int) -> list[dict]:
    """Round-robin sorted P groups without labels or availability filtering."""
    if count <= 0 or count > len(frame) or frame.sample_token.isna().any():
        raise ValueError("Invalid requested population")
    if frame.sample_token.duplicated().any() or frame.group_id.isna().any():
        raise ValueError("Duplicate query or missing group")
    groups = [
        g.sort_values("sample_token").to_dict("records")
        for _, g in frame.groupby("group_id", sort=True)
    ]
    rows = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)][
        :count
    ]
    # Parquet nested arrays must become plain, lossless JSON metadata.
    return json.loads(json.dumps(rows, default=lambda x: x.tolist()))


def summarize(values: list[float]) -> dict:
    """Summarize measured durations, rejecting missing/invalid observations."""
    array = np.asarray(values, np.float64)
    if not len(array) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("Missing, invalid or nonpositive cost")
    return {
        "n": len(array),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.5)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "queries_per_second": float(1000 / array.mean()),
    }


def route_order(index: int, repeat: int) -> tuple[str, str]:
    """Reverse paired order between passes to balance cache/order effects."""
    return ROUTES if (index + repeat) % 2 == 0 else (ROUTES[1], ROUTES[0])


def ensure_pair(reference: np.ndarray, local: np.ndarray) -> None:
    """Require exact native sensor values; never accept a favorable subset."""
    if any(x.shape != (46, 128, 128) or x.dtype != np.float32 for x in (reference, local)):
        raise ValueError("Garl input shape or precision changed")
    if not np.isfinite(reference).all() or not np.array_equal(reference, local):
        raise ValueError("Native/local sensor parity failed")


def register(campaign: Path, eap_root: Path, code_root: Path, full_public: Path) -> dict:
    """Freeze 64 P queries, routes and dependencies before any measurement."""
    manifest_path = owner() / "P_MANIFEST.json"
    manifest = read(manifest_path)
    rows_path = Path(manifest["rows_path"])
    source = campaign / "TRAIN40_ROWS.parquet"
    if sha(rows_path) != manifest["rows_sha256"] or sha(source) != manifest["source_sha256"]:
        raise ValueError("P/native rows identity changed")
    keys = pd.read_parquet(rows_path, columns=["sample_token", "group_id"])
    native = pd.read_parquet(source, columns=list(COLUMNS))
    if keys.sample_token.duplicated().any() or native.sample_token.duplicated().any():
        raise ValueError("Duplicate native/P query")
    frame = keys.merge(native, on="sample_token", how="left", validate="one_to_one", indicator=True)
    if len(frame) != manifest["population_size"] or (frame["_merge"] != "both").any():
        raise ValueError("P/native populations differ")
    selected = select_rows(frame.drop(columns="_merge"), 64)
    for row in selected:
        for relative in [row["events_path"], *row["rgb_shard_paths"]]:
            if not (eap_root / relative).is_file():
                raise FileNotFoundError(f"Registered input missing; no replacement: {relative}")
    paths = [
        Path(__file__),
        Path(garl_parity.__file__),
        ROOT / "operational/evttc_rgb_transfer/model.py",
        ROOT / "operational/range_transfer/common.py",
        Path(__file__).with_name("common.py"),
    ]
    dependencies = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    dependencies.update(garl_parity._local_source_hashes())
    result = {
        "schema": "v18_garl_paired_route_cost_v1",
        "queries": selected,
        "campaign": str(campaign),
        "eap_root": str(eap_root),
        "code_root": str(code_root),
        "full_public": str(full_public),
        "P_manifest_sha256": sha(manifest_path),
        "native_rows_sha256": sha(source),
        "native_sources": garl_parity._source_hashes(code_root, campaign),
        "dependencies": dependencies,
        "device": "cpu",
        "precision": "FP32",
        "batch_size": 1,
        "torch_threads": 1,
        "passes": 2,
        "prepared_warmups": 20,
        "prepared_samples": 64,
        "routes": {
            "UPSTREAM": "garl_parity._reference_sensor -> full native model",
            "LOCAL": "garl_parity._local_sensor -> same full native model",
        },
        "optimization_scope": "Existing input adapter only; both timevolumes already vectorized",
        "selection": (
            "64 P queries, sorted group/token round robin; no labels or availability filtering"
        ),
        "parity": "Exact float32 sensors and heights on every paired query/pass",
        "cold_scope": "Model load and first prepared forward; OS disk cache uncontrolled",
        "warm_scope": "Resident model, per-query reader pools reopened, no sensor/prediction cache",
        "cost_scope": (
            "Read/decode/normalize/ROI/timevolume + native full forward + signed emission"
        ),
        "memory_scope": (
            "Process RSS after each phase and lifetime peak working set; not isolated route peaks"
        ),
        "optimizer_updates": 0,
        "labels_read": False,
        "model_selection": False,
        "requires_no_other_project_compute": True,
    }
    freeze(OUT / "REGISTRATION.json", result)
    return result


def guard_idle() -> None:
    """Refuse concurrent project compute, permitting read-only hourly observers."""
    own = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in own or "python" not in (process.info["name"] or "").lower():
                continue
            command = " ".join(process.info["cmdline"] or [])
            if "hourly" in command.lower():
                continue
            if Path(process.cwd()).is_relative_to(ROOT.parent):
                raise RuntimeError(f"Project compute still live: PID {process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    require_resources(starting=True)


def _forward(model: FullGarl, sensor: np.ndarray) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(sensor[None])
    started = time.perf_counter()
    with torch.inference_mode():
        heights, _ = model.model.forward_test(tensor)
    output = heights.float().numpy()
    from operational.evttc_rgb_transfer.model import release_ttc

    ttc = release_ttc(output, 0.1)
    elapsed = (time.perf_counter() - started) * 1000
    if output.shape != (1, 2) or not np.isfinite(output).all():
        raise ValueError("Invalid native heights")
    # Preserve nonfinite TTC explicitly, never omit a hard query from latency.
    return np.concatenate((output[0], ttc)), elapsed


def _json_numbers(values: np.ndarray) -> list:
    return [float(x) if np.isfinite(x) else str(float(x)) for x in values]


def measure() -> dict:
    """Replay all registered pairs, preserving failed comparisons and resumable records."""
    guard_idle()
    # Serialize with existing GPU writers even though this CPU-only measurement
    # performs no CUDA work and consumes no GPU scope allocation.
    with gpu_ownership():
        return _measure_idle()


def _measure_idle() -> dict:
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError("At least 6 GiB available RAM required before full Garl load")
    registration = read(OUT / "REGISTRATION.json")
    binding = sha(OUT / "REGISTRATION.json")
    for name, expected in registration["dependencies"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Registered local source changed: {name}")
    code_root = Path(registration["code_root"])
    for name, expected in registration["native_sources"].items():
        if sha(code_root / name) != expected:
            raise ValueError(f"Registered native source changed: {name}")
    sys.path.insert(0, str(code_root))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    session = OUT / "sessions" / str(time.time_ns())
    write(
        session / "START.json",
        {
            "utc": utc(),
            "pid": os.getpid(),
            "registration_sha256": binding,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "threads": torch.get_num_threads(),
            "cpu_count": psutil.cpu_count(),
            "available_ram": psutil.virtual_memory().available,
            "environment_pythonpath": os.environ.get("PYTHONPATH", ""),
        },
    )
    started = time.perf_counter()
    model = FullGarl(Path(registration["full_public"]), code_root, "cpu")
    load_ms = (time.perf_counter() - started) * 1000
    eap_root = Path(registration["eap_root"])
    first, _, _ = garl_parity._local_sensor(registration["queries"][0], eap_root)
    _, first_ms = _forward(model, first)
    for _ in range(registration["prepared_warmups"]):
        _forward(model, first)
    prepared = [_forward(model, first)[1] for _ in range(registration["prepared_samples"])]
    write(
        session / "PREPARED.json",
        {
            "model_binding": model.bindings,
            "load_ms": load_ms,
            "first_forward_ms": first_ms,
            "warm_ms": prepared,
            "warm_summary": summarize(prepared),
            "input_scope": (
                "One fixed resident tensor, native forward and TTC emission; no raw preprocessing"
            ),
            "cold_disk_cache_measured": False,
        },
    )
    del first
    for repeat in range(registration["passes"]):
        for index, row in enumerate(registration["queries"]):
            output = OUT / "pairs" / f"{repeat}_{index:03d}.json"
            if output.exists():
                old = read(output)
                if old["registration_sha256"] != binding or old["status"] != "PASS":
                    raise ValueError(f"Unadmitted prior pair: {output}")
                continue
            guard_idle()
            record: dict[str, Any] = {
                "registration_sha256": binding,
                "sample_token": row["sample_token"],
                "sequence_id": row["sequence_id"],
                "group_id": row["group_id"],
                "repeat": repeat,
                "index": index,
                "session": session.name,
                "order": list(route_order(index, repeat)),
                "routes": {},
            }
            sensors, predictions = {}, {}
            for route in record["order"]:
                started = time.perf_counter()
                try:
                    if route == "UPSTREAM":
                        sensor, meta = garl_parity._reference_sensor(row, eap_root)
                    else:
                        sensor, _, meta = garl_parity._local_sensor(row, eap_root)
                    prepared_at = time.perf_counter()
                    prediction, forward_ms = _forward(model, sensor)
                except Exception as error:
                    record.update(status="FAIL", error=str(error), failed_route=route)
                    write(output, record)
                    raise
                ended = time.perf_counter()
                memory = psutil.Process().memory_info()
                record["routes"][route] = {
                    "preprocess_ms": (prepared_at - started) * 1000,
                    "model_emission_ms": forward_ms,
                    "total_ms": (ended - started) * 1000,
                    "sensor_sha256": hashlib.sha256(sensor.tobytes()).hexdigest(),
                    "heights_ttc": _json_numbers(prediction),
                    "metadata": meta,
                    "rss_bytes": memory.rss,
                    "lifetime_peak_rss_bytes": getattr(memory, "peak_wset", None),
                }
                sensors[route], predictions[route] = sensor, prediction
            try:
                ensure_pair(sensors["UPSTREAM"], sensors["LOCAL"])
                if not np.array_equal(
                    predictions["UPSTREAM"], predictions["LOCAL"], equal_nan=True
                ):
                    raise ValueError("Paired Garl predictions differ")
            except ValueError as error:
                record.update(status="FAIL", error=str(error))
                write(output, record)
                raise
            record["status"] = "PASS"
            write(output, record)
            del sensors, predictions, sensor
    return report()


def report() -> dict:
    """Recompute complete paired costs, refusing missing/duplicate/unbound observations."""
    registration = read(OUT / "REGISTRATION.json")
    binding = sha(OUT / "REGISTRATION.json")
    records = []
    expected_files = {
        f"{repeat}_{index:03d}.json"
        for repeat in range(registration["passes"])
        for index in range(len(registration["queries"]))
    }
    if {p.name for p in (OUT / "pairs").glob("*.json")} != expected_files:
        raise ValueError("Missing or extra paired cost records")
    for repeat in range(registration["passes"]):
        for index, query in enumerate(registration["queries"]):
            row = read(OUT / "pairs" / f"{repeat}_{index:03d}.json")
            if (
                row["status"] != "PASS"
                or row["registration_sha256"] != binding
                or row["sample_token"] != query["sample_token"]
                or row["repeat"] != repeat
                or row["index"] != index
                or row["order"] != list(route_order(index, repeat))
            ):
                raise ValueError("Incomplete or incompatible paired cost evidence")
            left, right = (row["routes"][route] for route in ROUTES)
            if (
                left["sensor_sha256"] != right["sensor_sha256"]
                or left["heights_ttc"] != right["heights_ttc"]
            ):
                raise ValueError("Reported parity contradicts paired payloads")
            for route in ROUTES:
                values = row["routes"][route]
                records.append(
                    {
                        "sample_token": row["sample_token"],
                        "group_id": row["group_id"],
                        "repeat": repeat,
                        "route": route,
                        "session": row["session"],
                        **{
                            k: values[k]
                            for k in (
                                "preprocess_ms",
                                "model_emission_ms",
                                "total_ms",
                                "rss_bytes",
                                "lifetime_peak_rss_bytes",
                            )
                        },
                    }
                )
    result = {
        "status": "COMPLETE",
        "registration_sha256": binding,
        "unique_queries": len(registration["queries"]),
        "route_observations": len(records),
        "scope": registration["cost_scope"],
        "device": "cpu",
        "labels_read": False,
        "pair_file_sha256": {name: sha(OUT / "pairs" / name) for name in sorted(expected_files)},
        "summary": {
            route: {
                field: summarize([r[field] for r in records if r["route"] == route])
                for field in ("preprocess_ms", "model_emission_ms", "total_ms")
            }
            for route in ROUTES
        },
        "limitations": [
            "P timing sample, no accuracy ranking",
            "OS disk cache uncontrolled",
            "First-forward/process startup separated; not disk-cold latency",
            "CPU only; not GPU comparison or V18 fusion cost admission",
            "Both native and local timevolumes already vectorized",
            "RSS snapshots and process lifetime peak are not route-isolated peaks",
        ],
    }
    csv_write(OUT / "MEASUREMENTS.csv", records)
    write(OUT / "RESULT.json", result)
    return result


def main() -> None:
    """Register without model execution, then measure after compute owners finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "measure", "report"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--full-public", type=Path)
    args = parser.parse_args()
    if args.mode == "register":
        if any(
            getattr(args, name) is None
            for name in ("campaign", "eap_root", "code_root", "full_public")
        ):
            parser.error("Registration requires all four explicit input roots")
        result = register(
            *[
                getattr(args, name).resolve()
                for name in ("campaign", "eap_root", "code_root", "full_public")
            ]
        )
        print(json.dumps({"registered_queries": len(result["queries"])}))
    else:
        print(json.dumps(measure() if args.mode == "measure" else report(), indent=2))


if __name__ == "__main__":
    main()

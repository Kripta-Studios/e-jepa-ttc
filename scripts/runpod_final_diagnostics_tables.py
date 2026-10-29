"""Offline tables from preserved diagnostics; never import a model or execute inference."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_PHYSICAL = 5805
EXPECTED_TOKENS = 6762


@dataclass
class Reference:
    name: str
    source: str
    size: int
    mtime: float
    path: Path | None = None
    archive: tarfile.TarFile | None = None
    member: tarfile.TarInfo | None = None

    def read(self) -> bytes:
        if self.path is not None:
            return self.path.read_bytes()
        assert self.archive is not None and self.member is not None
        stream = self.archive.extractfile(self.member)
        assert stream is not None
        with stream:
            return stream.read()


class Evidence:
    """Read selected metadata and query NPZs directly from uncompressed local archives."""

    def __init__(self, stack: ExitStack, archive_root: Path | None, live_root: Path | None) -> None:
        self.refs: dict[str, list[Reference]] = defaultdict(list)
        self.consumed: dict[str, dict[str, Any]] = {}
        if archive_root is not None:
            patterns = (
                "test12_predictions_*.tar",
                "test12_events_*.tar",
                "final_diagnostics_*.tar",
                "final_delivery_*.tar",
                "final_fits_*.tar",
            )
            for path in sorted(
                {path for pattern in patterns for path in archive_root.glob(pattern)}
            ):
                archive = stack.enter_context(tarfile.open(path, "r:"))
                for member in archive.getmembers():
                    if member.isfile() and self.wanted(member.name):
                        self.refs[member.name.lstrip("/")].append(
                            Reference(
                                member.name.lstrip("/"),
                                f"{path}::{member.name}",
                                member.size,
                                member.mtime,
                                archive=archive,
                                member=member,
                            )
                        )
        if live_root is not None:
            # Only named evidence trees. No raw-media or model/checkpoint recursion.
            trees = (
                "ejepa_final_runs/final/test12/staged_sequences",
                "ejepa_final_runs/final/test12",
                "ejepa_final_diagnostics",
                "ejepa_test12_events/sequences",
                "ejepa_test12_events/raw_receipts",
                "ejepa_test12_events/tensors",
                "ejepa_test12_rgb/receipts",
            )
            seen: set[Path] = set()
            for tree in trees:
                folder = live_root / tree
                for path in folder.rglob("*"):
                    if not path.is_file() or path in seen:
                        continue
                    seen.add(path)
                    name = "workspace/" + path.relative_to(live_root).as_posix()
                    if self.wanted(name):
                        stat = path.stat()
                        self.refs[name].append(
                            Reference(name, str(path), stat.st_size, stat.st_mtime, path=path)
                        )

    @staticmethod
    def wanted(name: str) -> bool:
        if name.endswith(".npz"):
            return "/staged_sequences/" in name
        return name.endswith((".json", "_samples.jsonl"))

    def reference(self, name: str) -> Reference:
        return max(self.refs[name], key=lambda value: (value.mtime, value.source))

    def data(self, name: str) -> bytes:
        reference = self.reference(name)
        data = reference.read()
        if "/staged_sequences/diagnostics/" in name:
            for other in self.refs[name]:
                if other is not reference and other.read() != data:
                    raise ValueError(f"Immutable diagnostic copies conflict: {name}")
        self.consumed[name] = {
            "member": name,
            "source": reference.source,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        return data

    def json(self, name: str) -> dict[str, Any]:
        return json.loads(self.data(name))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Nonfinite or negative recorded duration")
    return {
        "count": len(values),
        "sum_seconds": float(array.sum()),
        "mean_seconds": float(array.mean()),
        "min_seconds": float(array.min()),
        "max_seconds": float(array.max()),
        **{f"p{q}_seconds": float(np.percentile(array, q, method="linear")) for q in (50, 95, 99)},
    }


def query_rows(evidence: Evidence) -> list[dict[str, Any]]:
    rows = {}
    for name in sorted(evidence.refs):
        if "/staged_sequences/diagnostics/" not in name or not name.endswith(".json"):
            continue
        value = evidence.json(name)
        if value.get("schema") != "test12_staged_query_diagnostic_v1":
            raise ValueError("Unknown query diagnostic schema")
        if (
            value.get("private_targets_read") is not False
            or value.get("latency_benchmark") is not False
            or value.get("gpu_kernel_time_measured") is not False
            or value.get("timing_scope")
            != "PREDICT_WITH_CACHE_AND_READ_ONLY_HOOKS_EXCLUDES_DIAGNOSTIC_IO"
        ):
            raise ValueError("Diagnostic scientific/timing scope differs")
        token = value["sample_token"]
        if token in rows:
            raise ValueError("Physical query diagnostic duplicated under another filename")
        npz_name = name.removesuffix(".json") + ".npz"
        raw = evidence.data(npz_name)
        if hashlib.sha256(raw).hexdigest() != value["npz_sha256"]:
            raise ValueError(f"Diagnostic NPZ differs: {token}")
        row = {
            key: value[key]
            for key in (
                "sample_token",
                "sequence_id",
                "track_id",
                "query_time_us",
                "prediction",
                "run_binding_sha256",
                "fallback_source",
                "instrumented_eager_wall_seconds",
                "timing_scope",
                "npz_sha256",
            )
        }
        row["diagnostic_npz_container_bytes"] = len(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as compressed:
            row["diagnostic_npy_members_uncompressed_bytes"] = sum(
                item.file_size for item in compressed.infolist()
            )
            row["diagnostic_npy_members_compressed_bytes"] = sum(
                item.compress_size for item in compressed.infolist()
            )
            row["diagnostic_zip_methods"] = sorted(
                {item.compress_type for item in compressed.infolist()}
            )
        for modality in ("event", "rgb"):
            cache = value["cache"][modality]
            row[f"{modality}_history_count"] = cache["history_length"]
            row[f"{modality}_cache_hits"] = cache["hits"]
            row[f"{modality}_cache_misses"] = cache["history_length"] - cache["hits"]
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            row["diagnostic_uncompressed_array_bytes"] = sum(
                arrays[key].nbytes for key in arrays.files
            )
            for key in sorted(arrays.files):
                value_array = arrays[key]
                if ".output_" in key and value_array.size <= 8:
                    for index, item in enumerate(value_array.reshape(-1)):
                        row[key if value_array.size == 1 else f"{key}.{index}"] = item.item()
                if key.endswith("_new_ids"):
                    row[f"{key}_count"] = value_array.size
        rows[token] = row
    return sorted(
        rows.values(),
        key=lambda value: (value["sequence_id"], value["query_time_us"], value["sample_token"]),
    )


def telemetry_rows(evidence: Evidence) -> list[dict[str, Any]]:
    rows = {}
    for name in sorted(evidence.refs):
        if "/ejepa_final_diagnostics/" not in name or not name.endswith("_samples.jsonl"):
            continue
        for line in evidence.data(name).splitlines():
            value = json.loads(line)
            row = {
                "utc": value["utc"],
                "session": name,
                "unix_seconds": value.get("unix_seconds"),
                "host_cpu_percent": value.get("host_cpu_percent"),
                "sample_collection_wall_seconds": value.get("collector_wall_seconds"),
            }
            cg = value.get("container_cgroup", {})
            for key, candidates in {
                "cgroup_memory_bytes": ("memory.current", "memory/memory.usage_in_bytes"),
                "cgroup_memory_limit_bytes": ("memory.max", "memory/memory.limit_in_bytes"),
                "cgroup_cpu_usage_ns": ("cpuacct/cpuacct.usage",),
            }.items():
                row[key] = next((cg[item] for item in candidates if item in cg), None)
            for index, gpu in enumerate(value.get("gpu_rows", [])):
                for field, item in zip(value["gpu_fields"], gpu, strict=True):
                    row[f"gpu{index}.{field}"] = item
            for key, item in value.get("network_counters", {}).items():
                row[f"network.{key}"] = item
            for path, status in value.get("status", {}).items():
                if isinstance(status.get("value"), dict):
                    row[f"stage.{Path(path).parent.name}.{Path(path).name}"] = status["value"].get(
                        "status"
                    )
            if row["utc"] in rows and rows[row["utc"]] != row:
                raise ValueError("Conflicting telemetry at identical UTC timestamp")
            rows[row["utc"]] = row
    return [rows[key] for key in sorted(rows)]


def resource_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize observed five-minute UTC bins; do not fill gaps with invented samples."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("unix_seconds") is not None:
            grouped[int(float(row["unix_seconds"]) // 300)].append(row)
    result = []
    for bucket, samples in sorted(grouped.items()):
        first, last = samples[0], samples[-1]
        seconds = float(last["unix_seconds"]) - float(first["unix_seconds"])
        record = {
            "bin_start_unix_seconds": bucket * 300,
            "first_sample_utc": first["utc"],
            "last_sample_utc": last["utc"],
            "samples": len(samples),
            "observed_span_seconds": seconds,
        }
        for field in (
            "gpu0.utilization.gpu",
            "gpu0.memory.used",
            "gpu0.power.draw",
            "gpu0.temperature.gpu",
            "cgroup_memory_bytes",
        ):
            values = []
            for sample in samples:
                try:
                    values.append(float(sample[field]))
                except (KeyError, TypeError, ValueError):
                    pass
            if values:
                record[f"{field}.sample_mean"] = float(np.mean(values))
                record[f"{field}.sample_max"] = max(values)
        for field, divisor in (
            ("cgroup_cpu_usage_ns", 1e9),
            ("network.bytes_recv", 1),
            ("network.bytes_sent", 1),
        ):
            try:
                delta = float(last[field]) - float(first[field])
                if seconds > 0 and delta >= 0:
                    record[f"{field}.delta_per_second"] = delta / divisor / seconds
            except (KeyError, TypeError, ValueError):
                pass
        result.append(record)
    return result


def build_report(
    *,
    archive_root: Path | None,
    live_root: Path | None,
    output: Path,
    mode: str,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    with ExitStack() as stack:
        evidence = Evidence(stack, archive_root, live_root)
        queries = query_rows(evidence)
        telemetry = telemetry_rows(evidence)
        sequence: dict[str, dict[str, Any]] = defaultdict(dict)
        prep_rows, training_rows = [], []
        prediction_receipt = package = stopped = None
        model_seal_sha256 = None
        head_export_receipt = None
        for name in sorted(evidence.refs):
            if not name.endswith(".json") or "/staged_sequences/diagnostics/" in name:
                continue
            selected = any(
                marker in name
                for marker in (
                    "/ejepa_test12_events/sequences/",
                    "/ejepa_test12_events/raw_receipts/",
                    "/ejepa_test12_events/tensors/",
                    "/ejepa_test12_rgb/receipts/",
                    ".complete.json",
                    "EVIDENCE_MANIFEST.json",
                    "RUN_PROVENANCE.json",
                    "/PREDICTION_RECEIPT.json",
                    "/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json",
                    "/COLLECTOR_STOPPED.json",
                    "/final/MODEL_SEAL.json",
                    "/final/PHASE_ONLY_HEAD.json",
                )
            )
            if not selected:
                continue
            value = evidence.json(name)
            if "/ejepa_test12_events/sequences/" in name:
                row = sequence[value["sequence"]]
                row.update(
                    event_npz_total_bytes=value["bytes"],
                    event_tensor_count=value["count"],
                    event_preparation_wall_seconds=value["seconds"],
                )
            elif "/ejepa_test12_events/raw_receipts/" in name:
                seq = Path(value["relative_path"]).parts[2]
                sequence[seq].update(
                    raw_event_h5_bytes=value["size_bytes"],
                    raw_event_sha256=value["sha256"],
                    raw_event_storage_state=value.get("storage_state"),
                )
            elif "/ejepa_test12_events/tensors/" in name:
                prep_rows.append(
                    {
                        "sample_token": value["sample_token"],
                        "event_preparation_wall_seconds": value.get("seconds"),
                        "npz_container_bytes": value.get("bytes"),
                        "npz_sha256": value["sha256"],
                        "raw_sha256": value["raw_sha256"],
                    }
                )
            elif "/ejepa_test12_rgb/receipts/" in name:
                seq = Path(value["relative_path"]).parts[2]
                row = sequence[seq]
                row["raw_rgb_tar_bytes"] = row.get("raw_rgb_tar_bytes", 0) + value["size_bytes"]
                row["raw_rgb_shards"] = row.get("raw_rgb_shards", 0) + 1
            elif name.endswith(".complete.json"):
                row = sequence[value["sequence_id"]]
                row["prediction_status"] = value["status"]
                row["completed_canonical_queries"] = len(value["canonical_tokens"])
                row["feature_cache_scope"] = value.get("feature_cache_scope")
                feature = name.removesuffix(".complete.json") + ".features.npz"
                row["feature_npz_container_bytes"] = evidence.reference(feature).size
            elif name.endswith("EVIDENCE_MANIFEST.json"):
                training_rows.extend(
                    {**row, "evidence_manifest": name} for row in value.get("phase_timings", [])
                )
            elif name.endswith("RUN_PROVENANCE.json"):
                training_rows.append({"source": name, **value})
            elif name.endswith("/PREDICTION_RECEIPT.json"):
                prediction_receipt = value
            elif name.endswith("/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json"):
                package = value
            elif name.endswith("/COLLECTOR_STOPPED.json"):
                stopped = value
            elif name.endswith("/final/MODEL_SEAL.json"):
                model_seal_sha256 = evidence.consumed[name]["sha256"]
            elif name.endswith("/final/PHASE_ONLY_HEAD.json"):
                head_export_receipt = value
        if mode == "final":
            if not (
                prediction_receipt
                and prediction_receipt.get("status") == "COMPLETE"
                and prediction_receipt.get("physical_queries") == EXPECTED_PHYSICAL
                and prediction_receipt.get("population") == EXPECTED_TOKENS
                and prediction_receipt.get("private_targets_read") is False
                and package
                and package.get("status") == "PACKAGED_NOT_SUBMITTED"
                and package.get("external_submission") is False
                and stopped
                and stopped.get("status") == "STOPPED"
                and len(queries) == EXPECTED_PHYSICAL
            ):
                raise ValueError(
                    "FINAL requires package, 5805 physical queries/6762 tokens "
                    "and stopped telemetry"
                )
            sample_names = {name for name in evidence.consumed if name.endswith("_samples.jsonl")}
            stopped_names = {item["path"].lstrip("/") for item in stopped["sample_files"]}
            if sample_names != stopped_names:
                raise ValueError("Final telemetry session coverage differs from closed collector")
            for item in stopped["sample_files"]:
                source = evidence.consumed[item["path"].lstrip("/")]
                if source["sha256"] != item["sha256"] or source["bytes"] != item["bytes"]:
                    raise ValueError("Telemetry changed after collector closure")
        latency = []
        for seq in sorted(set(sequence) | {row["sequence_id"] for row in queries}):
            subset = [row for row in queries if row["sequence_id"] == seq]
            sequence[seq]["sequence_id"] = seq
            sequence[seq]["diagnostic_queries_present"] = len(subset)
            sequence[seq]["query_diagnostic_npz_bytes"] = sum(
                row["diagnostic_npz_container_bytes"] for row in subset
            )
            for group, selected_queries in (
                ("all", subset),
                (
                    "cache_miss_present",
                    [
                        row
                        for row in subset
                        if row["event_cache_misses"] + row["rgb_cache_misses"] > 0
                    ],
                ),
                (
                    "all_history_cached",
                    [
                        row
                        for row in subset
                        if row["event_cache_misses"] + row["rgb_cache_misses"] == 0
                    ],
                ),
            ):
                latency.append(
                    {
                        "sequence_id": seq,
                        "cache_stratum": group,
                        **distribution(
                            [row["instrumented_eager_wall_seconds"] for row in selected_queries]
                        ),
                    }
                )
        latency.append(
            {
                "sequence_id": "ALL_PHYSICAL_QUERIES",
                "cache_stratum": "all",
                **distribution([row["instrumented_eager_wall_seconds"] for row in queries]),
            }
        )
        methodology = {
            "status": "FINAL" if mode == "final" else "PARTIAL",
            "expected_official_tokens": EXPECTED_TOKENS,
            "expected_physical_queries": EXPECTED_PHYSICAL,
            "expected_alias_tokens": EXPECTED_TOKENS - EXPECTED_PHYSICAL,
            "observed_physical_query_diagnostics": len(queries),
            "alias_policy": "Time each physical query once; aliases are not extra inference.",
            "wall_timing": (
                "Recorded eager predict() with caches and read-only hooks; "
                "excludes diagnostic writes, model initialization and raw preparation."
            ),
            "latency_benchmark": False,
            "gpu_kernel_latency_measured": False,
            "resource_windows": "Five-minute UTC bins; observed samples only, no gap filling.",
            "query_resource_alignment": (
                "query_time_us is sensor time, not execution UTC; no invented alignment "
                "of individual query latency to passive device telemetry."
            ),
            "quantiles": "numpy.percentile(method=linear); physical queries, no alias weighting.",
            "phase_units": "Native point_phase/raw_location/q10/q90; TTC outputs in seconds.",
            "interval_calibration": "Not calibrated or coverage-scored on private test targets.",
            "compression": (
                "On-disk sizes: H5, TAR and NPZ contain different data; "
                "their size ratios are not equivalent-content compression trials."
            ),
            "raw_retention": "Report does not delete or regenerate any raw input.",
            "training_ledger": (
                "Snapshots may repeat fits and include zero-update re-entry; "
                "calendar spans are not active GPU time."
            ),
            "training_losses": (
                "Saved stage-specific training objectives, not test errors; "
                "loss magnitudes are not directly comparable across different stages."
            ),
            "official_accuracy": "No test GT, leaderboard score, error metric or SOTA claim.",
            "deterministic": "Sorted tables and hashes, no report-generation time in content.",
            "model_seal_file_sha256": model_seal_sha256,
            "head_export_receipt": head_export_receipt,
            "report_code_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "runpod_final_diagnostics_tables.py",
                    "runpod_final_diagnostics_report.py",
                )
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        csv_write(output / "QUERY_DIAGNOSTICS.csv", queries)
        csv_write(
            output / "EVENT_PREPARATION.csv", sorted(prep_rows, key=lambda row: row["sample_token"])
        )
        csv_write(output / "SEQUENCE_FOOTPRINT.csv", [sequence[key] for key in sorted(sequence)])
        csv_write(output / "INSTRUMENTED_WALL_DISTRIBUTIONS.csv", latency)
        csv_write(output / "PASSIVE_TELEMETRY.csv", telemetry)
        csv_write(output / "RESOURCE_WINDOWS_5MIN.csv", resource_windows(telemetry))
        csv_write(output / "HISTORICAL_TRAINING_LEDGER.csv", training_rows)
        source_manifest = [evidence.consumed[key] for key in sorted(evidence.consumed)]
        if evidence_root is not None:
            for filename in (
                "ENVIRONMENT_FULL_20261027.json",
                "VALIDATION_EVIDENCE.json",
                "DEPLOYMENT_TRAIN40_SCOPE.json",
                "SCIENTIFIC_CLAIM_INVENTORY.json",
            ):
                path = evidence_root / filename
                if path.is_file():
                    raw = path.read_bytes()
                    target = output / "evidence" / filename
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
                    source_manifest.append(
                        {
                            "member": f"local_evidence/{filename}",
                            "source": str(path),
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
        (output / "SOURCE_MANIFEST.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True)
        )
        (output / "METHODOLOGY.json").write_text(json.dumps(methodology, indent=2, sort_keys=True))
        summary = {
            "status": methodology["status"],
            "physical_queries": len(queries),
            "sequences_with_diagnostics": len({row["sequence_id"] for row in queries}),
            "telemetry_samples": len(telemetry),
            "input_files_read": len(source_manifest),
            "instrumented_wall": latency[-1],
            "source_manifest_sha256": hashlib.sha256(
                (output / "SOURCE_MANIFEST.json").read_bytes()
            ).hexdigest(),
        }
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        if mode == "partial":
            snapshot = output / ("PARTIAL_" + summary["source_manifest_sha256"][:16] + ".json")
            snapshot.write_text(json.dumps(summary, indent=2, sort_keys=True))
        whole = latency[-1]
        lines = [
            f"# {methodology['status']}: instrumented inference diagnostics",
            "",
            f"Physical queries represented: {len(queries)} / {EXPECTED_PHYSICAL}. "
            f"The official output has {EXPECTED_TOKENS} tokens, including "
            f"{EXPECTED_TOKENS - EXPECTED_PHYSICAL} aliases.",
            "",
            "These are existing eager predict calls with their caches and diagnostic hooks. "
            "They exclude diagnostic writes, model initialization and raw preparation. "
            "They are not isolated, GPU-kernel or whole-service latency.",
            "",
            "| Physical calls | p50 seconds | p95 seconds | p99 seconds |",
            "|---:|---:|---:|---:|",
            f"| {whole['count']} | {whole.get('p50_seconds')} | "
            f"{whole.get('p95_seconds')} | {whole.get('p99_seconds')} |",
            "",
            "QUERY_DIAGNOSTICS.csv preserves TTC predictions, phases, native phase intervals, "
            "cache hits/misses and compressed/uncompressed diagnostic sizes. "
            "SEQUENCE_FOOTPRINT.csv separates raw H5/RGB media, prepared event tensors, "
            "feature caches and query diagnostics.",
            "",
            "EVENT_PREPARATION.csv and HISTORICAL_TRAINING_LEDGER.csv retain recorded timing. "
            "PASSIVE_TELEMETRY.csv retains measured resource samples. Calendar spans include "
            "pauses and completed-run re-entry; they are not active training hours. "
            "Historical concurrent throughput uses one shared wall interval.",
            "",
            "Head-export batch1/7 parity is separate numerical evidence. No private test "
            "labels, accuracy results, official leaderboard measurements or SOTA claims "
            "are included. No training, inference, checkpoint selection or previous tests "
            "were repeated to generate these tables.",
            "",
        ]
        (output / "WHITEPAPER_DIAGNOSTICS.md").write_text("\n".join(lines))
        return summary

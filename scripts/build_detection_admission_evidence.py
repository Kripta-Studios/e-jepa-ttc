"""Build and verify the public data/RAM admission evidence bundle v2.

Only the explicit allow-list below is packaged.  Dataset media, model state and
synthetic evaluation payloads are deliberately excluded.  Machine-local paths
are recursively redacted before deterministic gzip compression.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "docs/detection_sota_20261029/evidence/current_v2"
SCHEMA = "phase_ttc_detection_admission_evidence_manifest_v2"


@dataclass(frozen=True)
class SourceSpec:
    source: str
    packaged: str
    kind: str = "json"
    gzip_compress: bool = False
    redact_paths: bool = True


SOURCES = (
    # Final D4096 v5 population and exact 28,054-member media admission.
    SourceSpec("output/campaign_data/d4096_v5_readable/receipt.json", "data_v5/receipt.json"),
    SourceSpec(
        "output/campaign_data/d4096_v5_readable/d4096_coverage.json",
        "data_v5/d4096_coverage.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/campaign_data/d4096_v5_readable/d4096_manifest.json",
        "data_v5/d4096_manifest.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/campaign_data/d4096_v5_readable/population_audit.json",
        "data_v5/population_audit.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/campaign_data/d4096_v5_readable/media_admission.json",
        "data_v5/media_admission.json.gz",
        gzip_compress=True,
    ),
    # Original exhaustive preflight and its progress ledgers.
    SourceSpec(
        "output/campaign_data/rgb_full_preflight_v1/summary.json",
        "preflight/summary.json",
    ),
    SourceSpec(
        "output/campaign_data/rgb_full_preflight_v1/run_manifest.json",
        "preflight/run_manifest.json",
    ),
    SourceSpec(
        "output/campaign_data/rgb_full_preflight_v1/token_status.jsonl",
        "preflight/token_status.jsonl.gz",
        kind="jsonl",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/campaign_data/rgb_full_preflight_v1/shard_status.jsonl",
        "preflight/shard_status.jsonl.gz",
        kind="jsonl",
        gzip_compress=True,
    ),
    # Strict failure reproduction and native TAR hash receipt (no TAR bytes).
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/failure_rgb_reproduction.json",
        "failure/failure_rgb_reproduction.json",
    ),
    SourceSpec(
        "output/detection_campaign/r0_d4096_seed7_v2/corrupt_member_shard_hash.json",
        "failure/native_tar_hash_receipt.json",
    ),
    # Final memory profile and all seven receipts named by profile.evidence.
    SourceSpec(
        "output/evaluation_memory/streaming_writer_profile_v1.json",
        "memory/streaming_writer_profile_v1.json",
    ),
    SourceSpec(
        "output/evaluation_memory/serialization_comparison_1024_v1.json",
        "memory/serialization_comparison_1024_v1.json",
    ),
    SourceSpec(
        "output/evaluation_memory/worker_4096_stream.json",
        "memory/worker_4096_stream.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/evaluation_memory/watchdog_4096_stream.json",
        "memory/watchdog_4096_stream.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/evaluation_memory/worker_4096.json",
        "memory/worker_4096.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/evaluation_memory/watchdog_4096.json",
        "memory/watchdog_4096.json.gz",
        gzip_compress=True,
    ),
    SourceSpec(
        "output/evaluation_memory/full_selection_streaming_admission_v1.json",
        "memory/full_selection_streaming_admission_v1.json",
    ),
    SourceSpec(
        "output/evaluation_memory/metric_parity_receipt_v1.json",
        "memory/metric_parity_receipt_v1.json",
    ),
    # Exact source files used to produce/admit the evidence.
    SourceSpec(
        "scripts/preflight_detection_rgb_assets.py",
        "source/preflight_detection_rgb_assets.py",
        kind="text",
    ),
    SourceSpec(
        "scripts/revise_detection_campaign_media.py",
        "source/revise_detection_campaign_media.py",
        kind="text",
    ),
    SourceSpec(
        "scripts/profile_detection_evaluation_memory.py",
        "source/profile_detection_evaluation_memory.py",
        kind="text",
    ),
    SourceSpec(
        "scripts/migrate_detection_campaign_writer.py",
        "source/migrate_detection_campaign_writer.py",
        kind="text",
    ),
)

SOURCE_BY_PATH = {spec.source: spec for spec in SOURCES}
MEMORY_PREFIX = "output/evaluation_memory/"
DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/]")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _gzip(data: bytes) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=target, compresslevel=9, mtime=0) as stream:
        stream.write(data)
    return target.getvalue()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str) and DRIVE_PATH.search(value):
        if DRIVE_PATH.match(value):
            name = PureWindowsPath(value.replace("/", "\\")).name
            return f"<LOCAL_PATH>/{name}" if name else "<LOCAL_PATH>"
        return "<LOCAL_PATH_REDACTED_STRING>"
    return value


def _transform(spec: SourceSpec, raw: bytes) -> bytes:
    if not spec.redact_paths:
        return raw
    if spec.kind == "json":
        return _json_bytes(_redact(json.loads(raw.decode("utf-8"))))
    if spec.kind == "jsonl":
        lines = raw.decode("utf-8").splitlines()
        return b"".join(
            json.dumps(_redact(json.loads(line)), sort_keys=True, ensure_ascii=False).encode()
            + b"\n"
            for line in lines
            if line.strip()
        )
    if spec.kind == "text":
        text = raw.decode("utf-8")
        return re.sub(r"[A-Za-z]:[\\/][^\"']+", "<LOCAL_PATH>", text).encode("utf-8")
    raise RuntimeError(f"path redaction unsupported for {spec.kind}: {spec.source}")


def _load_json(payloads: dict[str, bytes], source: str) -> Any:
    _require(source in payloads, f"source not present in bundle: {source}")
    return json.loads(payloads[source].decode("utf-8"))


def _load_jsonl(payloads: dict[str, bytes], source: str) -> list[Any]:
    _require(source in payloads, f"source not present in bundle: {source}")
    return [json.loads(line) for line in payloads[source].decode("utf-8").splitlines() if line]


def _derive(payloads: dict[str, bytes], source_hashes: dict[str, str]) -> dict[str, Any]:
    receipt = _load_json(payloads, SOURCES[0].source)
    coverage = _load_json(payloads, SOURCES[1].source)
    manifest = _load_json(payloads, SOURCES[2].source)
    audit = _load_json(payloads, SOURCES[3].source)
    media = _load_json(payloads, SOURCES[4].source)
    preflight = _load_json(payloads, SOURCES[5].source)
    preflight_run = _load_json(payloads, SOURCES[6].source)
    tokens = _load_jsonl(payloads, SOURCES[7].source)
    shards = _load_jsonl(payloads, SOURCES[8].source)
    failure = _load_json(payloads, SOURCES[9].source)
    tar_receipt = _load_json(payloads, SOURCES[10].source)
    memory = _load_json(payloads, SOURCES[11].source)

    _require(receipt["status"] == "PASS", "v5 receipt is not PASS")
    _require(receipt["population_audit_byte_exact"] is True, "population audit not byte exact")
    _require(receipt["train_frames"] == 4096, "unexpected v5 train size")
    _require(receipt["selection_frames"] == 1024, "unexpected selection size")
    _require(receipt["full_selection_frames"] == 23958, "unexpected full selection size")
    replacement = receipt["replacement"]
    _require(replacement["old"] != replacement["new"], "replacement must change one token")
    _require(coverage["replacement"]["old_index"] == replacement["index"], "index mismatch")
    _require(
        coverage["replacement"]["new"]["sample_token"] == replacement["new"],
        "replacement token mismatch",
    )
    _require(len(coverage["replacement"]["attempts"]) == 1, "expected one admitted attempt")
    _require(manifest["schema"].startswith("eap_detection_campaign_manifest"), "bad manifest")
    _require(audit["schema"].startswith("eap_detection_population_audit"), "bad audit")
    _require(media["status"] == "PASS", "media admission is not PASS")
    _require(len(media["excluded"]) == 1, "expected exactly one excluded token")
    _require(len(media["final_media"]) == 28054, "media ledger must contain 28,054 hashes")
    _require(replacement["old"] not in media["final_media"], "excluded token remains admitted")
    _require(replacement["new"] in media["final_media"], "replacement token not admitted")

    token_counts = {
        status: sum(row["status"] == status for row in tokens) for status in ("PASS", "FAIL")
    }
    shard_counts = {
        status: sum(row["status"] == status for row in shards) for status in ("PASS", "FAIL")
    }
    _require(len(tokens) == 28054, "preflight token ledger length mismatch")
    _require(token_counts == preflight["token_status_counts"], "preflight token counts mismatch")
    _require(len(shards) == 107, "preflight shard ledger length mismatch")
    _require(shard_counts == preflight["shard_status_counts"], "preflight shard counts mismatch")
    tar_file_hash_matches = sum(row["sha256_matches_pinned"] is True for row in shards)
    _require(tar_file_hash_matches == len(shards), "not all TAR file hashes match their pins")
    _require(preflight_run["pil_load_truncated_images"] is False, "preflight was not strict")
    failed_frames = [row for row in failure["frames"] if row["status"] == "FAILED"]
    _require(len(failed_frames) == 1, "failure reproduction must contain one failure")
    _require(failed_frames[0]["sample_token"] == replacement["old"], "failure token mismatch")
    _require(tar_receipt["matches_pinned_sha256"] is True, "native TAR differs from pin")
    _require(tar_receipt["sha256"] == tar_receipt["expected"]["sha256"], "TAR hash mismatch")

    _require(memory["status"] == "PASS", "memory profile is not PASS")
    _require(memory["peak_memory_admission"] is True, "memory admission did not pass")
    _require(memory["full_population_observed"] is False, "projection mislabeled as observed")
    _require(len(memory["evidence"]) == 7, "memory profile must reference exactly seven receipts")
    for item in memory["evidence"].values():
        source = MEMORY_PREFIX + item["path"]
        _require(source in source_hashes, f"missing memory evidence: {source}")
        _require(
            source_hashes[source] == item["sha256"],
            f"memory evidence hash mismatch: {source}",
        )

    preflight_script = "scripts/preflight_detection_rgb_assets.py"
    profile_script = "scripts/profile_detection_evaluation_memory.py"
    _require(
        source_hashes[preflight_script] == preflight_run["implementation_sha256"],
        "preflight implementation hash mismatch",
    )
    _require(
        source_hashes[profile_script] == memory["profiler_sha256"],
        "memory profiler hash mismatch",
    )

    return {
        "schema": "phase_ttc_detection_admission_results_v2",
        "data_admission": {
            "status": receipt["status"],
            "manifest_sha256": receipt["new_manifest_sha256"],
            "media_admission_sha256": receipt["media_admission_sha256"],
            "train_frames": receipt["train_frames"],
            "selection_frames": receipt["selection_frames"],
            "full_selection_frames": receipt["full_selection_frames"],
            "media_hash_records": len(media["final_media"]),
            "substitution_count": len(media["excluded"]),
            "replacement": replacement,
            "train_evaluable_v2": coverage["train"]["evaluable_annotations_v2"],
            "trainable_legacy": coverage["train"]["trainable_annotations_legacy_profile"],
            "rare_trainable_identities": {
                name: coverage["train"]["trainable_identity_count"][name]
                for name in ("bicycle", "other_vehicle")
            },
        },
        "original_preflight": {
            "status": preflight["status"],
            "token_records": len(tokens),
            "token_status_counts": token_counts,
            "shard_records": len(shards),
            "shard_status_counts": shard_counts,
            "tar_file_hash_matches": tar_file_hash_matches,
            "native_bad_member_count": token_counts["FAIL"],
            "strict_decode": not preflight_run["pil_load_truncated_images"],
            "native_tar_matches_pin": tar_receipt["matches_pinned_sha256"],
        },
        "memory_admission": {
            "status": memory["status"],
            "profile_scope": memory["profile_scope"],
            "observed_frames": memory["streaming_rss_observed_sample_count"],
            "full_population_frames": memory["full_population_sample_count"],
            "full_population_observed": memory["full_population_observed"],
            "serializer_byte_parity": memory["serializer_byte_parity"],
            "metric_parity": memory["metric_parity"],
            "measured_evaluator_peak_rss_bytes": memory["measured_evaluator_peak_rss_bytes"],
            "projected_full_evaluator_peak_rss_bytes": memory[
                "projected_full_evaluator_peak_rss_bytes"
            ],
            "projected_full_total_physical_ram_bytes": memory[
                "projected_full_total_physical_ram_bytes"
            ],
            "required_physical_ram_before_migration_bytes": memory[
                "required_physical_ram_before_migration_bytes"
            ],
            "reference_receipt_count": len(memory["evidence"]),
        },
        "source_script_sha256": {
            spec.source: source_hashes[spec.source]
            for spec in SOURCES
            if spec.source.startswith("scripts/")
        },
        "limitations": [
            "The 23,958-frame memory result is a conservative projection, not an observed run.",
            "The memory population is resource-only and makes no scientific metric claim.",
            "The v5 data admission changes exactly one unreadable train token.",
            "No raw media, events, checkpoints, optimizer state, or synthetic payload is packaged.",
        ],
    }


def _render(results: dict[str, Any]) -> bytes:
    data = results["data_admission"]
    preflight = results["original_preflight"]
    memory = results["memory_admission"]
    classes = sorted(data["train_evaluable_v2"])
    lines = [
        "# Data and RAM admission evidence v2",
        "",
        "All values below are derived from the packaged receipts and ledgers.",
        "",
        "## Data admission v5",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Status | {data['status']} |",
        f"| Train frames | {data['train_frames']} |",
        f"| Selection frames | {data['selection_frames']} |",
        f"| Full selection frames | {data['full_selection_frames']} |",
        f"| Per-member media hashes | {data['media_hash_records']} |",
        f"| Exact substitutions | {data['substitution_count']} |",
        "",
        f"Replacement at index `{data['replacement']['index']}`: "
        f"`{data['replacement']['old']}` -> `{data['replacement']['new']}`.",
        "",
        "| Class | GT evaluable v2 (train) | Trainable legacy |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {data['train_evaluable_v2'][name]} | {data['trainable_legacy'][name]} |"
        for name in classes
    )
    lines.extend(
        [
            "",
            "Rare trainable identities: "
            f"bicycle={data['rare_trainable_identities']['bicycle']}, "
            f"other_vehicle={data['rare_trainable_identities']['other_vehicle']}.",
            "",
            "## Original strict preflight",
            "",
            "| Records | PASS | FAIL |",
            "|---|---:|---:|",
            f"| Tokens ({preflight['token_records']}) | "
            f"{preflight['token_status_counts']['PASS']} | "
            f"{preflight['token_status_counts']['FAIL']} |",
            f"| TAR overall member-read status ({preflight['shard_records']}) | "
            f"{preflight['shard_status_counts']['PASS']} | "
            f"{preflight['shard_status_counts']['FAIL']} |",
            "",
            f"TAR file hashes matching their pins: `{preflight['tar_file_hash_matches']}/"
            f"{preflight['shard_records']}`. Native unreadable members: "
            f"`{preflight['native_bad_member_count']}`.",
            "",
            f"Strict decode: `{preflight['strict_decode']}`. Native TAR matches its pinned "
            "SHA-256: "
            f"`{preflight['native_tar_matches_pin']}`.",
            "",
            "## RAM/serialization admission",
            "",
            "| Item | Value |",
            "|---|---:|",
            f"| Status | {memory['status']} |",
            f"| Observed frames | {memory['observed_frames']} |",
            f"| Full population frames | {memory['full_population_frames']} |",
            f"| Full population observed | {memory['full_population_observed']} |",
            "| Measured evaluator peak RSS (bytes) | "
            f"{memory['measured_evaluator_peak_rss_bytes']} |",
            f"| Projected full evaluator RSS (bytes) | "
            f"{memory['projected_full_evaluator_peak_rss_bytes']} |",
            f"| Projected detector + evaluator RAM (bytes) | "
            f"{memory['projected_full_total_physical_ram_bytes']} |",
            "| Required RAM before loading trainer, historical admission (bytes) | "
            f"{memory['required_physical_ram_before_migration_bytes']} |",
            f"| Hash-bound reference receipts | {memory['reference_receipt_count']} |",
            f"| Serializer byte parity | {memory['serializer_byte_parity']} |",
            f"| Metric parity | {memory['metric_parity']} |",
            "",
            "The 23,958-frame figures are projections, not observations or scientific metrics.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _readme() -> bytes:
    return (
        b"# Evidence bundle current_v2\n\n"
        b"This public, self-contained bundle verifies final data admission v5 and the "
        b"RAM/serialization admission. It uses an explicit source allow-list. Local paths are "
        b"redacted recursively; large ledgers use deterministic gzip (`mtime=0`).\n\n"
        b"Excluded: raw RGB/events, TAR archives, datasets, checkpoints, optimizer state and "
        b"large synthetic evaluation payloads. Original-byte SHA-256 values remain attestations "
        b"in `manifest.json`; each redacted packaged payload has its own verified hash.\n\n"
        b"Verify without the private `output/` tree:\n\n"
        b"```powershell\n"
        b"python scripts/build_detection_admission_evidence.py --from-bundle --check\n"
        b"```\n"
    )


def _encoding(spec: SourceSpec) -> str:
    parts = [f"{spec.kind}-local-path-redaction-v1" if spec.redact_paths else "identity"]
    if spec.gzip_compress:
        parts.append("gzip-mtime0")
    return "+".join(parts)


def _build() -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}
    entries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for spec in SOURCES:
        path = REPO_ROOT / spec.source
        _require(path.is_file(), f"missing allow-listed source: {spec.source}")
        raw = path.read_bytes()
        transformed = _transform(spec, raw)
        packaged = _gzip(transformed) if spec.gzip_compress else transformed
        payloads[spec.source] = transformed
        hashes[spec.source] = _sha256(raw)
        files[BUNDLE / spec.packaged] = packaged
        entries.append(
            {
                "source": spec.source,
                "source_bytes": len(raw),
                "source_sha256": hashes[spec.source],
                "packaged": spec.packaged,
                "packaged_bytes": len(packaged),
                "packaged_sha256": _sha256(packaged),
                "encoding": _encoding(spec),
            }
        )
    results = _derive(payloads, hashes)
    generated = {
        "results.json": _json_bytes(results),
        "ADMISSION_RESULTS.md": _render(results),
        "README.md": _readme(),
        "generator/build_detection_admission_evidence.py": Path(__file__).read_bytes(),
    }
    files.update({BUNDLE / name: data for name, data in generated.items()})
    manifest = {
        "schema": SCHEMA,
        "bundle": "current_v2",
        "deterministic": True,
        "explicit_source_count": len(SOURCES),
        "source_sha256_semantics": (
            "original-byte attestation; redacted payloads are bound separately by packaged_sha256"
        ),
        "sources": entries,
        "generated": [
            {"packaged": name, "packaged_bytes": len(data), "packaged_sha256": _sha256(data)}
            for name, data in generated.items()
        ],
        "excluded": [
            "raw RGB/images/events/TAR archives/datasets",
            "checkpoints/model weights/optimizer state",
            "synthetic evaluation payloads",
            "files outside the explicit SOURCES allow-list",
        ],
    }
    files[BUNDLE / "manifest.json"] = _json_bytes(manifest)
    return files


def _bundle_path(relative: str) -> Path:
    _require(not Path(relative).is_absolute(), f"absolute package path: {relative}")
    path = BUNDLE / relative
    _require(
        path.resolve().is_relative_to(BUNDLE.resolve()),
        f"package path escapes bundle: {relative}",
    )
    return path


def _verified_bytes(entry: dict[str, Any]) -> bytes:
    path = _bundle_path(entry["packaged"])
    _require(path.is_file(), f"missing packaged file: {entry['packaged']}")
    data = path.read_bytes()
    _require(len(data) == entry["packaged_bytes"], f"size mismatch: {entry['packaged']}")
    _require(_sha256(data) == entry["packaged_sha256"], f"hash mismatch: {entry['packaged']}")
    return data


def _from_bundle() -> dict[Path, bytes]:
    manifest_path = BUNDLE / "manifest.json"
    _require(manifest_path.is_file(), "missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == SCHEMA, "unsupported manifest schema")
    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for entry in manifest["sources"]:
        data = _verified_bytes(entry)
        if "gzip-mtime0" in entry["encoding"]:
            data = gzip.decompress(data)
        payloads[entry["source"]] = data
        hashes[entry["source"]] = entry["source_sha256"]
        if entry["encoding"] == "identity":
            _require(len(data) == entry["source_bytes"], f"source size mismatch: {entry['source']}")
            _require(
                _sha256(data) == entry["source_sha256"],
                f"source hash mismatch: {entry['source']}",
            )
    for entry in manifest["generated"]:
        _verified_bytes(entry)
    results = _derive(payloads, hashes)
    expected = {
        BUNDLE / "results.json": _json_bytes(results),
        BUNDLE / "ADMISSION_RESULTS.md": _render(results),
    }
    for path, data in expected.items():
        _require(path.read_bytes() == data, f"derived output differs: {path.name}")
    return expected


def _check(files: dict[Path, bytes], *, exact_set: bool) -> int:
    errors: list[str] = []
    if exact_set:
        expected = {path.relative_to(BUNDLE).as_posix() for path in files}
        actual = {
            path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob("*") if path.is_file()
        }
        if expected != actual:
            errors.append(
                f"bundle file set differs; missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
    for path, data in files.items():
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(REPO_ROOT)}")
        elif path.read_bytes() != data:
            errors.append(f"byte mismatch: {path.relative_to(REPO_ROOT)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: verified {len(files)} files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument(
        "--from-bundle",
        action="store_true",
        help="derive results using only current_v2, without private output files",
    )
    args = parser.parse_args()
    if args.from_bundle:
        files = _from_bundle()
        print("PASS: self-contained bundle hashes and derived tables verified")
        return 0
    files = _build()
    if args.check:
        return _check(files, exact_set=True)
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"WROTE: {len(files)} files to {BUNDLE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

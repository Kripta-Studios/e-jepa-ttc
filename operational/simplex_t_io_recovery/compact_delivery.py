"""Compact portable inference delivery; no scientific or pinned source changes.

Publish only after writers stop, using --publish (preferably through run.py).
Extracted users run: python verify_compact_delivery.py --verify-root . --output VERIFY.json
Completed optimizer states remain local and are omitted only after admission of
sealed numeric exports. Partial scientific states remain in the compact archive.
N1 weights/cache belong to the separately verified H16 archive referenced by SHA.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
H16 = ROOT / "artifacts/simplex_t/h16_replication_20261003"


def record(path: Path) -> dict:
    """Read one explicit JSON binding."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash bytes without importing the scientific environment."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Reject absolute or escaping archive/export paths."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe portable member: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"portable path escaped root: {name}")
    return path


def npz_headers(path: Path) -> dict[str, dict]:
    """Verify NPZ CRC and safe NPY headers using only the standard library."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names or len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("numeric export has duplicate members or invalid CRC")
        for name in names:
            if not name.endswith(".npy") or "/" in name or "\\" in name:
                raise ValueError("numeric export contains a non-array member")
            with archive.open(name) as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("invalid NPY magic")
                version = stream.read(2)
                if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                    raise ValueError("unsupported NPY version")
                width = 2 if version == b"\x01\x00" else 4
                length = struct.unpack("<H" if width == 2 else "<I", stream.read(width))[0]
                if length > 65536:
                    raise ValueError("unbounded numeric header")
                header = ast.literal_eval(stream.read(length).decode("utf-8"))
                if (
                    not isinstance(header, dict)
                    or not isinstance(header.get("descr"), str)
                    or not isinstance(header.get("shape"), tuple)
                    or any(type(v) is not int or v < 0 for v in header["shape"])
                    or "O" in header["descr"]
                ):
                    raise ValueError("unsafe numeric array header")
                result[name[:-4]] = header
    return result


def admit_exports(root: Path, inventory: dict) -> dict[str, dict]:
    """Bind every numeric head to a frozen endpoint, protocol and cached fragments."""
    execution = root / "execution"
    path = execution / "ANALYSIS_EXPORT_INDEX.json"
    if not path.exists():
        return {}
    index = record(path)
    protocol = record(root / "PROTOCOL_COST_CONTEXT.json")
    pin = sha(root / "PROTOCOL_COST_CONTEXT.json")
    if index["protocol_sha256"] != pin:
        raise ValueError("numeric export protocol differs")
    authorized = {row["key"]: row for row in protocol["fits"]}
    physical = {row["id"]: row for row in inventory["fits"]}
    frozen = {}
    for family, binding in index["families"].items():
        # The portable selector includes the family seal at the same execution path.
        local_name = Path(binding["seal_path"]).name
        seal_path = execution / local_name
        if sha(seal_path) != binding["seal_sha256"]:
            raise ValueError("family inventory changed since numeric export")
        seal = record(seal_path)
        if seal["protocol_sha256"] != pin or not (
            seal.get("all_family_frozen_before_evaluation")
            or seal.get("frozen_before_first_evaluation")
        ):
            raise ValueError("numeric head lacks a frozen pre-evaluation inventory")
        for row in seal["fits"]:
            if row["family"] != family or row["completed_updates"] != 2500:
                raise ValueError("numeric family contains a partial endpoint")
            if row["key"] in frozen:
                raise ValueError("duplicate frozen endpoint")
            frozen[row["key"]] = row
    admitted = {}
    for key, row in index["fits"].items():
        spec, endpoint, local = authorized[key], frozen[key], physical[key]
        source = protocol["sources"][str(row["fold"])]["wrapped"][row["arm"]]
        if (
            row["arm"] != spec["arm"]
            or row["fold"] != spec["fold"]
            or row["seed"] != spec["seed"]
            or local["saved_updates"] != 2500
            or row["checkpoint_sha256"] != local["checkpoint_sha256"]
            or row["checkpoint_sha256"] != endpoint["checkpoint_sha256"]
            or row["train_source_sha256"] != source["train_sha256"]
            or row["dev_source_sha256"] != source["dev_sha256"]
        ):
            raise ValueError("numeric head is not the admitted frozen scientific endpoint")
        weight = safe_path(execution, row["weights_path"])
        if sha(weight) != row["weights_sha256"]:
            raise ValueError("numeric weights differ from canonical export")
        if any(h["descr"] not in ("<f4", "=f4") for h in npz_headers(weight).values()):
            raise ValueError("numeric model weights must be FP32")
        for field in ("predictions", "publication"):
            if sha(safe_path(execution, row[field + "_path"])) != row[field + "_sha256"]:
                raise ValueError(f"numeric {field} changed")
        cursor = 0
        if not row["fragments"]:
            raise ValueError("numeric head has no cached inference fragments")
        for fragment in row["fragments"]:
            if fragment["start"] != cursor or fragment["stop"] <= cursor:
                raise ValueError("cached inference fragments are not contiguous")
            for field in ("input", "receipt"):
                if (
                    sha(safe_path(execution, fragment[field + "_path"]))
                    != fragment[field + "_sha256"]
                ):
                    raise ValueError(f"cached {field} changed")
            receipt = record(safe_path(execution, fragment["receipt_path"]))
            if (
                receipt["protocol_sha256"] != pin
                or receipt["source_sha256"] != row["dev_source_sha256"]
                or receipt["endpoint_sha256"] != row["checkpoint_sha256"]
                or receipt["start"] != fragment["start"]
                or receipt["stop"] != fragment["stop"]
                or receipt["input_sha256"] != fragment["input_sha256"]
            ):
                raise ValueError("cached inputs are not bound to this source/endpoint")
            cursor = fragment["stop"]
        admitted[key] = row
    for arm in {row["arm"] for row in admitted.values()}:
        rows = [row for row in admitted.values() if row["arm"] == arm]
        if len(rows) != 3 or {row["fold"] for row in rows} != {0, 1, 2}:
            raise ValueError("portable inference exports require all three folds per arm")
    return admitted


def h16_reference() -> dict:
    """Require the independently regenerated N1 bundle, without copying its payload."""
    receipt = record(H16 / "FINAL_DELIVERY.json")
    regeneration = H16 / "REGENERATION.json"
    archive = Path(receipt["archive"])
    if (
        receipt["status"] != "DELIVERED_REGENERATED_INDEPENDENT_H16"
        or archive.stat().st_size != receipt["bytes"]
        or sha(archive) != receipt["sha256"]
        or sha(regeneration) != receipt["regeneration_sha256"]
        or record(regeneration)["status"] != "SIX_NEW_HEADS_REGENERATED_FROM_EXTRACTED_BYTES"
    ):
        raise ValueError("separate H16 delivery is not verified")
    return dict(
        archive_name=archive.name,
        sha256=receipt["sha256"],
        bytes=receipt["bytes"],
        protocol_sha256=receipt["protocol_sha256"],
        regeneration_sha256=receipt["regeneration_sha256"],
        numerical_heads=6,
        separately_regenerated=True,
        included_in_this_archive=False,
    )


def admit_prior_inventory(
    night: Path = NIGHT, h16: Path = H16, root: Path = ROOT
) -> tuple[dict, dict]:
    """Reuse physical admission from N5 attempt005 after unchanged-byte checks.

    This function does not deserialize Torch states. It binds the prior audit's
    execution evidence, inventory time window, current journals and all physical
    checkpoint SHAs. Any changed or unproven dependency rejects reuse.
    """
    inventory_path = night / "ENDPOINT_INVENTORY.json"
    inventory = record(inventory_path)
    attempt_path = night / "driver_receipts/N5_DELIVERY/attempt_005.json"
    attempt = record(attempt_path)
    log_path = attempt_path.with_suffix(".log")
    if (
        attempt["phase"] != "N5_DELIVERY"
        or attempt["returncode"] != 1
        or sha(log_path) != attempt["log_sha256"]
        or "operational.simplex_t_cost_context.deliver" not in attempt["command"]
    ):
        raise ValueError("prior N5 physical admission execution evidence differs")
    log = log_path.read_text(encoding="utf-8")
    if not all(
        text in log
        for text in (
            "receipt = bundle(selected_files(), decision)",
            "resources.reserve_artifacts(",
            "InterruptedError: PAUSED_RESOURCE:",
        )
    ):
        raise ValueError("prior N5 did not reach publication after physical admission")
    start = datetime.fromisoformat(attempt["started_utc"]).timestamp()
    end = datetime.fromisoformat(attempt["ended_utc"]).timestamp()
    reconciliation_path = night / "CHECKPOINT_JOURNAL_RECONCILIATION.json"
    for path in (inventory_path, reconciliation_path):
        if not start <= path.stat().st_mtime <= end:
            raise ValueError("prior physically admitted inventory was replaced outside N5 attempt")
    protocol_path, h16_protocol_path = night / "PROTOCOL_COST_CONTEXT.json", h16 / "PROTOCOL.json"
    p, p16 = record(protocol_path), record(h16_protocol_path)
    protocol_pin, h16_pin = sha(protocol_path), sha(h16_protocol_path)
    source_pin = record(night / "SOURCE_PIN_COST_CONTEXT.json")
    if (
        source_pin["protocol_sha256"] != protocol_pin
        or source_pin["files"] != p["source_pin"]["files"]
        or p["parent_h16_protocol_sha256"] != h16_pin
        or p["window_authorization"] != record(night / "WINDOW_AUTHORIZATION.json")
        or sha(Path(p["launch"]["freeze"])) != p["historical_freeze_sha256"]
    ):
        raise ValueError("prior admission protocol/source/window/freeze pin differs")
    for row in source_pin["files"]:
        path = Path(row["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(root.resolve())
            or path.stat().st_size != row["bytes"]
            or sha(path) != row["sha256"]
        ):
            raise ValueError("pinned scientific/publication source changed")
    # The pinned main performs physical admission before inventory/report/bundle.
    # Its unchanged source and the failure location above bind the prior audit.
    queue_path = night / "QUEUE_AUTHORIZED.json"
    queue = record(queue_path)
    queued = {row["id"]: row for row in queue["fits"]}
    authorized = {row["key"]: dict(row, arm="TPR-D1-H16-C160") for row in p16["fits"]}
    authorized.update({row["key"]: row for row in p["fits"]})
    rows = inventory["fits"]
    if (
        len(queued) != 24
        or len(queue["fits"]) != 24
        or len(rows) != 24
        or {row["id"] for row in rows} != set(queued)
        or set(queued) != set(authorized)
        or queue["fit_count"] != 24
        or queue["scientific_saved_update_cap"] != 60000
    ):
        raise ValueError("prior inventory/authorized union identity differs")
    journals = dict(
        N1=record(h16 / "PHYSICAL_WORK.json"),
        N2_N3=record(night / "execution/PHYSICAL_WORK.json"),
    )
    if journals != inventory["journals"]:
        raise ValueError("physical work journals changed after prior admission")
    technical_path = night / "TECHNICAL_WORK.json"
    if inventory["technical"] != (record(technical_path) if technical_path.exists() else {}):
        raise ValueError("technical work accounting changed after prior admission")
    reconciliation = record(reconciliation_path)
    reconciled = []
    for row in rows:
        key = row["id"]
        spec, queued_row = authorized[key], queued[key]
        for field in ("fold", "seed", "arm"):
            if row[field] != spec[field] or row[field] != queued_row[field]:
                raise ValueError("prior fit differs from registered recipe")
        if queued_row["endpoint"] != 2500 or spec["updates"] != 2500:
            raise ValueError("registered endpoint differs")
        base = h16 if row["family"] == "N1" else night / "execution"
        expected = base / "fits" / key / "checkpoint_last.pt"
        progress = journals["N1" if row["family"] == "N1" else "N2_N3"]["fits"].get(key, {})
        journal_saved = progress.get("completed", 0)
        pending = progress.get("pending")
        upper = pending[1] if pending else journal_saved
        saved = row["saved_updates"]
        if (
            not journal_saved <= saved <= upper <= 2500
            or row["journal_saved_updates"] != journal_saved
            or row["checkpoint_journal_reconciled"] != (saved != journal_saved)
            or row["pending_unknown_upper"] != (upper - saved if pending else 0)
            or row["uncertain_lost_upper"] != progress.get("uncertain_lost_upper", 0)
        ):
            raise ValueError("prior physical count/journal reconciliation differs")
        if row["checkpoint"] is None:
            if expected.exists() or saved:
                raise ValueError("unadmitted checkpoint appeared after prior admission")
        elif (
            Path(row["checkpoint"]).resolve(strict=True) != expected.resolve(strict=True)
            or sha(expected) != row["checkpoint_sha256"]
        ):
            raise ValueError("physical checkpoint bytes changed after prior admission")
        if saved != journal_saved:
            reconciled.append(
                {field: row[field] for field in ("id", "journal_saved_updates", "saved_updates")}
            )
    if reconciliation != dict(
        status="READ_ONLY_PHYSICAL_CHECKPOINT_ADMISSION",
        journal_files_unchanged=True,
        fits=reconciled,
        optimizer_updates=0,
    ):
        raise ValueError("prior reconciliation receipt differs")
    saved = sum(row["saved_updates"] for row in rows)
    repeated = sum(j["accounting"]["scientific_uncertain_lost_upper"] for j in journals.values())
    repeated += sum(row["pending_unknown_upper"] for row in rows)
    if (
        saved != inventory["scientific_saved_updates"]
        or repeated != inventory["repeated_or_uncertain_upper"]
        or saved + repeated != inventory["scientific_physical_upper"]
        or sum(row["saved_updates"] == 2500 for row in rows) != inventory["endpoints"]
    ):
        raise ValueError("prior inventory aggregate accounting differs")
    by_key = {row["id"]: row for row in rows}
    seals = {}
    for path in (
        h16 / "ENDPOINTS.json",
        night / "execution/N2_ENDPOINTS.json",
        night / "execution/N3_ENDPOINTS.json",
    ):
        if not path.exists():
            continue
        seal = record(path)
        if seal["protocol_sha256"] != (h16_pin if path.parent == h16 else protocol_pin):
            raise ValueError("prior endpoint seal protocol differs")
        for endpoint in seal["fits"]:
            row = by_key[endpoint["key"]]
            if (
                row["saved_updates"] != 2500
                or row["checkpoint_sha256"] != endpoint["checkpoint_sha256"]
            ):
                raise ValueError("sealed endpoint changed since prior admission")
        seals[str(path.relative_to(root))] = sha(path)
    exports = admit_exports(night, inventory)
    evidence = dict(
        schema="reused_prior_physical_inventory_admission_v1",
        status="PRIOR_TORCH_PHYSICAL_ADMISSION_REUSED_UNCHANGED_BYTES",
        inventory_sha256=sha(inventory_path),
        prior_attempt_path=str(attempt_path.relative_to(night)),
        prior_attempt_sha256=sha(attempt_path),
        prior_attempt_log_sha256=sha(log_path),
        reconciliation_sha256=sha(reconciliation_path),
        queue_sha256=sha(queue_path),
        protocol_sha256=protocol_pin,
        h16_protocol_sha256=h16_pin,
        source_pin_sha256=sha(night / "SOURCE_PIN_COST_CONTEXT.json"),
        seals=seals,
        export_index_sha256=sha(night / "execution/ANALYSIS_EXPORT_INDEX.json"),
        included_numeric_heads=sorted(exports),
        checkpoint_sha256_all_verified=True,
        journals_logically_equal_to_prior_inventory=True,
        scientific_saved_updates=saved,
        endpoints=inventory["endpoints"],
        new_torch_state_deserialization_performed=False,
        optimizer_updates=0,
    )
    return inventory, evidence


def preserve_admitted_record(path: Path, value: dict, expected_sha256: str) -> None:
    """Check an attempted inventory write, leaving its bytes and timestamp intact."""
    if sha(path) != expected_sha256 or record(path) != value:
        raise ValueError("publisher attempted to replace prior admitted inventory")


def measured_zip_bound(files: dict[str, Path], manifest_size: int) -> dict:
    """Measure raw DEFLATE by streaming, without creating another ZIP or payload.

    Eight-KiB chunks match ZipFile.write's copy loop. Conservative per-member
    header/ZIP64/filename allowances and a compression allowance remain reserved.
    """
    compressed = 0
    uncompressed = manifest_size
    for path in files.values():
        uncompressed += path.stat().st_size
        compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
        with path.open("rb") as stream:
            while block := stream.read(8192):
                compressed += len(compressor.compress(block))
        compressed += len(compressor.flush())
    overhead = 65536 + sum(1024 + 2 * len(name.encode("utf-8")) for name in files)
    upper = compressed + manifest_size + uncompressed // 1000 + overhead
    return dict(
        uncompressed_bytes=uncompressed,
        measured_deflate_bytes=compressed,
        zip_upper_bytes=upper,
        zip_overhead_allowance_bytes=overhead,
        peak_bytes=max(3 * upper, 2 * upper + uncompressed) + 2 * 1024**2,
        compression_level=6,
        chunk_bytes=8192,
        zlib_version=zlib.ZLIB_VERSION,
        extra_zip_created=False,
    )


def omit_redundant_exports(
    files: dict[str, Path], admitted: dict[str, dict], execution: Path
) -> list[dict]:
    """Omit only hash-bound duplicates; preserve all original physical files."""
    omissions = []
    candidates = []
    for row in admitted.values():
        canonical = safe_path(execution, row["predictions_path"])
        if canonical.suffix != ".parquet" or sha(canonical) != row["predictions_sha256"]:
            raise ValueError("duplicate omission requires an admitted canonical Parquet")
        publication = record(safe_path(execution, row["publication_path"]))
        if sha(safe_path(execution, row["publication_path"])) != row["publication_sha256"]:
            raise ValueError("duplicate omission publication differs")
        csv_name = "execution/" + Path(row["predictions_path"]).with_suffix(".csv").as_posix()
        if csv_name in files and "csv_sha256" in publication:
            candidates.append(
                (csv_name, "CANONICAL_PARQUET_RETAINS_QUERY_PREDICTIONS", publication["csv_sha256"])
            )
        for fragment in row["fragments"]:
            receipt_path = safe_path(execution, fragment["receipt_path"])
            if sha(receipt_path) != fragment["receipt_sha256"]:
                raise ValueError("duplicate omission fragment receipt differs")
            receipt = record(receipt_path)
            if receipt["payload_sha256"] != fragment["output_sha256"]:
                raise ValueError("regenerable output differs from fragment receipt")
            candidates.append(
                (
                    "execution/" + Path(fragment["output_path"]).as_posix(),
                    "OUTPUT_ARRAYS_REGENERABLE_FROM_INCLUDED_WEIGHTS_AND_INPUTS",
                    fragment["output_sha256"],
                )
            )
    # Validate every candidate before filtering, leaving the selection unchanged
    # if any SHA binding fails. Scientific partial states are never candidates.
    for name, reason, expected_sha in candidates:
        if name not in files or sha(files[name]) != expected_sha:
            raise ValueError("duplicate payload differs from its canonical binding")
        omissions.append(
            dict(
                archive_path=name,
                bytes=files[name].stat().st_size,
                sha256=expected_sha,
                reason=reason,
            )
        )
    for name, _, _ in candidates:
        files.pop(name)
    return omissions


def omit_completed_draw_copies(files: dict[str, Path], execution: Path) -> list[dict]:
    """Reference recovery draw copies only after exact completed concatenation.

    Canonical draws, loss arrays, fragment receipts and progress remain included.
    Incomplete families retain every recovery copy in the selected payload.
    """
    omissions = []
    protocol_pin = sha(execution.parent / "PROTOCOL_COST_CONTEXT.json")
    for family in ("N2", "N3"):
        prefix = f"execution/analysis/{family}/"
        required = {
            "results": prefix + "RESULTS.json",
            "analysis": prefix + "ANALYSIS_RECEIPT.json",
            "complete": prefix + "bootstrap/.resume/COMPLETE.json",
            "inputs": prefix + "bootstrap/.resume/INPUTS.json",
            "canonical": prefix + "bootstrap/HIERARCHICAL_DRAWS.jsonl",
            "losses": prefix + "bootstrap/BOOTSTRAP_LOSSES.npy",
        }
        if not all(name in files for name in required.values()):
            continue
        result = record(files[required["results"]])
        analysis = record(files[required["analysis"]])
        if result.get("status") != "COMPLETE_EXPLORATORY_FAMILY_ANALYSIS":
            continue
        complete = record(files[required["complete"]])
        inputs = record(files[required["inputs"]])
        if (
            analysis.get("status") != result["status"]
            or analysis["results_sha256"] != sha(files[required["results"]])
            or result["family"] != family
            or result["protocol_sha256"] != protocol_pin
            or inputs["binding"]["family"] != family
            or inputs["binding"]["protocol_sha256"] != protocol_pin
            or complete != result["bootstrap"]
            or complete["valid_draws"] != analysis["accepted_draws"]
            or complete["attempts"] != analysis["draw_attempts"]
        ):
            raise ValueError("completed bootstrap family binding differs")
        canonical = files[required["canonical"]]
        canonical_sha = sha(canonical)
        if canonical_sha != inputs["draw_file_sha256"]:
            raise ValueError("canonical bootstrap draw bytes differ")
        candidates = sorted(
            name
            for name in files
            if name.startswith(prefix + "bootstrap/.resume/fragment_")
            and name.endswith("/DRAWS.jsonl")
        )
        pending = []
        concatenated = hashlib.sha256()
        attempt, byte_offset, accepted = 0, 0, 0
        with canonical.open("rb") as stream:
            for name in candidates:
                directory = files[name].parent
                receipt_name = name.removesuffix("DRAWS.jsonl") + "RECEIPT.json"
                losses_name = name.removesuffix("DRAWS.jsonl") + "LOSSES.npy"
                if receipt_name not in files or losses_name not in files:
                    raise ValueError("completed draw fragment lacks included receipt/array")
                receipt = record(files[receipt_name])
                if (
                    receipt["start"] != attempt
                    or receipt["stop"] <= attempt
                    or receipt["input_sha256"] != sha(files[required["inputs"]])
                    or sorted(receipt["accepted_ids"] + receipt["rejected_ids"])
                    != list(range(receipt["start"], receipt["stop"]))
                    or sha(files[losses_name]) != receipt["files"]["LOSSES.npy"]
                ):
                    raise ValueError("completed draw fragment continuity/identity differs")
                fragment_sha = hashlib.sha256()
                start_byte = byte_offset
                with (directory / "DRAWS.jsonl").open("rb") as fragment:
                    while block := fragment.read(8192):
                        if stream.read(len(block)) != block:
                            raise ValueError(
                                "bootstrap recovery copies differ from canonical bytes"
                            )
                        fragment_sha.update(block)
                        concatenated.update(block)
                        byte_offset += len(block)
                digest = fragment_sha.hexdigest()
                if digest != receipt["files"]["DRAWS.jsonl"]:
                    raise ValueError("recovery draw-copy SHA differs from fragment receipt")
                pending.append(
                    dict(
                        archive_path=name,
                        bytes=byte_offset - start_byte,
                        sha256=digest,
                        reason="RECOVERY_DRAW_COPY_LOCAL_CANONICAL_CONCATENATION_VERIFIED",
                        canonical_source=required["canonical"],
                        canonical_sha256=canonical_sha,
                        canonical_start_byte=start_byte,
                        canonical_stop_byte=byte_offset,
                        receipt_path=receipt_name,
                        receipt_sha256=sha(files[receipt_name]),
                    )
                )
                attempt = receipt["stop"]
                accepted += len(receipt["accepted_ids"])
            if stream.read(1):
                raise ValueError("bootstrap fragment copies do not cover canonical draws")
        if (
            attempt != complete["attempts"]
            or accepted != complete["valid_draws"]
            or concatenated.hexdigest() != canonical_sha
        ):
            raise ValueError("bootstrap concatenation is incomplete")
        for row in pending:
            row["concatenation_sha256"] = concatenated.hexdigest()
        omissions.extend(pending)
    # Filter only after every complete candidate family passed its admission.
    for row in omissions:
        files.pop(row["archive_path"])
    return omissions


def select_compact(namespace: dict, original: Callable[[], dict[str, Path]]) -> dict[str, Path]:
    """Filter payload only; keep every original local state and full inventory intact."""
    files = original()
    inventory = record(NIGHT / "ENDPOINT_INVENTORY.json")
    admitted = admit_exports(NIGHT, inventory)
    reference = h16_reference()
    allowed_h16 = {
        "PROTOCOL.json",
        "PROTOCOL.sha256",
        "FINAL_REPORT_H16.md",
        "NEXT_DECISION_H16.json",
        "FINAL_DELIVERY.json",
        "REGENERATION.json",
        "PHYSICAL_WORK.json",
        "ENDPOINTS.json",
        "analysis/RESULTS.json",
    }
    files = {
        name: path
        for name, path in files.items()
        if not name.startswith("compact_delivery/")
        and name
        not in {"RECOVERED_DRIVER.log", "IO_REPLACE_RETRIES.jsonl", "FINAL_DELIVERY_COMPACT.json"}
        and (not name.startswith("h16/") or name[4:] in allowed_h16)
    }
    omitted, partials, portable_rows = [], [], []
    payload_omissions = omit_redundant_exports(files, admitted, NIGHT / "execution")
    payload_omissions.extend(omit_completed_draw_copies(files, NIGHT / "execution"))

    def omit(name: str, reason: str, expected_sha: str) -> None:
        if name not in files:
            raise ValueError("registered duplicate payload missing before compact admission")
        path = files[name]
        if sha(path) != expected_sha:
            raise ValueError("duplicate payload differs from its canonical binding")
        payload_omissions.append(
            dict(archive_path=name, bytes=path.stat().st_size, sha256=expected_sha, reason=reason)
        )
        files.pop(name)

    for local in inventory["fits"]:
        row = dict(local)
        key = row["id"]
        if row["family"] == "N1":
            if row["checkpoint"] is not None and row["saved_updates"] != 2500:
                # N1 is normally closed; do not omit a hypothetical partial state.
                name = "h16/fits/" + key + "/checkpoint_last.pt"
                files[name] = Path(row["checkpoint"])
                partials.append(key)
            else:
                row["checkpoint"] = None
                row["state_scope"] = "SEPARATE_VERIFIED_H16_BUNDLE"
        elif row["checkpoint"] is not None:
            name = "execution/fits/" + key + "/checkpoint_last.pt"
            if key in admitted:
                if name not in files:
                    raise ValueError(
                        "completed local checkpoint was absent before compact filtering"
                    )
                files.pop(name)
                row["checkpoint"] = None
                row["state_scope"] = "NUMERIC_INFERENCE_EXPORT_ONLY_FULL_STATE_LOCAL"
                omitted.append(key)
            else:
                if name not in files:
                    raise ValueError("recoverable scientific checkpoint would be omitted")
                row["state_scope"] = "FULL_RECOVERY_STATE_INCLUDED_NO_NUMERIC_EXPORT"
                partials.append(key)
        portable_rows.append(row)
    scope = dict(
        schema="compact_nocturnal_inference_scope_v1",
        h16_external=reference,
        included_numeric_heads=sorted(admitted),
        completed_full_states_omitted=sorted(omitted),
        full_recovery_states_included=sorted(partials),
        all_local_checkpoints_preserved=True,
        local_full_inventory_sha256=sha(NIGHT / "ENDPOINT_INVENTORY.json"),
        raw_train_reconstruction=False,
        optimizer_updates=0,
        all_full_states_included=False,
        canonical_query_predictions="PREDICTIONS.parquet validated against ANALYSIS_EXPORT_INDEX",
        omitted_local_payloads=payload_omissions,
        omitted_payload_local_root=str(NIGHT),
        omitted_payloads_preserved_locally=True,
        raw_output_npz_container_bitwise_regeneration_certified=False,
        bootstrap_statistics_scope=(
            "Canonical draws, losses and receipts included; duplicate recovery draw copies "
            "referenced locally after byte-exact concatenation"
        ),
        technical_optimizer_states_included=True,
    )
    reuse = NIGHT / "verification/COMPACT_INVENTORY_REUSE_ADMISSION.json"
    if reuse.exists() and record(reuse)["inventory_sha256"] == scope["local_full_inventory_sha256"]:
        scope["physical_inventory_admission"] = dict(
            receipt_path=reuse.relative_to(NIGHT).as_posix(),
            receipt_sha256=sha(reuse),
            new_torch_state_deserialization_performed=False,
            status=record(reuse)["status"],
        )
    portable = dict(inventory, fits=portable_rows, scope=scope)
    staging = NIGHT / "compact_delivery"
    staging.mkdir(exist_ok=True)
    for name, value in (("ENDPOINT_INVENTORY.json", portable), ("COMPACT_SCOPE.json", scope)):
        payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        path = staging / (hashlib.sha256(payload).hexdigest()[:12] + "_" + name)
        namespace["atomic_bytes"](path, payload)
        files[name] = path
    files["LOCAL_ENDPOINT_INVENTORY.json"] = NIGHT / "ENDPOINT_INVENTORY.json"
    files["verify_compact_delivery.py"] = Path(__file__).resolve()
    # Technical optimizer proof states are not scientific fit recovery states.
    # Omit these only when required by the own-artifact cap; retain receipt and
    # hashes, and explicitly limit independent optimizer-state auditing.
    preliminary_manifest_size = len(
        (
            json.dumps(
                dict(
                    schema="essential_nocturnal_compact_manifest_v1",
                    members={
                        name: dict(bytes=path.stat().st_size, sha256="0" * 64)
                        for name, path in files.items()
                    },
                    scope=scope,
                    optimizer_updates_for_regeneration=0,
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    budget = measured_zip_bound(files, preliminary_manifest_size)
    from operational.simplex_t_cost_context.engine import ARTIFACT_CAP, artifact_inventory

    if artifact_inventory()["used_bytes"] + budget["peak_bytes"] > ARTIFACT_CAP:
        proof = record(NIGHT / "verification/ENGINE_RESUME.json")
        if (
            proof["status"] != "EXACT_SYNTHETIC_RESUME_EQUIVALENCE"
            or proof["optimizer_updates"] != 80
        ):
            raise ValueError("technical proof receipt does not permit referenced-only scope")
        for name in list(files):
            if name.startswith("verification/technical/") and name.endswith("checkpoint_last.pt"):
                omit(name, "TECHNICAL_OPTIMIZER_PROOF_STATE_LOCAL_REFERENCE_ONLY", sha(files[name]))
        scope["technical_optimizer_states_included"] = False
        scope["independent_technical_optimizer_state_audit_included"] = False
        scope["technical_resume_evidence"] = dict(
            path="verification/ENGINE_RESUME.json",
            sha256=sha(NIGHT / "verification/ENGINE_RESUME.json"),
            optimizer_updates_confirmed=80,
        )
        portable["scope"] = scope
        for name, value in (("ENDPOINT_INVENTORY.json", portable), ("COMPACT_SCOPE.json", scope)):
            payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            path = staging / (hashlib.sha256(payload).hexdigest()[:12] + "_" + name)
            namespace["atomic_bytes"](path, payload)
            files[name] = path
    return files


def bundle_compact(namespace: dict, files: dict[str, Path], decision: dict) -> dict:
    """Reuse pinned ZIP fragments and regenerate every included numerical head."""
    from operational.simplex_t_cost_context.engine import Resources

    resources = Resources()
    resources.check()
    scope = record(files["COMPACT_SCOPE.json"])
    manifest: dict = dict(
        schema="essential_nocturnal_compact_manifest_v1",
        members={
            name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in files.items()
        },
        scope=scope,
        optimizer_updates_for_regeneration=0,
    )
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    pin = hashlib.sha256(payload).hexdigest()
    delivery = NIGHT / "compact_delivery"
    archive = (
        delivery
        / f"E_JEPA_TTC_SIMPLEX_T_COMPACT_{decision['scientific_saved_updates']:05d}_{pin[:12]}.zip"
    )
    extract = delivery / ("extracted_" + pin[:12])
    compression = measured_zip_bound(files, len(payload))
    peak = compression["peak_bytes"]
    known = [
        archive,
        archive.with_suffix(".zip.pending"),
        archive.with_suffix(".zip.checkpoint"),
        archive.with_name(archive.name + ".checkpoint.next"),
    ]
    existing = sum(path.stat().st_size for path in known if path.is_file())
    if extract.exists():
        existing += sum(path.stat().st_size for path in extract.rglob("*") if path.is_file())
    resources.reserve_artifacts(
        max(0, peak - existing), "COMPACT_ZIP_AND_INDEPENDENT_EXTRACTION_PEAK"
    )
    if not archive.exists():
        namespace["fragmented_zip"](archive, files, manifest, payload, resources)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError("compact archive CRC or duplicate failure")
        if set(z.namelist()) != set(files) | {"CONTENT_MANIFEST.json"}:
            raise ValueError("compact archive scope changed")
        extract.mkdir(exist_ok=True)
        for member in z.infolist():
            resources.check()
            expected = (
                pin
                if member.filename == "CONTENT_MANIFEST.json"
                else manifest["members"][member.filename]["sha256"]
            )
            with z.open(member) as stream:
                namespace["durable_stream"](safe_path(extract, member.filename), stream, expected)
    archive_sha = sha(archive)
    namespace["atomic_bytes"](
        archive.with_suffix(".zip.sha256"), (archive_sha + "  " + archive.name + "\n").encode()
    )
    receipt = dict(
        status="COMPACT_ARCHIVE_INTEGRITY_VERIFIED_REGENERATION_PENDING",
        archive=str(archive),
        sha256=archive_sha,
        bytes=archive.stat().st_size,
        extracted_root=str(extract),
        content_manifest_sha256=pin,
        crc_verified=True,
        extraction_verified=True,
        verified_members=len(files) + 1,
        all_sha256_verified=True,
        head_regeneration="PENDING_AFTER_PUBLISHER_TORCH_PROCESS_EXIT",
        included_numeric_heads=scope["included_numeric_heads"],
        scope=scope,
        h16_separate_regeneration_receipt_sha256=scope["h16_external"]["regeneration_sha256"],
        all_local_full_states_preserved=True,
        all_full_states_included=False,
        termination_reason=decision["termination_reason"],
        optimizer_updates=0,
        measured_compression_reservation=compression,
    )
    namespace["atomic_json"](NIGHT / "FINAL_DELIVERY_COMPACT.json", receipt)
    return receipt


def verify_portable(root: Path, output: Path) -> None:
    """Check sources/weights, included recovery states and all new-head inference."""
    root = root.resolve(strict=True)
    manifest = record(root / "CONTENT_MANIFEST.json")
    for name, row in manifest["members"].items():
        path = safe_path(root, name)
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"portable member changed: {name}")
    scope = record(root / "COMPACT_SCOPE.json")
    local = root / "LOCAL_ENDPOINT_INVENTORY.json"
    if sha(local) != scope["local_full_inventory_sha256"]:
        raise ValueError("preserved local-state inventory differs")
    admitted = admit_exports(root, record(local))
    if sorted(admitted) != scope["included_numeric_heads"]:
        raise ValueError("compact inference head scope differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = {}
    portable_inventory = record(root / "ENDPOINT_INVENTORY.json")
    for label, script in (
        ("partial_checkpoint_verification", "verify_checkpoints.py"),
        ("head_regeneration", "regenerate.py"),
    ):
        if label == "partial_checkpoint_verification" and not any(
            row["checkpoint"] is not None for row in portable_inventory["fits"]
        ):
            steps[label] = dict(
                status="NO_SCIENTIFIC_RECOVERY_STATES_INCLUDED", optimizer_updates=0
            )
            continue
        if label == "head_regeneration" and not admitted:
            steps[label] = dict(status="NO_NUMERIC_HEADS_INCLUDED", optimizer_updates=0)
            continue
        target = output.with_name(output.stem + "_" + label + ".json")
        serial_startup_guard(root)
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                str(root / script),
                "--root",
                str(root),
                "--output",
                str(target),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        target.with_suffix(".log").write_text(child.stdout + child.stderr, encoding="utf-8")
        if child.returncode != 0:
            raise RuntimeError(f"portable {label} failed: {target}")
        receipt = record(target)
        if receipt["optimizer_updates"] != 0:
            raise ValueError("regeneration unexpectedly reports optimizer work")
        if label == "head_regeneration":
            if {row["key"] for row in receipt["heads"]} != set(admitted):
                raise ValueError("regeneration did not check every included head")
        steps[label] = dict(status=receipt["status"], receipt_sha256=sha(target))
    output.write_text(
        json.dumps(
            dict(
                status="COMPACT_PORTABLE_INPUT_WEIGHT_SOURCE_AND_INFERENCE_VERIFIED",
                included_numeric_heads=sorted(admitted),
                head_regeneration=steps["head_regeneration"]["status"],
                verification_steps=steps,
                optimizer_updates=0,
                raw_reconstruction=False,
                h16_separately_verified_reference=scope["h16_external"],
                all_full_states_included=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def publish_only(*, reuse_verified_inventory: bool = False) -> None:
    """Adapt pinned publisher globals inside this process, never edit its files."""
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    from operational.simplex_t_cost_context.engine import EXEC, Resources, _owner

    if any(_owner(path) is not None for path in (H16 / "WRITER.lock", EXEC / "WRITER.lock")):
        raise RuntimeError("compact publication forbidden while a trainer owns its root")
    Resources().check()
    state = runpy.run_module(
        "operational.simplex_t_cost_context.deliver", run_name="compact_publisher"
    )
    namespace = state["main"].__globals__
    original_select, original_report = namespace["selected_files"], namespace["report"]
    if reuse_verified_inventory:
        inventory, evidence = admit_prior_inventory()
        if "torch" in sys.modules:
            raise RuntimeError("prior-inventory publication unexpectedly imported Torch")
        admission_path = NIGHT / "verification/COMPACT_INVENTORY_REUSE_ADMISSION.json"
        if admission_path.exists() and record(admission_path) != evidence:
            raise ValueError("existing inventory-reuse admission differs")
        if not admission_path.exists():
            atomic_stdlib(admission_path, evidence)
        protected = {
            NIGHT / "ENDPOINT_INVENTORY.json": evidence["inventory_sha256"],
            NIGHT / "CHECKPOINT_JOURNAL_RECONCILIATION.json": evidence["reconciliation_sha256"],
        }
        original_atomic = namespace["atomic_json"]

        def preserve_prior_inventory(path: Path, value: dict) -> None:
            if path in protected:
                preserve_admitted_record(path, value, protected[path])
                return
            original_atomic(path, value)

        def reused_inventory(*, admit_physical_checkpoints: bool = False) -> dict:
            if (
                not admit_physical_checkpoints
                or sha(NIGHT / "ENDPOINT_INVENTORY.json") != evidence["inventory_sha256"]
            ):
                raise ValueError("exact prior physical inventory is unavailable")
            return inventory

        namespace["inventory"] = reused_inventory
        namespace["atomic_json"] = preserve_prior_inventory

    def report_compact(inventory: dict) -> dict:
        decision = original_report(inventory)
        decision["delivery_scope"] = "COMPACT_CACHED_INFERENCE_AND_INCLUDED_RECOVERY_STATES"
        decision["all_full_states_included"] = False
        decision["all_local_full_states_preserved"] = True
        decision["h16_payload_scope"] = "SEPARATE_VERIFIED_BUNDLE_REFERENCE"
        decision["delivery_verification_status"] = "PENDING_AFTER_PUBLISHER_PROCESS_EXIT"
        if reuse_verified_inventory:
            decision["physical_inventory_admission"] = evidence
        namespace["atomic_json"](NIGHT / "NEXT_DECISION_NOCTURNA.json", decision)
        path = NIGHT / "INFORME_NOCTURNO_SIMPLEX_T.md"
        text = path.read_text(encoding="utf-8") + (
            "\n## Alcance de la publicación compacta\n\n"
            "Los estados completos de entrenamiento permanecen intactos localmente. "
            "Sólo se omiten del ZIP los endpoints2500 nuevos con pesos NPZ y entradas "
            "cacheadas admitidos contra su sello; los estados parciales se incluyen. "
            "LOCAL_ENDPOINT_INVENTORY.json conserva paths y hashes físicos. "
            "COMPACT_SCOPE.json enumera el alcance exacto. H16 se entrega por referencia "
            "SHA-256 a su bundle independiente ya regenerado; sus cachés/estados no "
            "se duplican aquí. Este ZIP permite inferencia de todas sus cabezas numéricas "
            "incluidas, pero no recuperar el optimizador de los estados omitidos ni "
            "reconstruir desde eventos raw/TRAIN. No se eliminaron artefactos previos.\n"
        )
        namespace["atomic_bytes"](path, text.encode("utf-8"))
        if reuse_verified_inventory:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    "\nEl inventario físico procede de la auditoría Torch de N5 attempt005. "
                    "La publicación compacta comprueba de nuevo los SHA de todos los checkpoints "
                    "y la igualdad de journals, protocolos y sellos; no vuelve a deserializar "
                    "sus estados. La regeneración independiente queda pendiente hasta su recibo.\n"
                )
        return decision

    namespace["selected_files"] = lambda: select_compact(namespace, original_select)
    namespace["report"] = report_compact
    namespace["bundle"] = lambda files, decision: bundle_compact(namespace, files, decision)
    prior_argv = sys.argv
    try:
        sys.argv = ["compact_publisher"]
        state["main"]()
    finally:
        sys.argv = prior_argv
    if reuse_verified_inventory and "torch" in sys.modules:
        raise RuntimeError("prior-inventory publisher unexpectedly imported Torch")


def publish_reusing_verified_inventory() -> None:
    """Publish admitted prior bytes without a new Torch import or verification."""
    coordinator_guard(heavy_child=False)
    with verification_lease(NIGHT / "compact_delivery"):
        publish_only(reuse_verified_inventory=True)


def atomic_stdlib(path: Path, value: dict) -> None:
    """Publish a small own receipt without importing any numeric package."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    with os.fdopen(handle, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def serial_startup_guard(root: Path) -> None:
    """Probe pinned portable limits and reserve measured Torch startup commit."""
    namespace = runpy.run_path(str(root / "regenerate.py"), run_name="portable_resource_probe")
    namespace["resource_guard"](root)
    if os.name != "nt":
        return

    class Performance(ctypes.Structure):
        _fields_ = (
            [("cb", ctypes.c_ulong)]
            + [
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ]
            + [(name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
        )

    value = Performance()
    value.cb = ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value), value.cb):
        raise OSError("Windows committed-memory counters unavailable")
    headroom = (value.CommitLimit - value.CommitTotal) * value.PageSize
    floor = 7 * 1024**3 // 2
    if headroom < floor:
        raise InterruptedError(
            "WINDOWS_COMMIT_BELOW_SERIAL_TORCH_STARTUP_RESERVATION: "
            + json.dumps(dict(available_bytes=headroom, required_bytes=floor))
        )


def validate_new_sidecar(
    archive: Path, extract: Path, output: Path, expected_sha: str | None = None
) -> None:
    """Allow a separate verification receipt, never an archive/member/publication target."""
    archive, extract, output = archive.resolve(), extract.resolve(), output.resolve()
    if (
        output.parent != archive.parent
        or output.is_relative_to(extract)
        or output.suffix != ".json"
        or not output.name.startswith("EXISTING_")
        or not output.name.endswith("_VERIFICATION.json")
        or output
        in (
            archive,
            archive.with_suffix(".zip.sha256"),
            archive.with_suffix(".zip.checkpoint"),
            archive.with_suffix(".publication.json"),
        )
    ):
        raise ValueError(
            "verification output must be a new separate EXISTING_*_VERIFICATION.json sidecar"
        )
    if output.exists():
        prior = record(output)
        if (
            expected_sha is None
            or prior.get("sha256") != expected_sha
            or prior.get("status") != "EXISTING_FROZEN_ARCHIVE_VERIFICATION_COMPLETE"
        ):
            raise ValueError("existing sidecar has a dissimilar receipt; never overwrite it")


@contextmanager
def verification_lease(directory: Path) -> Iterator[None]:
    """OS advisory lock for this delivery root; release automatically on process exit.

    The tiny lease file remains reusable. No prior ZIP, backup or receipt is removed.
    """
    directory.mkdir(exist_ok=True)
    path = directory / "VERIFICATION_WRITER.lock"
    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(handle, "r+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another verifier owns the global delivery lease") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another verifier owns the global delivery lease") from exc
        try:
            stream.seek(0)
            stream.write(
                json.dumps(dict(pid=os.getpid(), status="VERIFYING", optimizer_updates=0)).encode()
            )
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            stream.seek(0)
            stream.write(
                json.dumps(dict(pid=os.getpid(), status="RELEASED", optimizer_updates=0)).encode()
            )
            stream.truncate()
            stream.flush()
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_existing_archive(
    archive: Path, extract: Path, output: Path, expected_sha256: str | None
) -> None:
    """Acquire the sole delivery verifier lease before admitting the precise archive."""
    validate_new_sidecar(archive, extract, output, expected_sha256)
    directory = (NIGHT / "delivery").resolve()
    if not archive.resolve().is_relative_to(directory):
        raise ValueError("existing archive escaped authorized delivery root")
    coordinator_guard(heavy_child=False)
    with verification_lease(directory):
        _verify_existing_archive(archive, extract, output, expected_sha256)


def _verify_existing_archive(
    archive: Path, extract: Path, output: Path, expected_sha256: str | None
) -> None:
    """Finish only this frozen ZIP's extraction, then verify in serial child processes.

    No new ZIP, publisher, numeric export or original final receipt is created.
    Successful atomically published members remain reusable after a resource pause.
    """
    if "torch" in sys.modules:
        raise RuntimeError("existing-archive supervisor must run without Torch")
    delivery = (NIGHT / "delivery").resolve()
    archive, extract, output = archive.resolve(strict=True), extract.resolve(), output.resolve()
    if not all(path.is_relative_to(delivery) for path in (archive, extract, output)):
        raise ValueError("existing verification paths must stay in the authorized delivery root")
    observed = coordinator_guard(heavy_child=False)
    sidecar = archive.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()
    actual_sha = sha(archive)
    if (
        len(sidecar) != 2
        or sidecar[1] != archive.name
        or actual_sha != sidecar[0]
        or (expected_sha256 is not None and actual_sha != expected_sha256)
    ):
        raise ValueError("existing archive SHA differs from its fixed sidecar/request")
    validate_new_sidecar(archive, extract, output, actual_sha)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate existing archive members")
        payload = z.read("CONTENT_MANIFEST.json")
        manifest = json.loads(payload)
        manifest_sha = hashlib.sha256(payload).hexdigest()
        if not archive.name.endswith("_" + manifest_sha[:12] + ".zip"):
            raise ValueError("existing archive filename does not bind its manifest")
        if extract.name != "extracted_" + manifest_sha[:12]:
            raise ValueError("existing extraction root does not bind the same manifest")
        if set(names) != set(manifest["members"]) | {"CONTENT_MANIFEST.json"}:
            raise ValueError("existing archive scope differs from its manifest")
        missing, pending_credit = [], 0
        for member in z.infolist():
            target = safe_path(extract, member.filename)
            row: dict = (
                dict(bytes=len(payload), sha256=manifest_sha)
                if member.filename == "CONTENT_MANIFEST.json"
                else manifest["members"][member.filename]
            )
            if member.file_size != row["bytes"]:
                raise ValueError("existing archive member size differs")
            # Stream every member to EOF: CRC and SHA are checked without extraction.
            h = hashlib.sha256()
            with z.open(member) as stream:
                while block := stream.read(1024 * 1024):
                    h.update(block)
            if h.hexdigest() != row["sha256"]:
                raise ValueError("existing archive member SHA differs")
            if target.exists():
                if target.stat().st_size != row["bytes"] or sha(target) != row["sha256"]:
                    raise ValueError("existing extracted member differs; no overwrite allowed")
            else:
                missing.append((member.filename, row))
                pending = target.with_name(target.name + ".pending")
                if pending.exists():
                    pending_credit += min(pending.stat().st_size, row["bytes"])
        missing_bytes = sum(row["bytes"] for _, row in missing)
        if output.exists():
            if missing or record(output)["admission"]["content_manifest_sha256"] != manifest_sha:
                raise ValueError("existing verification receipt/extraction is inconsistent")
            print(output.read_text(encoding="utf-8"), flush=True)
            return
        growth = missing_bytes - pending_credit
        observed = coordinator_guard(heavy_child=False)
        admission = dict(
            archive_sha256=actual_sha,
            content_manifest_sha256=manifest_sha,
            archive_members=len(names),
            missing_members=len(missing),
            missing_uncompressed_bytes=missing_bytes,
            additional_growth_bytes=growth,
            all_member_sha256_stream_verified=True,
            crc_stream_verified=True,
            resources=observed,
            optimizer_updates=0,
        )
        reasons = []
        if observed["own_artifact_bytes"] + growth + 2 * 1024**2 > 2 * 1024**3:
            reasons.append("OWN_ARTIFACTS_PLUS_REMAINING_EXTRACTION_EXCEED_2_GIB")
        if observed["disk_free_bytes"] - growth - 1024**3 < 10_000_000_000:
            reasons.append("DISK_AFTER_REMAINING_EXTRACTION_AND_1_GIB_RESERVATION_BELOW_10_GB")
        if reasons:
            raise InterruptedError(
                "EXISTING_EXTRACTION_BLOCKED: " + json.dumps(dict(**admission, reasons=reasons))
            )
        probe_payload = z.read("regenerate.py")
        if (
            hashlib.sha256(probe_payload).hexdigest()
            != manifest["members"]["regenerate.py"]["sha256"]
        ):
            raise ValueError("portable resource probe differs from the frozen archive")
        probe_namespace: dict = {
            "__name__": "frozen_portable_resource_probe",
            "__file__": str(extract / "regenerate.py"),
        }
        exec(compile(probe_payload, str(extract / "regenerate.py"), "exec"), probe_namespace)
        resource_probe = probe_namespace["resource_guard"]
        print(
            json.dumps(dict(status="EXISTING_ARCHIVE_STREAM_INTEGRITY_ADMITTED", **admission)),
            flush=True,
        )
        extract.mkdir(exist_ok=True)
        persisted = 0
        for position, (name, row) in enumerate(missing, 1):
            resource_probe(extract)
            if position == 1 or position % 32 == 0:
                coordinator_guard(heavy_child=False)
            target = safe_path(extract, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(target.name + ".pending")
            h = hashlib.sha256()
            with z.open(name) as source, pending.open("wb") as stream:
                while block := source.read(1024 * 1024):
                    h.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if h.hexdigest() != row["sha256"]:
                raise ValueError("stream changed while finishing existing extraction")
            os.replace(pending, target)
            persisted += row["bytes"]
            if position % 32 == 0 or position == len(missing):
                progress = dict(
                    status="EXISTING_EXTRACTION_FRAGMENT_PERSISTED",
                    archive_sha256=actual_sha,
                    confirmed_members=position,
                    remaining_members=len(missing) - position,
                    persisted_bytes=persisted,
                    optimizer_updates=0,
                )
                atomic_stdlib(output.with_name(output.stem + "_EXTRACTION_PROGRESS.json"), progress)
                print(json.dumps(progress), flush=True)
    results = {}
    for label, script in (
        ("checkpoint_verification", "verify_checkpoints.py"),
        ("head_regeneration", "regenerate.py"),
    ):
        if (
            label == "head_regeneration"
            and not (extract / "execution/ANALYSIS_EXPORT_INDEX.json").exists()
        ):
            results[label] = dict(
                status="NO_C0_SCIENTIFIC_PREDICTIONS_PUBLISHED", optimizer_updates=0
            )
            continue
        coordinator_guard(heavy_child=True)
        serial_startup_guard(extract)
        target = output.with_name(output.stem + "_" + label + ".json")
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                str(extract / script),
                "--root",
                str(extract),
                "--output",
                str(target),
            ],
            cwd=extract,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        target.with_suffix(".log").write_text(child.stdout + child.stderr, encoding="utf-8")
        if child.returncode != 0:
            detail = (child.stdout + child.stderr)[-4000:]
            message = (
                f"EXISTING_{label.upper()}_CHILD_FAILED: "
                f"log={target.with_suffix('.log')} detail={detail}"
            )
            if any(
                token in detail
                for token in ("InterruptedError", "PAUSED_RESOURCE", "WinError 1455")
            ):
                raise InterruptedError(message)
            raise RuntimeError(message)
        result = record(target)
        if result["optimizer_updates"] != 0:
            raise ValueError("verification reported optimizer work")
        results[label] = dict(status=result["status"], receipt_sha256=sha(target))
    coordinator_guard(heavy_child=False)
    atomic_stdlib(
        output,
        dict(
            status="EXISTING_FROZEN_ARCHIVE_VERIFICATION_COMPLETE",
            archive=str(archive),
            sha256=actual_sha,
            bytes=archive.stat().st_size,
            extracted_root=str(extract),
            admission=admission,
            verification_steps=results,
            all_member_sha256_verified=True,
            crc_verified=True,
            extraction_verified=True,
            optimizer_updates=0,
            supervisor_torch_imported=False,
            original_archive_and_final_receipts_unchanged=True,
            h16_separate_regeneration_receipt_sha256=sha(extract / "h16/REGENERATION.json")
            if (extract / "h16/REGENERATION.json").exists()
            else None,
        ),
    )
    print(output.read_text(encoding="utf-8"), flush=True)


def coordinator_guard(*, heavy_child: bool) -> dict:
    """Keep own budget/host limits before each serial child, without importing Torch."""
    import psutil

    for path in (H16 / "WRITER.lock", NIGHT / "execution/WRITER.lock"):
        if not path.exists():
            continue
        owner = record(path)
        try:
            process = psutil.Process(owner["pid"])
            if process.create_time() == owner["create_time"]:
                raise RuntimeError("compact publisher must wait for the training owner")
        except psutil.NoSuchProcess:
            pass
    available = psutil.virtual_memory().available
    own_bytes = 0
    for root in (H16, NIGHT):
        stack = [root]
        while stack:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        state = entry.stat(follow_symlinks=False)
                        reparse = bool(getattr(state, "st_file_attributes", 0) & 1024)
                        if entry.is_dir(follow_symlinks=False):
                            if not reparse:
                                stack.append(Path(entry.path))
                        else:
                            own_bytes += state.st_size
                    except FileNotFoundError:
                        continue
    commit = None
    if os.name == "nt":

        class Performance(ctypes.Structure):
            _fields_ = (
                [("cb", ctypes.c_ulong)]
                + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "CommitTotal",
                        "CommitLimit",
                        "CommitPeak",
                        "PhysicalTotal",
                        "PhysicalAvailable",
                        "SystemCache",
                        "KernelTotal",
                        "KernelPaged",
                        "KernelNonpaged",
                        "PageSize",
                    )
                ]
                + [
                    (name, ctypes.c_ulong)
                    for name in ("HandleCount", "ProcessCount", "ThreadCount")
                ]
            )

        value = Performance()
        value.cb = ctypes.sizeof(value)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value), value.cb):
            raise OSError("Windows committed-memory counters unavailable")
        commit = (value.CommitLimit - value.CommitTotal) * value.PageSize
    free = shutil.disk_usage(NIGHT).free
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    failure = []
    if available < 2 * 1024**3:
        failure.append("AVAILABLE_RAM_BELOW_2_GIB")
    if rss > 4 * 1024**3:
        failure.append("COORDINATOR_TREE_RSS_EXCEEDS_4_GIB")
    if own_bytes + 2 * 1024**2 > 2 * 1024**3:
        failure.append("OWN_ARTIFACTS_PLUS_RECEIPTS_EXCEED_2_GIB")
    if free - 1024**3 < 10_000_000_000:
        failure.append("DISK_AFTER_1_GIB_RESERVATION_BELOW_10_GB")
    # Observed startup cost ~2.5 GiB commit, plus the existing 1 GiB floor.
    commit_floor = 7 * 1024**3 // 2 if heavy_child else 1024**3
    if commit is not None and commit < commit_floor:
        failure.append("WINDOWS_COMMIT_BELOW_SERIAL_TORCH_STARTUP_RESERVATION")
    result = dict(
        available_ram_bytes=available,
        tree_rss_bytes=rss,
        own_artifact_bytes=own_bytes,
        disk_free_bytes=free,
        windows_commit_headroom_bytes=commit,
        commit_floor_bytes=commit_floor,
        reasons=failure,
        torch_imported="torch" in sys.modules,
    )
    if failure:
        raise InterruptedError("COMPACT_RESOURCE_BLOCK: " + json.dumps(result))
    return result


def publish() -> None:
    """Serial coordinator: publisher exits before inference verification starts."""
    if "torch" in sys.modules:
        raise RuntimeError("compact coordinator must start in a fresh interpreter without Torch")
    coordinator_guard(heavy_child=True)
    directory = NIGHT / "compact_delivery"
    directory.mkdir(exist_ok=True)
    child = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "operational/simplex_t_io_recovery/run.py"),
            "--script",
            str(Path(__file__).resolve()),
            "--publish-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    (directory / "PUBLISH_ONLY.log").write_text(child.stdout + child.stderr, encoding="utf-8")
    if child.returncode != 0:
        raise RuntimeError("compact archive publication blocked; preserved PUBLISH_ONLY.log")
    finish_verification()


def finish_verification() -> None:
    """Resume verification of the precise existing compact receipt, without republishing."""
    if "torch" in sys.modules:
        raise RuntimeError("verification coordinator must run without Torch")
    receipt_path = NIGHT / "FINAL_DELIVERY_COMPACT.json"
    directory = NIGHT / "compact_delivery"
    receipt = record(receipt_path)
    archive = Path(receipt["archive"])
    if not archive.resolve().is_relative_to((NIGHT / "compact_delivery").resolve()):
        raise ValueError("compact receipt archive escaped own output root")
    if sha(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("compact archive changed before separate verification")
    coordinator_guard(heavy_child=True)
    pin = receipt["content_manifest_sha256"]
    verification = directory / ("VERIFICATION_" + pin[:12] + ".json")
    verify_portable(Path(receipt["extracted_root"]), verification)
    result = record(verification)
    receipt.update(
        status="COMPACT_INFERENCE_DELIVERY_VERIFIED",
        head_regeneration=result["head_regeneration"],
        included_numeric_heads=result["included_numeric_heads"],
        verification_sha256=sha(verification),
        publisher_exited_before_verification=True,
        coordinator_torch_imported=False,
    )
    coordinator_guard(heavy_child=False)
    atomic_stdlib(receipt_path, receipt)
    atomic_stdlib(NIGHT / "FINAL_DELIVERY_NOCTURNA.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def self_test() -> None:
    """Run lightweight standard-library format/path tests, without Torch or optimizer."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        expected = (root / "execution/weights.npz").resolve()
        assert safe_path(root, "execution/weights.npz") == expected
        for name in ("../escape", str(root / "absolute")):
            try:
                safe_path(root, name)
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe path was accepted")
        archive, extract = root / "fixed.zip", root / "extracted_fixed"
        output = root / "EXISTING_FIXED_VERIFICATION.json"
        validate_new_sidecar(archive, extract, output, "a" * 64)
        for collision in (
            archive,
            archive.with_suffix(".zip.sha256"),
            archive.with_suffix(".publication.json"),
            extract / output.name,
        ):
            try:
                validate_new_sidecar(archive, extract, collision, "a" * 64)
            except ValueError:
                pass
            else:
                raise AssertionError("verification output collision was accepted")
        atomic_stdlib(
            output, dict(status="EXISTING_FROZEN_ARCHIVE_VERIFICATION_COMPLETE", sha256="b" * 64)
        )
        try:
            validate_new_sidecar(archive, extract, output, "a" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("dissimilar previous verification receipt would be overwritten")
        with verification_lease(root):
            try:
                with verification_lease(root):
                    raise AssertionError("duplicate verification writer was admitted")
            except RuntimeError:
                pass
        header = (repr(dict(descr="<f4", fortran_order=False, shape=(1,))) + "\n").encode()
        data = (
            b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<f", 1.0)
        )
        path = root / "weights.npz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("weight.npy", data)
        assert npz_headers(path)["weight"]["shape"] == (1,)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../unsafe.npy", data)
        try:
            npz_headers(path)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe NPZ member was accepted")
        # Exercise complete source/endpoint binding with synthetic metadata only.
        execution = root / "execution"
        execution.mkdir()
        protocol: dict = dict(fits=[], sources={})
        physical, endpoints, exports = [], [], {}
        for fold in range(3):
            key = f"SYNTHETIC/FULL_C0/fold{fold}/seed7"
            protocol["fits"].append(dict(key=key, arm="FULL_C0", fold=fold, seed=7))
            protocol["sources"][str(fold)] = dict(
                wrapped=dict(FULL_C0=dict(train_sha256="1" * 64, dev_sha256="2" * 64))
            )
            physical.append(dict(id=key, saved_updates=2500, checkpoint_sha256="3" * 64))
            endpoints.append(
                dict(key=key, family="N2", completed_updates=2500, checkpoint_sha256="3" * 64)
            )
            numeric = execution / f"weight{fold}.npz"
            with zipfile.ZipFile(numeric, "w") as z:
                z.writestr("weight.npy", data)
            prediction, publication = (
                execution / f"prediction{fold}",
                execution / f"publication{fold}",
            )
            prediction.write_bytes(b"synthetic metadata only")
            publication.write_text("{}", encoding="utf-8")
            exports[key] = dict(
                arm="FULL_C0",
                fold=fold,
                seed=7,
                checkpoint_sha256="3" * 64,
                train_source_sha256="1" * 64,
                dev_source_sha256="2" * 64,
                weights_path=numeric.name,
                weights_sha256=sha(numeric),
                predictions_path=prediction.name,
                predictions_sha256=sha(prediction),
                publication_path=publication.name,
                publication_sha256=sha(publication),
                fragments=[],
            )
        atomic_stdlib(root / "PROTOCOL_COST_CONTEXT.json", protocol)
        pin = sha(root / "PROTOCOL_COST_CONTEXT.json")
        for fold, export in enumerate(exports.values()):
            receipt_path = execution / f"fragment{fold}.json"
            fragment = dict(
                start=0,
                stop=1,
                input_path=export["weights_path"],
                input_sha256=export["weights_sha256"],
            )
            atomic_stdlib(
                receipt_path,
                dict(
                    **fragment,
                    protocol_sha256=pin,
                    source_sha256=export["dev_source_sha256"],
                    endpoint_sha256=export["checkpoint_sha256"],
                ),
            )
            export["fragments"] = [
                dict(**fragment, receipt_path=receipt_path.name, receipt_sha256=sha(receipt_path))
            ]
        seal_path = execution / "N2_ENDPOINTS.json"
        atomic_stdlib(
            seal_path,
            dict(protocol_sha256=pin, fits=endpoints, all_family_frozen_before_evaluation=True),
        )
        index = dict(
            protocol_sha256=pin,
            fits=exports,
            families=dict(N2=dict(seal_path=str(seal_path), seal_sha256=sha(seal_path))),
        )
        atomic_stdlib(execution / "ANALYSIS_EXPORT_INDEX.json", index)
        assert len(admit_exports(root, dict(fits=physical))) == 3
        exports[next(iter(exports))]["train_source_sha256"] = "4" * 64
        atomic_stdlib(execution / "ANALYSIS_EXPORT_INDEX.json", index)
        try:
            admit_exports(root, dict(fits=physical))
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched scientific source binding was accepted")
        canonical = execution / "canonical.parquet"
        csv = canonical.with_suffix(".csv")
        batch = execution / "batch.npz"
        partial = execution / "scientific_partial.pt"
        canonical.write_bytes(b"hash fixture only; no scientific predictions")
        csv.write_bytes(b"duplicate query CSV fixture")
        batch.write_bytes(b"regenerable array fixture")
        partial.write_bytes(b"scientific recovery state fixture")
        pub = execution / "canonical_publication.json"
        fragment_receipt = execution / "batch_receipt.json"
        atomic_stdlib(pub, dict(csv_sha256=sha(csv)))
        atomic_stdlib(fragment_receipt, dict(payload_sha256=sha(batch)))
        duplicate_exports = dict(
            fixture=dict(
                predictions_path=canonical.name,
                predictions_sha256=sha(canonical),
                publication_path=pub.name,
                publication_sha256=sha(pub),
                fragments=[
                    dict(
                        receipt_path=fragment_receipt.name,
                        receipt_sha256=sha(fragment_receipt),
                        output_path=batch.name,
                        output_sha256=sha(batch),
                    )
                ],
            )
        )
        selection = {
            "execution/" + p.name: p for p in (canonical, csv, batch, partial, fragment_receipt)
        }
        bad_selection = dict(selection)
        duplicate_exports["fixture"]["predictions_sha256"] = "0" * 64
        try:
            omit_redundant_exports(bad_selection, duplicate_exports, execution)
        except ValueError:
            assert bad_selection == selection
        else:
            raise AssertionError("invalid canonical Parquet allowed duplicate omission")
        duplicate_exports["fixture"]["predictions_sha256"] = sha(canonical)
        omissions = omit_redundant_exports(selection, duplicate_exports, execution)
        assert len(omissions) == 2
        assert set(selection) == {
            "execution/" + p.name for p in (canonical, partial, fragment_receipt)
        }
        assert csv.exists() and batch.exists() and partial.exists()
        manifest_bytes = b'{"fixture_only":true}'
        measured = measured_zip_bound(selection, len(manifest_bytes))
        actual_zip = root / "synthetic_compression.zip"
        with zipfile.ZipFile(actual_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, member in selection.items():
                archive.write(member, name)
            archive.writestr("CONTENT_MANIFEST.json", manifest_bytes)
        assert actual_zip.stat().st_size <= measured["zip_upper_bytes"]
        assert (
            measured["peak_bytes"]
            == max(
                3 * measured["zip_upper_bytes"],
                2 * measured["zip_upper_bytes"] + measured["uncompressed_bytes"],
            )
            + 2 * 1024**2
        )
        assert measured["extra_zip_created"] is False
        # Synthetic bytes exercise archival deduplication, never statistics.
        bootstrap_root = execution / "analysis/N2/bootstrap"
        control = bootstrap_root / ".resume"
        control.mkdir(parents=True)
        canonical_draws = bootstrap_root / "HIERARCHICAL_DRAWS.jsonl"
        canonical_draws.write_bytes(b"first synthetic draw\nsecond synthetic draw\n")
        (bootstrap_root / "BOOTSTRAP_LOSSES.npy").write_bytes(b"synthetic array fixture")
        inputs_path = control / "INPUTS.json"
        atomic_stdlib(
            inputs_path,
            dict(
                binding=dict(family="N2", protocol_sha256=pin),
                draw_file_sha256=sha(canonical_draws),
            ),
        )
        for offset, draw in enumerate(canonical_draws.read_bytes().splitlines(keepends=True)):
            fragment_root = control / f"fragment_{offset:05d}_{offset + 1:05d}"
            fragment_root.mkdir()
            (fragment_root / "DRAWS.jsonl").write_bytes(draw)
            (fragment_root / "LOSSES.npy").write_bytes(b"synthetic fragment array")
            atomic_stdlib(
                fragment_root / "RECEIPT.json",
                dict(
                    start=offset,
                    stop=offset + 1,
                    accepted_ids=[offset],
                    rejected_ids=[],
                    input_sha256=sha(inputs_path),
                    files={
                        name: sha(fragment_root / name) for name in ("DRAWS.jsonl", "LOSSES.npy")
                    },
                ),
            )
        complete = dict(attempts=2, valid_draws=2)
        atomic_stdlib(control / "COMPLETE.json", complete)
        analysis_root = bootstrap_root.parent
        result_path = analysis_root / "RESULTS.json"
        atomic_stdlib(
            result_path,
            dict(
                status="COMPLETE_EXPLORATORY_FAMILY_ANALYSIS",
                family="N2",
                protocol_sha256=pin,
                bootstrap=complete,
            ),
        )
        atomic_stdlib(
            analysis_root / "ANALYSIS_RECEIPT.json",
            dict(
                status="COMPLETE_EXPLORATORY_FAMILY_ANALYSIS",
                results_sha256=sha(result_path),
                accepted_draws=2,
                draw_attempts=2,
            ),
        )
        draw_selection = {
            "execution/" + p.relative_to(execution).as_posix(): p
            for p in analysis_root.rglob("*")
            if p.is_file()
        }
        incomplete_selection = dict(draw_selection)
        incomplete_selection.pop("execution/analysis/N2/bootstrap/.resume/COMPLETE.json")
        unchanged = dict(incomplete_selection)
        assert omit_completed_draw_copies(incomplete_selection, execution) == []
        assert incomplete_selection == unchanged
        altered = control / "fragment_00000_00001/DRAWS.jsonl"
        original = altered.read_bytes()
        altered.write_bytes(b"corrupted duplicate draw\n")
        invalid_selection = dict(draw_selection)
        try:
            omit_completed_draw_copies(invalid_selection, execution)
        except ValueError:
            assert invalid_selection == draw_selection
        else:
            raise AssertionError("different fragment concatenation was accepted")
        altered.write_bytes(original)
        draw_omissions = omit_completed_draw_copies(draw_selection, execution)
        assert len(draw_omissions) == 2
        assert sum(row["bytes"] for row in draw_omissions) == canonical_draws.stat().st_size
        assert all(row["concatenation_sha256"] == sha(canonical_draws) for row in draw_omissions)
        assert canonical_draws.exists() and altered.exists()
        assert all(not name.endswith("/DRAWS.jsonl") for name in draw_selection)
        assert any(name.endswith("/RECEIPT.json") for name in draw_selection)
        admitted_record = root / "admitted_inventory.json"
        admitted_value = dict(scientific_saved_updates=2132, fixture_only=True)
        atomic_stdlib(admitted_record, admitted_value)
        admitted_pin = sha(admitted_record)
        admitted_mtime = admitted_record.stat().st_mtime_ns
        preserve_admitted_record(admitted_record, admitted_value, admitted_pin)
        assert admitted_record.stat().st_mtime_ns == admitted_mtime
        for changed_value, changed_pin in (
            (dict(scientific_saved_updates=2500), admitted_pin),
            (admitted_value, "0" * 64),
        ):
            try:
                preserve_admitted_record(admitted_record, changed_value, changed_pin)
            except ValueError:
                assert sha(admitted_record) == admitted_pin
                assert admitted_record.stat().st_mtime_ns == admitted_mtime
            else:
                raise AssertionError("changed inventory replacement was accepted")
    assert "torch" not in sys.modules
    print(
        json.dumps(
            dict(status="STDLIB_COMPACT_TESTS_PASSED", optimizer_updates=0, torch_imported=False)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--publish", action="store_true")
    action.add_argument("--publish-reusing-verified-inventory", action="store_true")
    action.add_argument(
        "--finish-verification",
        action="store_true",
        help="Verify the exact existing compact archive without republishing",
    )
    action.add_argument(
        "--publish-only",
        action="store_true",
        help="Internal serial archive stage; verification remains pending",
    )
    action.add_argument("--verify-root", type=Path)
    action.add_argument("--verify-existing-archive", type=Path)
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extract-root", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.verify_existing_archive:
        if args.output is None or args.extract_root is None:
            parser.error("--verify-existing-archive requires --extract-root and --output")
        verify_existing_archive(
            args.verify_existing_archive, args.extract_root, args.output, args.expected_sha256
        )
    elif args.publish_reusing_verified_inventory:
        publish_reusing_verified_inventory()
    elif args.self_test:
        self_test()
    elif args.verify_root:
        if args.output is None:
            parser.error("--verify-root requires --output")
        verify_portable(args.verify_root, args.output)
    elif args.publish_only:
        publish_only()
    elif args.finish_verification:
        finish_verification()
    else:
        publish()


if __name__ == "__main__":
    try:
        main()
    except InterruptedError as exc:
        print(
            json.dumps(
                dict(
                    status="BLOCKED_RESOURCE_RECOVERABLE",
                    dependency=str(exc),
                    optimizer_updates=0,
                    no_automatic_retry=True,
                )
            ),
            flush=True,
        )
        raise SystemExit(3) from None

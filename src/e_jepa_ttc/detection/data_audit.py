"""Reproducible eAP metadata audit and bounded, pinned label recovery.

No dataset discovery, unbounded download, test-label reading, or geometry guessing
is performed. ``dataset_root=None`` means unavailable, not missing on the user PC.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from e_jepa_ttc.detection import geometry_contract as geometry_module
from e_jepa_ttc.detection.geometry_contract import geometry_blockers, validate_calibration

EAP_REPOSITORY = "NAIL-HNU/eAP-dataset"
EAP_REVISION = "a814a946e60d37f7242358af5f344542177e50c5"
EAP_CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)
DETECTION_FIELDS = (
    "class_name",
    "translation_m",
    "size_lwh_m",
    "yaw_rad",
    "velocity_mps",
    "score",
)
MISSING_TEST_RGB_REPORTED = (
    "3RUZHwnCwK/rgb_shards/rgb-00003.tar",
    "71wUSQYMpk/rgb_shards/rgb-00001.tar",
    "71wUSQYMpk/rgb_shards/rgb-00003.tar",
    "89fvtHGHx2/rgb_shards/rgb-00001.tar",
    "89fvtHGHx2/rgb_shards/rgb-00003.tar",
    "HHJe79eY2J/rgb_shards/rgb-00003.tar",
    "Mu6kaUjIz3/rgb_shards/rgb-00001.tar",
    "kafYzX6kJ4/rgb_shards/rgb-00003.tar",
    "vhjB5OViA7/rgb_shards/rgb-00003.tar",
)


def sha256_file(path: Path) -> str:
    """Hash a local file in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> str:
    """Reject absolute paths, traversal and platform-dependent separators."""

    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe dataset path: {value!r}")
    if str(path) != value:
        raise ValueError(f"Non-canonical dataset path: {value!r}")
    return value


def _rooted_path(root: Path, relative: str) -> Path:
    """Keep write targets inside the selected root, including existing symlinks."""

    candidate = root / safe_relative_path(relative)
    if not candidate.resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise ValueError(f"Dataset path resolves outside selected root: {relative!r}")
    return candidate


def read_pinned_manifest(path: Path) -> dict[str, Any]:
    """Validate the exact approved public revision and every file's metadata."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("repository") != EAP_REPOSITORY or manifest.get("revision") != EAP_REVISION:
        raise ValueError("The G0 manifest must use the reviewed repository and exact HF revision.")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Pinned manifest has no file inventory.")
    for name, record in manifest["files"].items():
        safe_relative_path(name)
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Invalid size for {name}")
        digest = record.get("sha256")
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for {name}")
    return manifest


def recovery_plan(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Plan only manifests and train labels; missing test TARs are metadata only."""

    selected = []
    for name, record in sorted(manifest["files"].items()):
        if name in {"data/train.parquet", "data/test.parquet"} or re.fullmatch(
            r"data/train/[A-Za-z0-9]+/labels\.parquet", name
        ):
            target = _rooted_path(root, name)
            if target.is_file():
                valid = target.stat().st_size == record["size_bytes"]
                valid = valid and sha256_file(target) == record["sha256"]
                state = "verified_present" if valid else "conflict_preserve_existing"
            else:
                state = "missing_in_supplied_root"
            selected.append({"path": name, **record, "state": state})
    return {
        "repository": EAP_REPOSITORY,
        "revision": EAP_REVISION,
        "files": selected,
        "root": str(root),
        "download_bytes": sum(
            r["size_bytes"] for r in selected if r["state"].startswith("missing")
        ),
        "test_media_downloaded": False,
    }


def download_recovery_plan(plan: Mapping[str, Any], root: Path) -> list[dict[str, object]]:
    """Fetch the bounded reviewed allowlist; refuse to replace existing bad files."""

    if plan.get("repository") != EAP_REPOSITORY or plan.get("revision") != EAP_REVISION:
        raise ValueError("Unreviewed download repository or revision.")
    files = plan["files"]
    if sum(item["size_bytes"] for item in files) > 140_000_000:
        raise ValueError("Recovery plan exceeds the 140 MB metadata/label ceiling.")
    for item in files:
        name = safe_relative_path(item["path"])
        if name not in {"data/train.parquet", "data/test.parquet"} and not re.fullmatch(
            r"data/train/[A-Za-z0-9]+/labels\.parquet", name
        ):
            raise ValueError(f"Download not allowlisted: {name}")
        if item["state"] == "conflict_preserve_existing":
            raise ValueError(f"Existing file differs; preserve it and investigate: {name}")
    receipts = []
    for item in files:
        name = item["path"]
        destination = _rooted_path(root, name)
        if destination.exists():
            if destination.stat().st_size != item["size_bytes"] or (
                sha256_file(destination) != item["sha256"]
            ):
                raise ValueError(f"Existing file differs; no overwrite: {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _rooted_path(root, name + ".g0-part")
            url = f"https://huggingface.co/datasets/{EAP_REPOSITORY}/resolve/{EAP_REVISION}/{name}"
            written = 0
            created_temporary = False
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    temporary.open("xb") as out,
                ):
                    created_temporary = True
                    while chunk := response.read(1024 * 1024):
                        written += len(chunk)
                        if written > item["size_bytes"]:
                            raise ValueError(f"Download larger than pinned size: {name}")
                        out.write(chunk)
                if written != item["size_bytes"] or sha256_file(temporary) != item["sha256"]:
                    raise ValueError(f"Downloaded bytes do not match pinned metadata: {name}")
                if destination.exists():
                    raise ValueError(f"Destination appeared during download: {name}")
                if os.name == "nt":
                    # Windows rename rejects an existing destination, including races,
                    # and works on exFAT, where hard links are unavailable.
                    os.rename(temporary, destination)
                else:
                    # POSIX rename would replace an existing file; keep no-replace link.
                    os.link(temporary, destination)
                    temporary.unlink()
            except BaseException:
                if created_temporary:
                    temporary.unlink(missing_ok=True)
                raise
        receipts.append(
            {"path": name, "sha256": sha256_file(destination), "bytes": item["size_bytes"]}
        )
    return receipts


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Parquet auditing requires pyarrow in an isolated environment."
        ) from error
    return parquet.read_table(path).to_pylist()


def validate_media_rows(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    """Validate keys, exposure clocks and matrix structure in a public manifest."""

    if split not in {"train", "test"} or not rows:
        raise ValueError("A non-empty train/test media table is required.")
    tokens: set[str] = set()
    times: dict[str, int] = {}
    paths: dict[str, set[str]] = {"events": set(), "rgb": set(), "labels": set()}
    calibration_seen: set[bytes] = set()
    counts: Counter[str] = Counter()
    exposure = []
    for row in rows:
        token, sequence = str(row["sample_token"]), str(row["sequence_id"])
        if row["split"] != split or not token.startswith(sequence + ":") or token in tokens:
            raise ValueError(f"Duplicate token, token/sequence mismatch or wrong split: {token}")
        tokens.add(token)
        counts[sequence] += 1
        start, end = row["rgb_exposure_start_timestamp_us"], row["rgb_exposure_end_timestamp_us"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
            raise ValueError(f"Non-integer exposure timestamp for {token}")
        if start < 0 or start > end or start <= times.get(sequence, -1):
            raise ValueError(f"Invalid or non-increasing exposure time for {token}")
        times[sequence] = start
        exposure.append(end - start)
        expected_prefix = f"data/{split}/{sequence}/"
        for column, kind in (("events_path", "events"), ("rgb_shard_path", "rgb")):
            path = safe_relative_path(str(row[column]))
            if not path.startswith(expected_prefix):
                raise ValueError(f"Path escapes its split/sequence: {path}")
            paths[kind].add(path)
        member = safe_relative_path(str(row["rgb_member_path"]))
        if not member.startswith("rgb/") or not member.endswith(".png"):
            raise ValueError(f"Unexpected RGB member: {member}")
        label = row["labels_path"]
        if split == "test" and label is not None:
            raise ValueError("Public test must have null labels_path; labels are not admissible.")
        if split == "train":
            if label != expected_prefix + "labels.parquet":
                raise ValueError(f"Missing or cross-sequence train label path: {token}")
            paths["labels"].add(label)
        calibration_key = json.dumps([row["K_event"], row["T_event_ego"]]).encode()
        if calibration_key not in calibration_seen:
            validate_calibration(row["K_event"], row["T_event_ego"])
            calibration_seen.add(calibration_key)
    return {
        "rows": len(rows),
        "unique_tokens": len(tokens),
        "sequence_count": len(counts),
        "sequence_frames": dict(sorted(counts.items())),
        "paths": {key: sorted(value) for key, value in paths.items()},
        "exposure_duration_us": {"min": min(exposure), "max": max(exposure)},
        "unique_calibrations": len(calibration_seen),
        "structural_check": "PASS",
        "rectification_verified": False,
        "physical_availability_verified": False,
    }


def validate_label_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sequence: str,
    media_tokens: set[str],
) -> dict[str, Any]:
    """Check label consistency without assigning physical meaning to permutations."""

    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    tracks: set[str] = set()
    tests: Counter[str] = Counter()
    for row in rows:
        token, track = str(row["sample_token"]), str(row["track_id"])
        if (
            row["sequence_id"] != sequence
            or row.get("split") != "train"
            or token not in media_tokens
            or not token.startswith(sequence + ":")
        ):
            failures["foreign_sequence_split_or_token"] += 1
        if (token, track) in seen:
            failures["duplicate_sample_track"] += 1
        seen.add((token, track))
        tracks.add(track)
        category = str(row["category"])
        counts[category] += 1
        if category not in EAP_CLASSES:
            failures["unknown_class"] += 1
        box = np.asarray(row["bbox_3d_ego"], dtype=np.float64)
        if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any():
            failures["invalid_box"] += 1
            continue
        for key, value in (
            ("size", box[[5, 3, 4]]),
            ("translation", box[:3]),
            ("ego_translation", box[:3]),
            ("yaw", box[6]),
        ):
            observed = np.asarray(row.get(key), dtype=np.float64)
            if observed.shape == np.asarray(value).shape and np.array_equal(observed, value):
                tests[key + "_matches_bbox_relation"] += 1
            else:
                failures[key + "_relation_mismatch"] += 1
        rotation = np.asarray(row.get("rotation"), dtype=np.float64)
        expected_rotation = np.asarray([np.cos(box[6] / 2), 0, 0, np.sin(box[6] / 2)])
        if rotation.shape == (4,) and np.allclose(rotation, expected_rotation, atol=1e-12, rtol=0):
            tests["rotation_numeric_matches_cos0_0_sin_half_yaw"] += 1
        else:
            failures["rotation_numeric_relation_mismatch"] += 1
        velocity = np.asarray(row.get("velocity"), dtype=np.float64)
        velocity3 = np.asarray(row.get("velocity_3d"), dtype=np.float64)
        if velocity.shape != (2,) or velocity3.shape != (3,) or not np.isfinite(velocity3).all():
            failures["invalid_velocity"] += 1
        elif np.array_equal(velocity, velocity3[:2]):
            tests["velocity_matches_first_two_velocity_3d"] += 1
            if velocity3[2] == 0:
                tests["velocity_3d_z_zero"] += 1
        else:
            failures["velocity_relation_mismatch"] += 1
    return {
        "sequence_id": sequence,
        "rows": len(rows),
        "tracks": len(tracks),
        "class_counts": dict(sorted(counts.items())),
        "numeric_relations": dict(tests),
        "failures": dict(failures),
        "structural_check": "PASS" if not failures else "FAIL",
        "relations_are_not_a_verified_geometry_conversion": True,
    }


def audit_sample_submission(path: Path, expected_tokens: set[str]) -> dict[str, object]:
    """Check the supplied template; empty detections do not validate box semantics."""

    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("Expected only submission.json in the supplied ZIP.")
        raw = archive.read("submission.json")
    sample = json.loads(raw)
    if set(sample) != {"meta", "results"} or set(sample["results"]) != expected_tokens:
        raise ValueError("Submission template keys/tokens differ from pinned test manifest.")
    if sample["meta"].get("classes") != list(EAP_CLASSES):
        raise ValueError("Submission class order differs from reviewed eAP template.")
    if sample["meta"].get("fields") != list(DETECTION_FIELDS):
        raise ValueError("Submission fields differ from reviewed eAP template.")
    if not all(value == [] for value in sample["results"].values()):
        raise ValueError("This audit expects an empty sample template, not predictions.")
    return {
        "zip_sha256": sha256_file(path),
        "submission_sha256": hashlib.sha256(raw).hexdigest(),
        "tokens": len(expected_tokens),
        "all_empty": True,
        "meta": sample["meta"],
        "check": "PASS",
        "per_object_conventions_verified_by_empty_template": False,
    }


def audit_eap(
    *,
    metadata_root: Path,
    manifest_path: Path,
    contract_path: Path,
    dataset_root: Path | None = None,
    sample_submission: Path | None = None,
    verify_present: bool = False,
    verify_tar_members: bool = False,
) -> dict[str, Any]:
    """Audit public train and structural test metadata while keeping G0 fail-closed."""

    manifest = read_pinned_manifest(manifest_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("dataset_revision") != EAP_REVISION:
        raise ValueError("Geometry contract is not bound to the inspected HF revision.")
    root_available = dataset_root is not None and dataset_root.is_dir()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    output: dict[str, Any] = {
        "schema_version": "eap_detection_g0_audit_v1",
        "revision": EAP_REVISION,
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": sha256_file(contract_path),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "producer_source_sha256": {
            "data_audit.py": sha256_file(Path(__file__)),
            "geometry_contract.py": sha256_file(Path(geometry_module.__file__)),
        },
        "dataset_root_status": "SUPPLIED_DIRECTORY_EXISTS"
        if root_available
        else "NOT_ACCESSIBLE_IN_THIS_SESSION",
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "metadata_root": str(metadata_root),
        "splits": {},
        "labels": [],
        "media_hash_check_requested": verify_present,
        "tar_member_check_requested": verify_tar_members,
        "media_hashes_verified": False,
        "tar_members_verified": False,
        "blockers": geometry_blockers(contract, evidence_root=contract_path.parent),
    }
    media: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        path = metadata_root / "data" / f"{split}.parquet"
        expected = manifest["files"][f"data/{split}.parquet"]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Manifest Parquet hash differs from pinned release: {path}")
        media[split] = _read_parquet(path)
        audit = validate_media_rows(media[split], split)
        audit["parquet_sha256"] = expected["sha256"]
        file_states: dict[str, dict[str, object]] = {}
        for kind, paths in audit["paths"].items():
            for name in paths:
                if name not in manifest["files"]:
                    raise ValueError(f"Reference absent from pinned public inventory: {name}")
            if not root_available and kind != "labels":
                file_states[kind] = {"referenced": len(paths), "local_status": "NOT_INSPECTED"}
                continue
            file_root = metadata_root if kind == "labels" else dataset_root
            assert file_root is not None
            present, missing, bad = [], [], []
            for name in paths:
                file = file_root / name
                if not file.is_file():
                    missing.append(name)
                elif file.stat().st_size != manifest["files"][name]["size_bytes"]:
                    bad.append(name)
                elif verify_present and sha256_file(file) != manifest["files"][name]["sha256"]:
                    bad.append(name)
                else:
                    present.append(name)
            file_states[kind] = {
                "referenced": len(paths),
                "present": len(present),
                "missing": missing,
                "invalid": bad,
                "verification_root": str(file_root),
            }
            if split == "train" and (missing or bad):
                output["blockers"].append(f"train {kind} unavailable or invalid")
        audit["file_states"] = file_states
        output["splits"][split] = audit
    if not root_available:
        output["blockers"].append("workstation_media_not_accessible")
    if not verify_present or not verify_tar_members:
        output["blockers"].append("media_hash_or_member_audit_not_run")
    train_media_complete = root_available and all(
        state.get("present") == state["referenced"]
        for state in output["splits"]["train"]["file_states"].values()
    )
    output["media_hashes_verified"] = verify_present and train_media_complete
    missing_tar_members = False
    if root_available and dataset_root is not None and verify_tar_members:
        members_by_tar: dict[str, set[str]] = {}
        for row in media["train"]:
            members_by_tar.setdefault(row["rgb_shard_path"], set()).add(row["rgb_member_path"])
        for name, required in members_by_tar.items():
            path = dataset_root / name
            if path.is_file():
                with tarfile.open(path, "r") as archive:
                    actual = {entry.name for entry in archive if entry.isfile()}
                if not required.issubset(actual):
                    missing_tar_members = True
                    output["blockers"].append(f"missing RGB members: {name}")
    output["tar_members_verified"] = (
        verify_tar_members and train_media_complete and not missing_tar_members
    )
    token_sets = {split: {row["sample_token"] for row in rows} for split, rows in media.items()}
    if token_sets["train"] & token_sets["test"]:
        raise ValueError("Train/test token overlap.")
    sequence_sets = {s: set(output["splits"][s]["sequence_frames"]) for s in ("train", "test")}
    if sequence_sets["train"] & sequence_sets["test"]:
        raise ValueError("Train/test sequence overlap.")
    for sequence in sorted(sequence_sets["train"]):
        name = f"data/train/{sequence}/labels.parquet"
        path = metadata_root / name
        if not path.is_file():
            output["blockers"].append(f"label bytes not inspected: {sequence}")
            continue
        expected = manifest["files"][name]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Label hash differs from pinned release: {name}")
        label_audit = validate_label_rows(
            _read_parquet(path), sequence=sequence, media_tokens=token_sets["train"]
        )
        label_audit["sha256"] = expected["sha256"]
        output["labels"].append(label_audit)
        if label_audit["failures"]:
            output["blockers"].append(f"label structural failure: {sequence}")
    if sample_submission is not None:
        output["sample_submission"] = audit_sample_submission(sample_submission, token_sets["test"])
    absent_report = {"data/test/" + path for path in MISSING_TEST_RGB_REPORTED}
    affected = [r for r in media["test"] if r["rgb_shard_path"] in absent_report]
    observed_states: dict[str, str] = {}
    for name in sorted(absent_report):
        if not root_available or dataset_root is None:
            observed_states[name] = "NOT_INSPECTED"
        elif not (dataset_root / name).is_file():
            observed_states[name] = "MISSING_IN_SUPPLIED_ROOT"
        elif (dataset_root / name).stat().st_size != manifest["files"][name]["size_bytes"]:
            observed_states[name] = "INVALID_SIZE_IN_SUPPLIED_ROOT"
        else:
            observed_states[name] = "PRESENT_SIZE_MATCHES_HASH_NOT_CHECKED"
            if verify_present:
                observed_states[name] = (
                    "PRESENT_HASH_VERIFIED"
                    if sha256_file(dataset_root / name) == manifest["files"][name]["sha256"]
                    else "INVALID_HASH_IN_SUPPLIED_ROOT"
                )
    output["prior_report_missing_test_rgb"] = {
        "paths": sorted(absent_report),
        "referenced_frames": len(affected),
        "sequence_count": len({r["sequence_id"] for r in affected}),
        "bytes": sum(manifest["files"][name]["size_bytes"] for name in absent_report),
        "observed_states": observed_states,
        "local_absence_revalidated": all(
            value == "MISSING_IN_SUPPLIED_ROOT" for value in observed_states.values()
        ),
        "note": (
            "Scope/counts revalidated against manifests; previous workstation absence not assumed."
        ),
    }
    output["train_label_files"] = len(output["labels"])
    output["train_label_rows"] = sum(row["rows"] for row in output["labels"])
    totals: Counter[str] = Counter()
    for row in output["labels"]:
        totals.update(row["class_counts"])
    output["train_class_counts"] = dict(sorted(totals.items()))
    output["status"] = "PASS" if not output["blockers"] else "BLOCKED"
    output["no_training_performed"] = True
    return output

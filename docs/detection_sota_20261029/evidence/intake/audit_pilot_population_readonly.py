"""Reproduce the PHASE-TTC pilot population diagnosis from public train metadata.

Dependencies used: CPython 3.12.14, numpy 2.5.4, pyarrow 26.0.0. No Torch, GPU,
image, event file, network, training or data/test.parquet access is required.
The geometric eligibility calculation assumes the documented 1280x720 images
and reproduces the pilot's 640x384 pinhole target filters; it does not validate
pixels, camera calibration, acquisition independence or the official scorer.
Distinct IDs mean (sequence_id, track_id), not verified independent objects.

Example (all three roots are explicit):
  python audit_pilot_population_readonly.py --repo /path/to/phase-ttc \
    --metadata-root /path/to/eap_metadata --output /tmp/population_reproduced.json
The metadata root must contain data/train.parquet and the 40 public train labels.
The output must not already exist. All inputs are opened read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq

BASE = "24b8f36f6e4aa17c965f9526d8cf486b512a8046"
REVISION = "a814a946e60d37f7242358af5f344542177e50c5"
RARE = {"bicycle", "other_vehicle"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filter_reason(box: list, row: dict, corners_fn) -> str:
    """Exact pilot filter formulas, including pixel-center resize and rule order."""
    intrinsic = np.asarray(row["K_event"], dtype=float)
    transform = np.asarray(row["T_event_ego"], dtype=float)
    resize = np.array([[0.5, 0, -0.25], [0, 384 / 720, (384 / 720 - 1) / 2], [0, 0, 1]])
    intrinsic = resize @ intrinsic
    rotation = transform[:3, :3]
    camera = corners_fn(box) @ rotation.T + transform[:3, 3]
    center = rotation @ np.asarray(box, dtype=float)[:3] + transform[:3, 3]
    if not 0.1 <= center[2] < 60:
        return "center_depth_outside_profile"
    if (camera[:, 2] <= 0.1).any():
        return "corner_behind_or_near_camera"
    projected = camera @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    uv_h = intrinsic @ center
    uv = uv_h[:2] / uv_h[2]
    if not (0 <= uv[0] < 640 and 0 <= uv[1] < 384):
        return "projected_center_outside_image"
    minimum = np.maximum(pixels.min(0), [0, 0])
    maximum = np.minimum(pixels.max(0), [639, 383])
    if min(maximum - minimum) <= 1:
        return "projected_box_too_small"
    distances = np.array(
        [uv[0] - minimum[0], maximum[0] - uv[0], uv[1] - minimum[1], maximum[1] - uv[1]]
    )
    if (distances < 0).any():
        return "projected_center_outside_clipped_box"
    return "admitted"


def main() -> None:
    # Import the pinned geometry helpers without writing __pycache__ to the repo.
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, metadata = args.repo.resolve(), args.metadata_root.resolve()
    if args.output.exists():
        parser.error("Output already exists; use a new path")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != BASE:
        parser.error(f"Expected inspected commit {BASE}; observed {commit}")
    relative_sources = [
        "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
        "src/e_jepa_ttc/detection/geometry_contract.py",
        "data/splits/eap_detection_candidate_v1.json",
        "docs/detection_local_20261029/evidence/pilot_manifest.json",
        "data/manifests/eap_detection_public_files_v1.json",
    ]
    for relative in relative_sources:
        committed_bytes = subprocess.check_output(
            ["git", "show", f"{BASE}:{relative}"], cwd=repo
        )
        if (repo / relative).read_bytes() != committed_bytes:
            parser.error(f"Source differs from the inspected commit: {relative}")
    split = json.loads((repo / relative_sources[2]).read_text())
    pilot = json.loads((repo / relative_sources[3]).read_text())
    inventory = json.loads((repo / relative_sources[4]).read_text())
    if split["dataset_revision"] != REVISION or inventory["revision"] != REVISION:
        parser.error(
            "The split or public file manifest has a different dataset revision"
        )
    assignments = split["assignments"]
    roles = {seq: role for role, sequences in assignments.items() for seq in sequences}
    assert set(assignments) == {"train", "selection", "calibration"}
    assert len(roles) == sum(map(len, assignments.values())) == 40
    token_roles = {
        token: role
        for role in ("train", "selection")
        for token in pilot[role]["sample_tokens"]
    }
    assert len(token_roles) == 512

    def verified_input(relative: str) -> Path:
        path = metadata / relative
        expected = inventory["files"][relative]
        if (
            path.stat().st_size != expected["size_bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Pinned public input bytes differ: {relative}")
        return path

    media_path = verified_input("data/train.parquet")
    assert sha256(media_path) == split["media_sha256"] == pilot["source_media_sha256"]
    geometry_path = repo / relative_sources[1]
    spec = importlib.util.spec_from_file_location("pinned_geometry", geometry_path)
    assert spec is not None and spec.loader is not None
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    role_counts = defaultdict(Counter)
    role_tracks = defaultdict(lambda: defaultdict(set))
    sequence_counts, label_hashes, selected_rows = {}, {}, []
    for seq in sorted(roles):
        relative = f"data/train/{seq}/labels.parquet"
        path = verified_input(relative)
        label_hashes[relative] = sha256(path)
        columns = [
            "split",
            "sequence_id",
            "sample_token",
            "category",
            "track_id",
            "bbox_3d_ego",
        ]
        rows = pq.read_table(path, columns=columns).to_pylist()
        assert all(
            row["split"] == "train" and row["sequence_id"] == seq for row in rows
        )
        sequence_counts[seq] = dict(Counter(row["category"] for row in rows))
        for row in rows:
            category, role = row["category"], roles[seq]
            role_counts[role][category] += 1
            role_tracks[role][category].add((seq, row["track_id"]))
            if category in RARE or row["sample_token"] in token_roles:
                selected_rows.append(row)
    tokens = sorted({row["sample_token"] for row in selected_rows})
    media_rows = pq.read_table(
        media_path,
        columns=["sample_token", "split", "sequence_id", "K_event", "T_event_ego"],
        filters=[("sample_token", "in", tokens)],
    ).to_pylist()
    media = {row["sample_token"]: row for row in media_rows}
    assert len(media_rows) == len(media) == len(tokens)
    assert all(row["split"] == "train" for row in media_rows)
    pilot_counts = defaultdict(Counter)
    pilot_classes = defaultdict(lambda: defaultdict(Counter))
    rare_counts, rare_tracks, rare_sequences = (
        defaultdict(Counter),
        defaultdict(set),
        defaultdict(set),
    )
    rare_by_sequence = defaultdict(Counter)
    for row in selected_rows:
        token, category, seq = row["sample_token"], row["category"], row["sequence_id"]
        assert media[token]["sequence_id"] == seq
        reason = filter_reason(
            row["bbox_3d_ego"], media[token], geometry.original_bundle_corners
        )
        if token in token_roles:
            role = token_roles[token]
            pilot_counts[role][reason] += 1
            pilot_classes[role][category][reason] += 1
        if category in RARE:
            key = f"{roles[seq]}/{category}"
            rare_counts[key]["raw"] += 1
            if reason == "admitted":
                rare_counts[key]["admitted"] += 1
                rare_tracks[key].add((seq, row["track_id"]))
                rare_sequences[key].add(seq)
                rare_by_sequence[seq][category] += 1
    result = {
        "schema": "pilot_population_readonly_diagnostic_v1",
        "scope": "Public train labels and geometry metadata only; no pixels/events/test/model run",
        "filter_assumption": "Documented original 1280x720 RGB; legacy pilot 640x384 pinhole formulas",
        "identity_limit": "Distinct (sequence_id, track_id) annotations; not independent acquisitions",
        "repository_commit": commit,
        "dataset_revision": REVISION,
        "split_sha256": sha256(repo / relative_sources[2]),
        "source_sha256": {name: sha256(repo / name) for name in relative_sources},
        "script_sha256": sha256(Path(__file__)),
        "media_sha256": sha256(media_path),
        "labels_sha256": label_hashes,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "raw_labels_by_role": {
            role: dict(counts) for role, counts in role_counts.items()
        },
        "distinct_sequence_track_per_class": {
            role: {category: len(ids) for category, ids in classes.items()}
            for role, classes in role_tracks.items()
        },
        "raw_sequence_class_counts": sequence_counts,
        "pilot_filter_counts": {
            role: dict(counts) for role, counts in pilot_counts.items()
        },
        "pilot_filter_counts_per_class": {
            role: {category: dict(counts) for category, counts in classes.items()}
            for role, classes in pilot_classes.items()
        },
        "rare_classes_under_legacy_pinhole_640x384_profile": {
            key: {
                **dict(counts),
                "admitted_distinct_sequence_track": len(rare_tracks[key]),
                "admitted_sequences": sorted(rare_sequences[key]),
            }
            for key, counts in rare_counts.items()
        },
        "rare_admitted_by_sequence": {
            seq: dict(counts) for seq, counts in rare_by_sequence.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "labels": len(label_hashes),
                "metadata_only": True,
            }
        )
    )


if __name__ == "__main__":
    main()

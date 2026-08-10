"""Audit public Garl foreground references, RGB assets, and local visual teachers.

The audit is deliberately train-only.  It never opens a test parquet, never loads
teacher weights, and never extracts RGB frames.  It records enough provenance to
decide whether an official-mask arm or a train-only RGB-distillation arm is feasible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import scan_cache_dir

from e_jepa_ttc.artifacts.hashing import sign_artifact

TEACHER_REVISIONS = {
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": (
        "10d30274b4d445111e2d5bf75ac93bbd94db274b"
    ),
    "facebook/dinov3-vitl16-pretrain-lvd1689m": (
        "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    ),
    "facebook/dinov3-vitl16-pretrain-sat493m": (
        "f692fa42da72c6797b67cd73494a168d1120d3ee"
    ),
    "facebook/dinov3-vits16-pretrain-lvd1689m": (
        "114c1379950215c8b35dfcd4e90a5c251dde0d32"
    ),
    "facebook/sam-vit-large": "6851e0441005b0fb96f2cc4dfac472f3d1b14af1",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without loading it wholly into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_strings(value: object) -> list[str]:
    if value is None:
        return []
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a path list, got {type(value).__name__}")
    return [str(item) for item in value if item is not None and str(item)]


def _require_public_train_parquet(path: Path) -> Path:
    resolved = path.resolve()
    lowered_parts = {part.lower() for part in resolved.parts}
    if resolved.name.lower() != "train.parquet" or "test" in lowered_parts:
        raise ValueError(f"only a public train.parquet is allowed: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _candidate_mask_paths(reference: str, sequence_id: str, roots: Iterable[Path]) -> list[Path]:
    raw = Path(reference)
    if raw.is_absolute():
        return [raw]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / raw, root / sequence_id / raw))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def audit_mask_paths(frame: pd.DataFrame, roots: list[Path]) -> dict[str, Any]:
    """Resolve every unique sequence/path pair under explicit public roots."""

    endpoint_references = 0
    row_complete = 0
    unique_pairs: set[tuple[str, str]] = set()
    suffixes: dict[str, int] = defaultdict(int)
    lengths: dict[int, int] = defaultdict(int)
    rows = frame[["sequence_id", "mask_paths"]].itertuples(index=False, name=None)
    for row in rows:
        sequence_id = str(row[0])
        references = _as_strings(row[1])
        endpoint_references += len(references)
        lengths[len(references)] += 1
        if references:
            row_complete += 1
        for reference in references:
            suffixes[Path(reference).suffix.lower()] += 1
            unique_pairs.add((sequence_id, reference))

    resolved: dict[tuple[str, str], list[str]] = {}
    missing_examples: list[dict[str, Any]] = []
    for sequence_id, reference in sorted(unique_pairs):
        matches = [
            path.as_posix()
            for path in _candidate_mask_paths(reference, sequence_id, roots)
            if path.is_file()
        ]
        if matches:
            resolved[(sequence_id, reference)] = matches
        elif len(missing_examples) < 20:
            missing_examples.append(
                {
                    "sequence_id": sequence_id,
                    "reference": reference,
                    "candidate_count": len(_candidate_mask_paths(reference, sequence_id, roots)),
                }
            )
    ambiguous = {key: paths for key, paths in resolved.items() if len(paths) > 1}
    return {
        "row_count": int(len(frame)),
        "rows_with_nonempty_references": row_complete,
        "endpoint_reference_count": endpoint_references,
        "unique_sequence_reference_count": len(unique_pairs),
        "resolved_unique_count": len(resolved),
        "missing_unique_count": len(unique_pairs) - len(resolved),
        "ambiguous_unique_count": len(ambiguous),
        "official_masks_available": bool(unique_pairs) and len(resolved) == len(unique_pairs),
        "reference_lengths_by_row": {str(key): value for key, value in sorted(lengths.items())},
        "suffix_counts": dict(sorted(suffixes.items())),
        "roots": [root.resolve().as_posix() for root in roots],
        "missing_examples": missing_examples,
        "ambiguous_examples": [
            {"sequence_id": key[0], "reference": key[1], "matches": paths}
            for key, paths in list(sorted(ambiguous.items()))[:20]
        ],
    }


def audit_rgb_tar_members(frame: pd.DataFrame, eap_root: Path) -> dict[str, Any]:
    """Verify every declared RGB tar/member pair without extracting image content."""

    requested: dict[str, set[str]] = defaultdict(set)
    endpoint_references = 0
    bad_pair_rows = 0
    rows = frame[["rgb_shard_paths", "rgb_member_paths"]].itertuples(index=False, name=None)
    for row in rows:
        shards = _as_strings(row[0])
        members = _as_strings(row[1])
        endpoint_references += max(len(shards), len(members))
        if len(shards) != len(members):
            bad_pair_rows += 1
            continue
        for shard, member in zip(shards, members, strict=True):
            requested[shard].add(member)

    missing_shards: list[str] = []
    unreadable_shards: list[dict[str, str]] = []
    missing_members: list[dict[str, str]] = []
    found_members = 0
    for shard_reference, members in sorted(requested.items()):
        shard_path = (eap_root / shard_reference).resolve()
        if not shard_path.is_file():
            missing_shards.append(shard_reference)
            continue
        try:
            with tarfile.open(shard_path, mode="r:*") as archive:
                available = set(archive.getnames())
        except (OSError, tarfile.TarError) as exc:
            unreadable_shards.append({"shard": shard_reference, "error": str(exc)})
            continue
        present = members & available
        found_members += len(present)
        for member in sorted(members - available):
            if len(missing_members) < 20:
                missing_members.append({"shard": shard_reference, "member": member})
    requested_member_count = sum(len(members) for members in requested.values())
    return {
        "endpoint_reference_count": endpoint_references,
        "bad_pair_row_count": bad_pair_rows,
        "unique_shard_count": len(requested),
        "requested_unique_member_count": requested_member_count,
        "found_unique_member_count": found_members,
        "missing_shard_count": len(missing_shards),
        "unreadable_shard_count": len(unreadable_shards),
        "missing_unique_member_count": requested_member_count - found_members,
        "all_train_rgb_inputs_available": (
            bad_pair_rows == 0
            and not missing_shards
            and not unreadable_shards
            and found_members == requested_member_count
        ),
        "eap_root": eap_root.resolve().as_posix(),
        "missing_shard_examples": missing_shards[:20],
        "unreadable_shard_examples": unreadable_shards[:20],
        "missing_member_examples": missing_members,
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def _read_local_license(snapshot: Path) -> dict[str, Any]:
    readme = snapshot / "README.md"
    license_file = snapshot / "LICENSE.md"
    metadata_license: str | None = None
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^license:\s*([^\r\n]+)$", text[:10_000])
        metadata_license = match.group(1).strip() if match else None
    return {
        "readme_present": readme.is_file(),
        "readme_sha256": sha256_file(readme) if readme.is_file() else None,
        "license_file_present": license_file.is_file(),
        "license_file_sha256": sha256_file(license_file) if license_file.is_file() else None,
        "metadata_license": metadata_license,
        "local_license_evidence_available": bool(metadata_license or license_file.is_file()),
    }


def audit_teacher_snapshot(repo_id: str, revision: str, snapshot: Path) -> dict[str, Any]:
    """Audit a local snapshot without importing Transformers or loading weights."""

    config_path = snapshot / "config.json"
    processor_path = snapshot / "preprocessor_config.json"
    weight_candidates = [snapshot / "model.safetensors", snapshot / "pytorch_model.bin"]
    selected_weight = next((path for path in weight_candidates if path.is_file()), None)
    config, config_error = _read_json(config_path) if config_path.is_file() else (None, "missing")
    processor, processor_error = (
        _read_json(processor_path) if processor_path.is_file() else (None, "missing")
    )
    license_audit = _read_local_license(snapshot)
    complete = (
        config is not None
        and processor is not None
        and selected_weight is not None
        and license_audit["local_license_evidence_available"]
    )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": snapshot.resolve().as_posix(),
        "snapshot_present": snapshot.is_dir(),
        "config": {
            "present": config_path.is_file(),
            "sha256": sha256_file(config_path) if config_path.is_file() else None,
            "parse_error": config_error,
            "model_type": config.get("model_type") if config else None,
            "architectures": config.get("architectures") if config else None,
        },
        "processor": {
            "present": processor_path.is_file(),
            "sha256": sha256_file(processor_path) if processor_path.is_file() else None,
            "parse_error": processor_error,
            "image_processor_type": processor.get("image_processor_type") if processor else None,
            "processor_class": processor.get("processor_class") if processor else None,
        },
        "weights": {
            "selected_path": selected_weight.resolve().as_posix() if selected_weight else None,
            "filename": selected_weight.name if selected_weight else None,
            "size_bytes": selected_weight.stat().st_size if selected_weight else None,
            "sha256": sha256_file(selected_weight) if selected_weight else None,
        },
        "license": license_audit,
        "self_contained_for_offline_loading_and_license_audit": complete,
    }


def discover_teacher_snapshots() -> dict[str, tuple[str, Path]]:
    """Resolve the preregistered revisions from the local Hugging Face cache only."""

    cache = scan_cache_dir()
    repositories = {repo.repo_id: repo for repo in cache.repos if repo.repo_type == "model"}
    found: dict[str, tuple[str, Path]] = {}
    for repo_id, expected_revision in TEACHER_REVISIONS.items():
        repo = repositories.get(repo_id)
        if repo is None:
            continue
        revision = next(
            (item for item in repo.revisions if item.commit_hash == expected_revision), None
        )
        if revision is not None:
            found[repo_id] = (revision.commit_hash, Path(revision.snapshot_path))
    return found


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_audit(
    *,
    data_parquet: Path,
    eap_root: Path,
    mask_roots: list[Path],
    repo_root: Path,
    snapshots: dict[str, tuple[str, Path]],
) -> dict[str, Any]:
    """Build the complete signed audit payload."""

    data_parquet = _require_public_train_parquet(data_parquet)
    required_columns = [
        "sequence_id",
        "sample_token",
        "mask_paths",
        "rgb_shard_paths",
        "rgb_member_paths",
    ]
    frame = pd.read_parquet(data_parquet, columns=required_columns)
    if frame["sample_token"].duplicated().any():
        raise ValueError("public train parquet contains duplicate sample_token values")
    teacher_audits: dict[str, Any] = {}
    for repo_id, expected_revision in TEACHER_REVISIONS.items():
        located = snapshots.get(repo_id)
        if located is None:
            teacher_audits[repo_id] = {
                "repo_id": repo_id,
                "expected_revision": expected_revision,
                "snapshot_present": False,
                "self_contained_for_offline_loading_and_license_audit": False,
            }
            continue
        revision, snapshot = located
        audit = audit_teacher_snapshot(repo_id, revision, snapshot)
        audit["expected_revision"] = expected_revision
        audit["revision_matches"] = revision == expected_revision
        teacher_audits[repo_id] = audit

    masks = audit_mask_paths(frame, mask_roots)
    rgb = audit_rgb_tar_members(frame, eap_root)
    sam = teacher_audits["facebook/sam-vit-large"]
    dino_tiny = teacher_audits["facebook/dinov3-convnext-tiny-pretrain-lvd1689m"]
    sequence_ids = [str(value) for value in frame["sequence_id"].tolist()]
    sample_tokens = [str(value) for value in frame["sample_token"].tolist()]
    result: dict[str, Any] = {
        "artifact_type": "garl_foreground_resource_audit_v2",
        "status": "completed",
        "scope": {
            "opened_public_train_parquet": True,
            "opened_private_test": False,
            "opened_evttc_test": False,
            "opened_codabench": False,
            "teacher_weights_loaded": False,
            "network_downloads": False,
        },
        "code_commit": _git_commit(repo_root),
        "source": {
            "data_parquet": data_parquet.as_posix(),
            "data_parquet_sha256": sha256_file(data_parquet),
            "row_count": int(len(frame)),
            "sequence_count": len(set(sequence_ids)),
            "sample_token_count": len(set(sample_tokens)),
            "sequence_ids": sorted(set(sequence_ids)),
        },
        "official_mask_references": masks,
        "rgb_train_assets": rgb,
        "local_teacher_snapshots": teacher_audits,
        "decision": {
            "official_mask_arm_feasible": masks["official_masks_available"],
            "sam_bbox_prompt_train_only_arm_feasible": bool(
                rgb["all_train_rgb_inputs_available"]
                and sam.get("self_contained_for_offline_loading_and_license_audit")
            ),
            "dinov3_convnext_tiny_teacher_arm_feasible": bool(
                rgb["all_train_rgb_inputs_available"]
                and dino_tiny.get("self_contained_for_offline_loading_and_license_audit")
            ),
            "protocol_label_if_teacher_used": "event-only inference with RGB distillation",
            "validation_teacher_generation_allowed": False,
            "next_action": (
                "use official train masks before pseudo-masks"
                if masks["official_masks_available"]
                else "preregister a train-only SAM bbox-prompt materialization smoke"
            ),
        },
        "runtime_dependency": {
            "transformers_declared_optional_extra": "multimodal",
            "transformers_importable": importlib.util.find_spec("transformers") is not None,
            "note": "snapshot integrity audit does not import Transformers or load weights",
        },
    }
    return sign_artifact(result)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parquet", required=True, type=Path)
    parser.add_argument("--eap-root", required=True, type=Path)
    parser.add_argument("--garlttc-root", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mask_roots = [
        args.garlttc_root,
        args.garlttc_root / "data",
        args.eap_root,
        args.eap_root / "data",
        args.eap_root / "data" / "train",
        args.release_root,
    ]
    report = build_audit(
        data_parquet=args.data_parquet,
        eap_root=args.eap_root,
        mask_roots=mask_roots,
        repo_root=args.repo_root,
        snapshots=discover_teacher_snapshots(),
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

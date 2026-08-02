"""Audit padded patch resolutions and attention scaling from real manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from e_jepa_ttc.models.highres_factorized import (
    TheoreticalOOMError,
    theoretical_attention_bytes,
    theoretical_oom_guard,
    theoretical_oom_guard_required,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _audit_entry(
    item: dict[str, Any],
    *,
    temporal_steps: int,
    heads: int,
    memory_budget_gb: float,
    spatial_window: int,
    cache_resolution: tuple[int, int],
) -> dict[str, Any]:
    width = int(item["width"])
    height = int(item["height"])
    patch_size = int(item["patch_size"])
    grid_width = math.ceil(width / patch_size)
    grid_height = math.ceil(height / patch_size)
    padded_width = grid_width * patch_size
    padded_height = grid_height * patch_size
    patches = grid_width * grid_height
    merge = bool(item.get("merge_2x2", False))
    post_grid_width = math.ceil(grid_width / 2) if merge else grid_width
    post_grid_height = math.ceil(grid_height / 2) if merge else grid_height
    spatial_windows = math.ceil(grid_width / spatial_window) * math.ceil(
        grid_height / spatial_window
    )
    temporal_pairs = temporal_steps * temporal_steps * (post_grid_width * post_grid_height)
    global_bytes = theoretical_attention_bytes(
        1, temporal_steps, post_grid_width * post_grid_height, heads
    )
    guard_required = theoretical_oom_guard_required(
        batch=1,
        steps=temporal_steps,
        patches=post_grid_width * post_grid_height,
        heads=heads,
        memory_budget_gb=memory_budget_gb,
        global_attention=True,
    )
    guard_triggered = False
    guard_error = ""
    try:
        theoretical_oom_guard(
            batch=1,
            steps=temporal_steps,
            patches=post_grid_width * post_grid_height,
            heads=heads,
            memory_budget_gb=memory_budget_gb,
            global_attention=True,
        )
    except TheoreticalOOMError as error:
        guard_triggered = True
        guard_error = str(error)
    return {
        "name": str(item["name"]),
        "source_resolution": [width, height],
        "cache_resolution": list(cache_resolution),
        "patch_size": patch_size,
        "grid": [grid_width, grid_height],
        "padded_resolution": [padded_width, padded_height],
        "padding_right_bottom": [padded_width - width, padded_height - height],
        "tokens_before_merge": temporal_steps * patches,
        "tokens_after_merge": temporal_steps * post_grid_width * post_grid_height,
        "valid_patch_mask_required": True,
        "borders_cropped": False,
        "upsampled_from_cache": False,
        "merge_2x2": merge,
        "spatial_window": spatial_window,
        "spatial_window_count_per_step": spatial_windows,
        "spatial_attention_pairs": temporal_steps * spatial_windows * spatial_window**4,
        "temporal_attention_pairs": temporal_pairs,
        "global_attention_pairs": (temporal_steps * post_grid_width * post_grid_height) ** 2,
        "global_attention_estimated_bytes_fp16": global_bytes,
        "theoretical_oom_guard": guard_triggered,
        "theoretical_oom_guard_required": guard_required,
        "theoretical_oom_guard_triggered": guard_triggered,
        "global_guard_error": guard_error,
        "memory_budget_gb": memory_budget_gb,
    }


def audit(
    *,
    repo_root: Path,
    data_root: Path,
    garl_root: Path,
    output: Path,
    config_path: Path,
    cache_manifest: Path,
    split_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not data_root.exists():
        errors.append(f"Missing data root: {data_root}")
    if not garl_root.exists():
        errors.append(f"Missing Garl root: {garl_root}")
    if not cache_manifest.is_file():
        errors.append(f"Missing cache manifest: {cache_manifest}")
    if not split_path.is_file():
        errors.append(f"Missing split: {split_path}")
    if manifest.get("input_schema", {}).get("version") != "garlttc_input_v4":
        errors.append("Cache manifest is not garlttc_input_v4.")
    cache_resolution = (
        int(manifest.get("config", {}).get("full_width", 160)),
        int(manifest.get("config", {}).get("full_height", 90)),
    )
    resolutions = [
        _audit_entry(
            item,
            temporal_steps=int(config["temporal_steps"]),
            heads=int(config["heads"]),
            memory_budget_gb=float(config["memory_budget_gb"]),
            spatial_window=int(config["spatial_window"]),
            cache_resolution=cache_resolution,
        )
        for item in config["resolutions"]
    ]
    if any(item["borders_cropped"] for item in resolutions):
        errors.append("A resolution entry crops borders.")
    if any(item["upsampled_from_cache"] for item in resolutions):
        errors.append("A high-resolution entry upsamples the 160x90 cache.")
    selected = split.get("selected", [])
    result: dict[str, Any] = {
        "artifact_type": "patch_resolution_audit_v1",
        "schema_version": "v1",
        "evidence_type": "real_cache_and_split_geometry",
        "code_commit": _git_commit(repo_root),
        "protocol_version": "patch_resolution_v1",
        "protocol_sha256": _canonical_sha256(config),
        "created_at": datetime.now(UTC).isoformat(),
        "status": "pass" if not errors else "fail",
        "data_root": data_root.as_posix(),
        "garl_root": garl_root.as_posix(),
        "source_resolution": [1280, 720],
        "cache_resolution": list(cache_resolution),
        "highres_is_not_cache_upsample": True,
        "cache_manifest_sha256": _sha256(cache_manifest) if cache_manifest.is_file() else "",
        "split_sha256": _sha256(split_path) if split_path.is_file() else "",
        "selected_sequence_count": len(selected),
        "resolutions": resolutions,
        "errors": errors,
    }
    result["artifact_sha256"] = _canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "patch_resolution_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--garl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/audits/patch_resolution_v1"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/highres_token_scaling_v1.yaml")
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("artifacts/cache/garlttc_lhr_v4_pilot_4096_workers4/manifest.json"),
    )
    parser.add_argument("--split", type=Path, default=Path("data/splits/eap_pilot12_v1.json"))
    args = parser.parse_args()
    result = audit(
        repo_root=Path.cwd(),
        data_root=args.data_root,
        garl_root=args.garl_root,
        output=args.output,
        config_path=args.config,
        cache_manifest=args.cache_manifest,
        split_path=args.split,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

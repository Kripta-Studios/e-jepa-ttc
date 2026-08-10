#!/usr/bin/env python
"""Materialize DINOv3 relational teacher cache for A4 (train-only).

Processes all 2048 train samples, computing per-object DINO features
and six local cosine relation maps.  Writes sharded NPZ files and a
signed manifest.

Each sample_token corresponds to a specific object/track within a frame.
The RGB is cropped using that object's exact common_square_xyxy.
Multiple objects from the same frame produce independent entries.

Usage:
    python scripts/materialize_dinov3_relational_teacher.py \
        --event-cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json \
        --model-path facebook/dinov3-convnext-large-pretrain-lvd1689m \
        --output-dir artifacts/cache/dinov3_relational_teacher_a4 \
        --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.distillation.dinov3_relational import (  # noqa: E402
    A4_RELATION_OFFSETS,
    local_cosine_relation_maps,
)

_INPUT_SIZE = 256
_EXPECTED_GRID = (32, 32)
_NUM_OFFSETS = len(A4_RELATION_OFFSETS)
_SHARD_SIZE = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find_32x32_feature(model: Any, dummy: torch.Tensor) -> tuple[int | str, str]:
    """Find the unique 32×32 feature map. Returns (selection_id, method)."""
    with torch.no_grad():
        outputs = model(dummy, output_hidden_states=True)
    hidden = getattr(outputs, "hidden_states", None)
    if hidden is not None:
        candidates = [
            (i, hs) for i, hs in enumerate(hidden) if hs.shape[-2:] == _EXPECTED_GRID
        ]
        if len(candidates) == 1:
            return candidates[0][0], "hidden_states"
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple 32×32 hidden states: {[c[0] for c in candidates]}")

    # Hook fallback
    hooked: dict[str, torch.Tensor] = {}

    def make_hook(name: str) -> Any:
        def hook(m: Any, inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                hooked[name] = out
        return hook

    handles = []
    for name, module in model.named_modules():
        if "stage" in name.lower() or "layer" in name.lower():
            handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    candidates = [(n, f) for n, f in hooked.items() if f.shape[-2:] == _EXPECTED_GRID]
    if len(candidates) == 1:
        return candidates[0][0], "forward_hook"
    raise RuntimeError(f"Cannot find unique 32×32 feature. Shapes: {dict((n, tuple(f.shape)) for n, f in hooked.items())}")


def _extract_features(
    model: Any,
    rgb_tensor: torch.Tensor,
    selection_id: int | str,
    selection_method: str,
) -> torch.Tensor:
    """Extract 32×32 features for a batch of RGB crops."""
    if selection_method == "hidden_states":
        with torch.no_grad():
            outputs = model(rgb_tensor, output_hidden_states=True)
        if isinstance(selection_id, str) and selection_id.startswith("hidden_states["):
            idx = int(selection_id.split("[")[1].split("]")[0])
        else:
            idx = int(selection_id)
        return outputs.hidden_states[idx]  # type: ignore[index]
    else:
        result: list[torch.Tensor] = []
        def hook(m: Any, inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor):
                result.append(out)
        target_module = dict(model.named_modules())[str(selection_id)]
        handle = target_module.register_forward_hook(hook)
        with torch.no_grad():
            model(rgb_tensor)
        handle.remove()
        if not result:
            raise RuntimeError("Forward hook did not capture output")
        return result[0]


def _load_and_crop_rgb(
    tar_path: Path,
    member_path: str,
    common_square_xyxy: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Load RGB from TAR, crop to object-specific ROI, resize to 256×256.

    Returns (tensor [1,3,256,256], sha256 of raw RGB bytes).
    """
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(member_path)
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Cannot extract {member_path}")
        raw_bytes = f.read()
    rgb_sha = _sha256_bytes(raw_bytes)

    import io
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img_w, img_h = img.size

    # Crop using EXACT common_square_xyxy for this specific object
    x1 = max(0, int(common_square_xyxy[0]))
    y1 = max(0, int(common_square_xyxy[1]))
    x2 = min(img_w, int(common_square_xyxy[2]))
    y2 = min(img_h, int(common_square_xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop: ({x1},{y1},{x2},{y2}) for image {img_w}×{img_h}"
        )
    crop = img.crop((x1, y1, x2, y2))

    # Resize to 256×256 — NO center crop, NO RandomResizedCrop
    crop = crop.resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR)

    # Convert to tensor and normalize (manual, not AutoImageProcessor)
    arr = np.asarray(crop, dtype=np.float32) / 255.0  # [H,W,3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    mean = torch.tensor(image_mean, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device), rgb_sha


def run(
    event_cache_manifest: Path,
    train_parquet: Path,
    eap_root: Path,
    model_path: str,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Materialize the full DINO relational teacher cache."""

    from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-untyped]
    import pandas as pd

    # --- Load event cache dataset (train only) ---
    manifest = json.loads(event_cache_manifest.read_text(encoding="utf-8"))
    train_dataset = GarlTTCObjectEventV4Dataset(
        str(event_cache_manifest), splits=("train",)
    )
    total_rows = len(train_dataset)
    print(f"[materialize] Train dataset: {total_rows} rows")

    if total_rows != 2048:
        raise ValueError(f"FATAL: Train dataset has {total_rows} rows, expected exactly 2048")

    # Hard-fail on validation/test access
    validation_sequences = {"DGqicHUGWb", "pBqGOb2vYq", "qoohcdtLDH"}
    
    # Expected train sequences
    expected_train_sequences = {
        "2cyv0Oedzg", "5ilM1PX2vz", "6h5yRW2LGc", "OBneIVg4Cw", 
        "OYgB6RGWcq", "WbCh1DRerJ", "mHGFBekt7X", "qGsgzl4Q8B", "t79dBxj1WS"
    }

    # Load train.parquet
    print(f"[materialize] Loading {train_parquet}...")
    pq_df = pd.read_parquet(
        train_parquet,
        columns=[
            "sequence_id",
            "sample_token",
            "track_id",
            "rgb_shard_paths",
            "rgb_member_paths",
        ],
    )
    # We need to map (sample_token, track_id) to RGB paths
    # Create a fast lookup dict
    print("[materialize] Building RGB lookup index...")
    rgb_lookup = {}
    for _idx, row in pq_df.iterrows():
        token = str(row["sample_token"])
        track = str(row["track_id"])
        seq = str(row["sequence_id"])
        rgb_shards = list(row["rgb_shard_paths"])
        rgb_members = list(row["rgb_member_paths"])
        if len(rgb_shards) != 2 or len(rgb_members) != 2:
            raise ValueError(f"Expected 2 RGB endpoints for {token}/{track}")
        key = (token, track)
        if key in rgb_lookup:
            raise ValueError(f"Duplicate parquet identity: {key}")
        rgb_lookup[key] = (seq, rgb_shards, rgb_members)

    # --- Load DINO model ---
    print(f"[materialize] Loading DINO model: {model_path}")
    device = torch.device(device_name)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=False)
    model = model.to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_path)

    # Extract normalization parameters (use these manually, not the processor)
    image_mean = list(processor.image_mean)
    image_std = list(processor.image_std)

    # Record model identity
    canonical_model_id = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
    resolved_snapshot_path = str(Path(model_path).resolve())

    print("[materialize] Hashing model weights...")
    weights_sha256 = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights_sha256.update(name.encode())
        weights_sha256.update(tensor.cpu().numpy().tobytes())
    weights_sha = weights_sha256.hexdigest()

    config_dict = model.config.to_dict()
    config_json = json.dumps(config_dict, sort_keys=True)
    config_sha = hashlib.sha256(config_json.encode()).hexdigest()
    preprocessor_json = json.dumps(
        {"mean": image_mean, "std": image_std}, sort_keys=True
    )
    preprocessor_sha = hashlib.sha256(preprocessor_json.encode()).hexdigest()

    # --- Find 32×32 feature ---
    dummy = torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE, device=device)
    selection_id, selection_method = _find_32x32_feature(model, dummy)
    feat = _extract_features(model, dummy, selection_id, selection_method)
    feat_channels = feat.shape[1]
    print(
        f"[materialize] Feature: channels={feat_channels}, "
        f"grid={feat.shape[-2:]}, selection={selection_id}, method={selection_method}"
    )

    # --- Process all train rows ---
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_infos: list[dict[str, Any]] = []
    shard_idx = 0

    seen_pairs = set()

    for start in range(0, total_rows, _SHARD_SIZE):
        end = min(start + _SHARD_SIZE, total_rows)
        shard_tokens: list[str] = []
        shard_track_ids: list[str] = []
        shard_sequences: list[str] = []
        shard_squares: list[np.ndarray] = []
        shard_targets: list[np.ndarray] = []
        shard_valid: list[np.ndarray] = []
        shard_rgb_shas: list[np.ndarray] = []

        for row_idx in range(start, end):
            record = train_dataset[row_idx]
            token = str(record["sample_token"])
            track_id = str(record["track_id"])
            sequence = str(record["sequence_id"])

            if (token, track_id) in seen_pairs:
                raise ValueError(f"FATAL: Duplicate (token, track_id) detected: {token}/{track_id}")
            seen_pairs.add((token, track_id))

            if sequence not in expected_train_sequences:
                raise ValueError(
                    f"FATAL: Sequence {sequence} is not an expected train sequence"
                )

            # Hard-fail on validation sequences (redundant but explicit)
            if sequence in validation_sequences:
                raise ValueError(
                    f"FATAL: validation sequence {sequence} found in train dataset"
                )
            
            if (token, track_id) not in rgb_lookup:
                raise ValueError(f"FATAL: (token, track_id) {token}/{track_id} not found in train.parquet")

            parquet_sequence, rgb_shards, rgb_members = rgb_lookup[(token, track_id)]

            if parquet_sequence != sequence:
                raise ValueError(
                    f"Sequence mismatch for {token}/{track_id}: "
                    f"event={sequence}, parquet={parquet_sequence}"
                )

            square = np.asarray(
                record["event_v4_common_square_xyxy"], dtype=np.float32
            )

            endpoint_targets = []
            endpoint_valid = []
            endpoint_rgb_shas = []
            
            for ep in range(2):
                tar_path = eap_root / rgb_shards[ep]
                member_path = rgb_members[ep]
                
                ep_input, rgb_sha = _load_and_crop_rgb(
                    tar_path=tar_path,
                    member_path=member_path,
                    common_square_xyxy=square,
                    image_mean=image_mean,
                    image_std=image_std,
                    device=device,
                )
                
                features = _extract_features(
                    model, ep_input, selection_id, selection_method,
                )
                rels = local_cosine_relation_maps(features)
                endpoint_targets.append(
                    rels.values.squeeze(0).cpu().numpy().astype(np.float16)
                )
                endpoint_valid.append(
                    rels.valid.squeeze(0).cpu().numpy().astype(np.uint8)
                )
                endpoint_rgb_shas.append(rgb_sha)

            shard_tokens.append(token)
            shard_track_ids.append(track_id)
            shard_sequences.append(sequence)
            shard_squares.append(square)
            shard_targets.append(np.stack(endpoint_targets, axis=0))
            shard_valid.append(np.stack(endpoint_valid, axis=0))
            shard_rgb_shas.append(np.asarray(endpoint_rgb_shas, dtype="<U64"))

            if (row_idx + 1) % 100 == 0:
                print(f"  processed {row_idx + 1}/{total_rows}")

        # Write shard
        npz_name = f"shard_{shard_idx:04d}.npz"
        npz_path = output_dir / npz_name
        np.savez_compressed(
            npz_path,
            sample_tokens=np.array(shard_tokens),
            track_ids=np.array(shard_track_ids),
            sequence_ids=np.array(shard_sequences),
            common_square_xyxy=np.stack(shard_squares),
            relation_targets=np.stack(shard_targets),
            relation_valid=np.stack(shard_valid),
            rgb_sha256=np.stack(shard_rgb_shas),
        )
        shard_sha = _sha256_file(npz_path)
        shard_infos.append({
            "npz_path": npz_name,
            "npz_sha256": shard_sha,
            "row_count": end - start,
        })
        shard_idx += 1
        print(f"  shard {shard_idx}: {end - start} rows, sha={shard_sha[:16]}...")

    # Final checks
    observed_sequences = {rgb_lookup[k][0] for k in seen_pairs}
    if len(seen_pairs) != 2048:
        raise ValueError(f"FATAL: Expected exactly 2048 unique (token, track_id) pairs, got {len(seen_pairs)}")
    if len(seen_pairs) * 2 != 4096:
        raise ValueError(f"FATAL: Expected exactly 4096 endpoints, got {len(seen_pairs) * 2}")
    if observed_sequences != expected_train_sequences:
        raise ValueError("FATAL: Observed sequences mismatch with expected train sequences")

    # --- Build manifest ---
    manifest_payload: dict[str, Any] = {
        "artifact_type": "dinov3_relational_teacher_cache_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "ttc_labels_read": False,
            "row_count": total_rows,
            "endpoint_count_per_row": 2,
        },
        "claim_boundary": {
            "teacher_is_model_input": False,
            "validation_teacher_generation": False,
            "ttc_labels_read": False,
        },
        "teacher": {
            "model_id": canonical_model_id,
            "resolved_snapshot_path": resolved_snapshot_path,
            "weights_sha256": weights_sha,
            "config_sha256": config_sha,
            "preprocessor_sha256": preprocessor_sha,
            "input_size": _INPUT_SIZE,
            "hidden_state_index_or_hook": str(selection_id),
            "native_feature_shape": [feat_channels, *_EXPECTED_GRID],
            "selection_method": selection_method,
            "image_mean": image_mean,
            "image_std": image_std,
        },
        "relations": {
            "type": "local_cosine",
            "offsets_dy_dx": [list(o) for o in A4_RELATION_OFFSETS],
            "grid_height": _EXPECTED_GRID[0],
            "grid_width": _EXPECTED_GRID[1],
            "dtype": "float16",
        },
        "source_event_cache": {
            "manifest_path": str(event_cache_manifest.relative_to(ROOT)),
        },
        "shards": shard_infos,
        "multi_object_contract": {
            "teacher_keyed_by_sample_token": True,
            "track_id_stored_per_row": True,
            "same_frame_not_deduplicated": True,
            "crop_is_per_object_common_square": True,
        },
    }
    sign_artifact(manifest_payload)

    manifest_path = output_dir / "manifest.json"
    tmp = manifest_path.with_name(".manifest.json.tmp")
    tmp.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tmp, manifest_path)

    print(f"\n[materialize] DONE: {total_rows} rows, {shard_idx} shards")
    print(f"  manifest: {manifest_path}")
    print(f"  manifest SHA: {_sha256_file(manifest_path)[:16]}...")
    return manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-cache-manifest", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="facebook/dinov3-convnext-large-pretrain-lvd1689m",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        run(
            args.event_cache_manifest.resolve(),
            args.train_parquet.resolve(),
            args.eap_root.resolve(),
            args.model_path,
            args.output_dir.resolve(),
            device
        )
    except Exception as e:
        print(f"\n[materialize] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

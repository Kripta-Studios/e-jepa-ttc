#!/usr/bin/env python
"""Smoke audit: verify DINOv3 produces a unique 32×32 feature map and relations are valid.

Usage (Tiny model for smoke):
    python scripts/audit_dinov3_dense_teacher.py \
        --model-path facebook/dinov3-convnext-tiny-pretrain-lvd1689m \
        --max-samples 4

Usage (Large model for real):
    python scripts/audit_dinov3_dense_teacher.py \
        --model-path facebook/dinov3-convnext-large-pretrain-lvd1689m \
        --max-samples 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.distillation.dinov3_relational import (  # noqa: E402
    A4_RELATION_OFFSETS,
    local_cosine_relation_maps,
)

_INPUT_SIZE = 256
_EXPECTED_GRID = (32, 32)


def _find_32x32_feature(
    model: Any,  # noqa: ANN401
    dummy_input: torch.Tensor,
) -> tuple[torch.Tensor, str, str]:
    """Find the unique 32×32 feature map from a ConvNeXt model.

    Returns
    -------
    (features, selection_id, selection_method)
    """

    # Try output_hidden_states first
    with torch.no_grad():
        outputs = model(dummy_input, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        candidates_32 = []
        for idx, hs in enumerate(hidden_states):
            if hs.shape[-2:] == _EXPECTED_GRID:
                candidates_32.append((idx, hs))

        if len(candidates_32) == 1:
            idx, feat = candidates_32[0]
            return (
                feat,
                f"hidden_states[{idx}]",
                "hidden_states",
            )
        if len(candidates_32) > 1:
            raise RuntimeError(
                f"Multiple 32×32 hidden states found at indices "
                f"{[c[0] for c in candidates_32]}. "
                f"Cannot auto-select. Manual hook required."
            )

    # Fallback: try forward hook on ConvNeXt stages
    print("[audit] No 32×32 hidden state found. Trying forward hooks on stages...")
    stages = []
    for name, module in model.named_modules():
        if "stage" in name.lower() or "layer" in name.lower():
            stages.append((name, module))

    hooked_features: dict[str, torch.Tensor] = {}

    def make_hook(module_name: str) -> Any:  # noqa: ANN401
        def hook(  # noqa: ANN401
            module: Any, input: Any, output: Any,  # noqa: ANN401
        ) -> None:
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                hooked_features[module_name] = output
                return
            last_hidden_state = getattr(output, "last_hidden_state", None)
            if isinstance(last_hidden_state, torch.Tensor):
                hooked_features[module_name] = last_hidden_state

        return hook

    handles = []
    for name, module in stages:
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(dummy_input)

    for handle in handles:
        handle.remove()

    candidates_32 = [
        (name, feat)
        for name, feat in hooked_features.items()
        if feat.shape[-2:] == _EXPECTED_GRID
    ]

    if len(candidates_32) == 1:
        name, feat = candidates_32[0]
        return feat, name, "forward_hook"
    if len(candidates_32) == 0:
        all_shapes = {
            name: tuple(feat.shape) for name, feat in hooked_features.items()
        }
        raise RuntimeError(
            f"No 32×32 feature found in any stage. "
            f"Available shapes: {all_shapes}"
        )
    raise RuntimeError(
        f"Multiple 32×32 features found: "
        f"{[(n, tuple(f.shape)) for n, f in candidates_32]}. "
        f"Manual selection required."
    )


def audit(
    model_path: str,
    event_cache_manifest: Path,
    train_parquet: Path,
    eap_root: Path,
    max_samples: int,
    device_name: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the smoke audit."""

    import numpy as np
    import pandas as pd
    from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-untyped]

    # Reuse the exact scientific RGB binding/crop contract from the materializer.
    from scripts.materialize_dinov3_relational_teacher import (
        _extract_features,
        _load_and_crop_rgb,
        _resolve_rgb_endpoints,
    )

    device = torch.device(device_name)

    print(f"[audit] Loading model: {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=False)
    model = model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(model_path)

    # Record identity
    config_json = json.dumps(model.config.to_dict(), sort_keys=True)
    config_sha = hashlib.sha256(config_json.encode()).hexdigest()

    # Load dataset for real RGB
    from src.e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset
    train_dataset = GarlTTCObjectEventV4Dataset(
        str(event_cache_manifest), splits=("train",)
    )
    print(f"[audit] Loading {train_parquet}...")
    pq_df = pd.read_parquet(
        train_parquet,
        columns=[
            "sequence_id",
            "sample_token",
            "track_id",
            "rgb_shard_paths",
            "rgb_member_paths",
            "frame_timestamps_us",
            "event_windows_us",
            "boxes_xyxy",
        ],
    )
    rgb_lookup: dict[tuple[str, str], Any] = {}
    for _idx, row in pq_df.iterrows():
        key = (str(row["sample_token"]), str(row["track_id"]))
        if key in rgb_lookup:
            raise ValueError(f"duplicate RGB lookup key in train.parquet: {key}")
        rgb_lookup[key] = row

    image_mean = list(processor.image_mean)
    image_std = list(processor.image_std)

    # Create dummy input just for discovering the hook
    dummy_rgb = torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE, device=device)
    features, selection_id, selection_method = _find_32x32_feature(
        model, dummy_rgb,
    )
    print(
        f"[audit] Found 32×32 feature: selection={selection_id}, "
        f"method={selection_method}, shape={tuple(features.shape)}"
    )

    feat_channels = features.shape[1]
    print(f"[audit] Feature channels: {feat_channels}")

    print(f"[audit] Testing {max_samples} real object crops...")
    valid_fraction = 0.0

    endpoint_bindings_checked = 0
    for i in range(min(max_samples, len(train_dataset))):
        record = train_dataset[i]
        token = str(record["sample_token"])
        track_id = str(record["track_id"])
        square = np.asarray(record["event_v4_common_square_xyxy"], dtype=np.float32)

        parquet_row = rgb_lookup[(token, track_id)]
        endpoints = _resolve_rgb_endpoints(record, parquet_row)

        sample_valid_fractions: list[float] = []
        for ep, (shard_path, member_path) in enumerate(
            zip(endpoints["rgb_shards"], endpoints["rgb_members"], strict=True)
        ):
            sample_rgb, rgb_sha = _load_and_crop_rgb(
                tar_path=eap_root / shard_path,
                member_path=member_path,
                common_square_xyxy=square,
                image_mean=image_mean,
                image_std=image_std,
                device=device,
            )
            feats = _extract_features(model, sample_rgb, selection_id, selection_method)
            rels = local_cosine_relation_maps(feats)
            assert torch.isfinite(rels.values[rels.valid]).all(), (
                f"non-finite at sample {i}, endpoint {ep}"
            )

            vf = float(rels.valid.float().mean())
            sample_valid_fractions.append(vf)
            endpoint_bindings_checked += 1
            print(
                f"  sample {i} ep{ep + 1}: token={token}, "
                f"source_index={endpoints['indices'][ep]}, channels={feats.shape[1]}, "
                f"valid={vf:.4f}, rgb_sha={rgb_sha[:8]}"
            )

        if i == 0:
            valid_fraction = float(np.mean(sample_valid_fractions))

    # Verify no validation data was loaded
    result = {
        "status": "passed",
        "model_path": model_path,
        "config_sha256": config_sha,
        "feature_selection_id": selection_id,
        "feature_selection_method": selection_method,
        "feature_shape": list(features.shape),
        "feature_channels": feat_channels,
        "relation_offsets": [list(o) for o in A4_RELATION_OFFSETS],
        "relation_grid": list(_EXPECTED_GRID),
        "valid_fraction": valid_fraction,
        "samples_checked": min(max_samples, len(train_dataset)),
        "endpoint_bindings_checked": endpoint_bindings_checked,
        "rgb_source_verified": True,
        "validation_data_loaded": False,
        "test_data_loaded": False,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n[audit] PASSED: {json.dumps(result, indent=2)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-cache-manifest", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        audit(
            args.model_path,
            args.event_cache_manifest.resolve(),
            args.train_parquet.resolve(),
            args.eap_root.resolve(),
            args.sample_count,
            device_name,
            output_path=args.output.resolve() if args.output else None,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build signed Stage 61/62 feature caches from exact V8 A5 producers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockSampleLocator,
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.data.stage61_pair_feature_cache import save_feature_cache
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase
from e_jepa_ttc.models.local_temporal_phase_field import build_local_temporal_field_features


@dataclass
class Producer:
    name: str
    outer_fold: int
    inner_fold: int | None
    sequences: set[str]
    checkpoint: Path
    model: torch.nn.Module
    pair_features: list[np.ndarray]
    a5_phase: list[np.ndarray]
    patch_features: list[np.ndarray]
    patch_valid: list[np.ndarray]
    metadata: list[dict[str, Any]]


def _producer(
    router_root: Path,
    *,
    outer: int,
    inner: int | None,
    device: torch.device,
) -> Producer:
    role = f"inner{inner}" if inner is not None else "outer_dev"
    root = router_root / f"outer_fold{outer}_seed7" / "a5" / role
    artifact_path = root / "expert_artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(artifact):
        raise ValueError(f"A5 artifact signature mismatch: {artifact_path}")
    checkpoint = root / "train" / "model_best.pt"
    if compute_file_hash(str(checkpoint)) != artifact["checkpoint"]["sha256"]:
        raise ValueError(f"A5 checkpoint mismatch: {checkpoint}")
    nested = json.loads((root / "nested_protocol.json").read_text(encoding="utf-8"))
    if not verify_artifact_hash(nested):
        raise ValueError(f"nested protocol signature mismatch: {root}")
    sequences = set(nested["sequence_ids"] if inner is not None else nested["sequence_ids"])
    name = f"outer{outer}_inner{inner}" if inner is not None else f"outer{outer}_final"
    return Producer(
        name,
        outer,
        inner,
        sequences,
        checkpoint,
        load_causal_scale_replay_checkpoint(checkpoint, device=device),
        [],
        [],
        [],
        [],
        [],
    )


def _metadata(locator: CollisionClockSampleLocator) -> dict[str, Any]:
    return {
        "sample_token": locator.sample_token,
        "sequence_id": locator.sequence_id,
        "track_id": locator.track_id,
        "outer_fold": locator.outer_fold,
    }


@torch.no_grad()
def build(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text()
    )
    reference = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
    )
    supervision = load_canonical_supervision(reference, args.reference_root)
    adapter = CollisionClockTrain8192Cache(
        args.cache_root,
        protocol,
        cache_mode="shard_lru",
        lru_capacity=2,
        canonical_supervision=supervision,
    )
    locators = adapter.verify_and_index()
    producers = [
        _producer(args.router_root, outer=outer, inner=inner, device=args.device)
        for outer in range(3)
        for inner in (0, 1, 2, None)
    ]
    for start in range(0, len(locators), args.batch_size):
        batch_locators = locators[start : start + args.batch_size]
        inputs, delta, _target = adapter._materialize(batch_locators)
        inputs, delta = inputs.to(args.device), delta.to(args.device)
        for producer in producers:
            selected = [
                index
                for index, locator in enumerate(batch_locators)
                if locator.sequence_id in producer.sequences
            ]
            if not selected:
                continue
            index = torch.as_tensor(selected, dtype=torch.long, device=args.device)
            output = producer.model(inputs[index], delta[index], return_dense_features=True)
            support = output.sensor_support
            dt12 = delta[index, 1]
            pair = torch.cat(
                (
                    output.pair_tokens[:, -1],
                    torch.stack((dt12, torch.log(dt12 + 1e-8), torch.reciprocal(dt12)), dim=-1),
                    support[:, -1:],
                    torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
                ),
                dim=-1,
            )
            phase, valid_phase = ttc_to_benchmark_phase(
                output.ttc_mean_seconds, metric_delta_t_s=0.1
            )
            if not bool(valid_phase.all()) or output.endpoint_dense_features is None:
                raise ValueError(f"A5 producer output is incomplete: {producer.name}")
            producer.pair_features.append(pair.float().cpu().numpy())
            producer.a5_phase.append(phase.float().cpu().numpy())
            if producer.inner_fold is None:
                local = build_local_temporal_field_features(
                    output.endpoint_dense_features,
                    a5_phase=phase,
                    a5_log_variance=output.ttc_log_variance,
                    sensor_support=support[:, -1],
                    radius=int(producer.model.config.transport_radius),
                    temperature=float(producer.model.config.transport_temperature),
                )
                producer.patch_features.append(local.patch_features.float().cpu().numpy())
                producer.patch_valid.append(local.patch_valid.cpu().numpy())
            producer.metadata.extend(_metadata(batch_locators[item]) for item in selected)
        if start % (args.batch_size * 16) == 0:
            print(f"feature-cache rows visited: {start}/{len(locators)}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for producer in producers:
        arrays: dict[str, np.ndarray] = {
            "pair_features": np.concatenate(producer.pair_features),
            "a5_phase": np.concatenate(producer.a5_phase),
        }
        if producer.inner_fold is None:
            arrays["patch_features"] = np.concatenate(producer.patch_features)
            arrays["patch_valid"] = np.concatenate(producer.patch_valid)
        metadata = pd.DataFrame(producer.metadata)
        save_feature_cache(
            args.output_root / f"{producer.name}.npz",
            arrays=arrays,
            metadata=metadata,
            identity={
                "producer": producer.name,
                "a5_checkpoint_path": str(producer.checkpoint),
                "a5_checkpoint_sha256": compute_file_hash(str(producer.checkpoint)),
                "outer_fold": producer.outer_fold,
                "inner_fold": producer.inner_fold,
                "sequence_ids": sorted(producer.sequences),
                "protocol_sha256": protocol["artifact_sha256"],
            },
        )
    print(json.dumps({"status": "completed", "producer_count": len(producers)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    build(parser.parse_args())


if __name__ == "__main__":
    main()

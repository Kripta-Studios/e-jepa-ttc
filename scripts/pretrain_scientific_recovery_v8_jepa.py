#!/usr/bin/env python
"""Pretrain the label-free D2/D3/D4 V8 JEPA encoder for one outer fold.

This command consumes only the signed public-train V8 temporal cache.  It
intentionally has no TTC, box, mask or category argument; those values are
never placed on the JEPA model boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v8_cache import collate_scientific_recovery_v8  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v8_jepa_data import open_jepa_dataset  # noqa: E402
from e_jepa_ttc.models.causal_scale_jepa_v8 import (  # noqa: E402
    CausalScaleJEPAV8,
    CausalScaleJEPAV8Config,
    ordered_state_sha256,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.training.scientific_recovery_v8_jepa import (  # noqa: E402
    ScientificRecoveryV8JEPATrainer,
    ScientificRecoveryV8JEPATrainerConfig,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _write_signed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)
    return payload


class _IndexView(Dataset[dict[str, Any]]):
    def __init__(self, source: Dataset[dict[str, Any]], indices: Sequence[int]) -> None:
        self.source, self.indices = source, tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[self.indices[index]]


class _CrossTrackBatchSampler(Sampler[list[int]]):
    """Seeded batches where D4 always admits a complete cross-track matching."""

    def __init__(self, dataset: Dataset[dict[str, Any]], *, batch_size: int, seed: int) -> None:
        if batch_size < 2:
            raise ValueError("D4 requires batch_size >= 2")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            grouped[str(dataset[index]["track_id"])].append(index)
        if len(grouped) < 2:
            raise ValueError("D4 requires at least two outer-train tracks")
        self.batches: list[list[int]] = []
        rng = random.Random(seed)
        pools = {track: list(indices) for track, indices in grouped.items()}
        for values in pools.values():
            rng.shuffle(values)
        active = sorted(pools)
        cursor = 0
        current: list[int] = []
        while active:
            track = active[cursor % len(active)]
            current.append(pools[track].pop())
            if not pools[track]:
                active.remove(track)
                cursor = 0
            else:
                cursor += 1
            if len(current) == batch_size:
                self.batches.append(current)
                current = []
        if len(current) >= 2:
            counts: dict[str, int] = defaultdict(int)
            for index in current:
                counts[str(dataset[index]["track_id"])] += 1
            if max(counts.values()) <= len(current) - max(counts.values()):
                self.batches.append(current)
        if not self.batches:
            raise ValueError("D4 cross-track sampler could not form any feasible batch")

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def _endpoint_config(raw: dict[str, Any], *, channels: int) -> CausalScaleTTCConfig:
    model = raw.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("config.model must be a mapping")
    allowed = set(CausalScaleTTCConfig.__dataclass_fields__)
    values = {key: value for key, value in model.items() if key in allowed}
    values["in_channels"] = channels
    return CausalScaleTTCConfig(**values)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JEPA config must be a mapping")
    return value


def _winner_endpoint_config(path: Path, *, channels: int) -> CausalScaleTTCConfig:
    winner = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(winner, dict) or not verify_artifact_hash(winner):
        raise ValueError("winner artifact must be signed")
    raw = winner.get("downstream_model_config")
    if not isinstance(raw, dict):
        raise ValueError("winner artifact lacks downstream_model_config")
    allowed = set(CausalScaleTTCConfig.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed}
    values["in_channels"] = channels
    return CausalScaleTTCConfig(**values)


def run_pretraining(
    *,
    config_path: Path,
    cache_manifest: Path,
    fold: int,
    output_dir: Path,
    device: str,
    allow_fixture_cache: bool = False,
    endpoint_config_override: CausalScaleTTCConfig | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Run real label-free updates and emit signed checkpoint provenance."""

    raw = _load_config(config_path)
    experiment = raw.get("experiment", {})
    training = raw.get("jepa_pretrain", {})
    if not isinstance(experiment, dict) or not isinstance(training, dict):
        raise ValueError("JEPA config requires experiment and jepa_pretrain mappings")
    arm, seed = str(experiment.get("arm", "")), int(experiment.get("seed", 7))
    if arm not in {"D2", "D3", "D4"}:
        raise ValueError("pretrain command only accepts D2, D3 or D4 configs")
    shuffled = arm == "D4"
    manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_artifact_hash(manifest):
        raise ValueError("JEPA cache manifest must be a signed JSON artifact")
    dataset = open_jepa_dataset(
        cache_manifest=cache_manifest,
        protocol_path=protocol_path,
        allow_fixture_cache=allow_fixture_cache,
    )
    train_indices = [
        index for index in range(len(dataset)) if int(dataset[index]["outer_fold"]) != int(fold)
    ]
    if not train_indices:
        raise ValueError("outer-train pool is empty")
    view = _IndexView(dataset, train_indices)
    batch_size = int(training.get("batch_size", 32))
    sampler = _CrossTrackBatchSampler(view, batch_size=batch_size, seed=seed)
    loader = DataLoader(view, batch_sampler=sampler, collate_fn=collate_scientific_recovery_v8)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    endpoint_config = endpoint_config_override or _endpoint_config(raw, channels=dataset.shape[1])
    if endpoint_config.in_channels != dataset.shape[1]:
        raise ValueError("JEPA endpoint model channels differ from the signed cache representation")
    endpoint = CausalScaleTTC(endpoint_config)
    initial_encoder_sha = ordered_state_sha256(endpoint.encoder)
    objective = CausalScaleJEPAV8(
        endpoint,
        CausalScaleJEPAV8Config(
            ema_total_updates=int(training.get("total_updates", 1000)),
            predictor_hidden_dim=int(training.get("predictor_hidden_dim", 64)),
        ),
    ).to(device)
    trainer_config = ScientificRecoveryV8JEPATrainerConfig(
        learning_rate=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        total_updates=int(training.get("total_updates", 1000)),
        seed=seed,
        shuffled_future=shuffled,
    )
    trainer = ScientificRecoveryV8JEPATrainer(objective, trainer_config)
    schedule: list[list[str]] = []
    iterator = iter(loader)
    histories: list[dict[str, Any]] = []
    for _ in range(trainer_config.total_updates):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        schedule.append(list(batch.token_id))
        representations = batch.representations.to(device)
        result = trainer.step(
            representations[:, 0],
            representations[:, 1],
            representations[:, 2],
            track_ids=batch.track_id,
        )
        histories.append(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "jepa_checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    compute = trainer.compute_manifest()
    equal_config = {**trainer_config.__dict__, "shuffled_future": "controlled_future_pairing"}
    equal_compute_manifest = {**compute, "shuffled_future": "controlled_future_pairing"}
    checkpoint_manifest = _write_signed(
        output_dir / "jepa_checkpoint_manifest.json",
        {
            "artifact_type": "scientific_recovery_v8_jepa_checkpoint_manifest_v1",
            "status": "completed",
            "arm": arm,
            "fold": int(fold),
            "seed": seed,
            "cache_manifest_artifact_sha256": manifest["artifact_sha256"],
            "config_sha256": _file_sha(config_path),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _file_sha(checkpoint),
            "model_initialization_sha256": initial_encoder_sha,
            "batch_schedule_sha256": _sha(schedule),
            "trainer_config_sha256": _sha(equal_config),
            "full_trainer_config_sha256": trainer_config.sha256(),
            "compute_manifest": compute,
            "compute_manifest_sha256": _sha(equal_compute_manifest),
            "full_compute_manifest_sha256": _sha(compute),
            "total_updates": trainer.update_count,
            "shuffled_future": shuffled,
            "uses_labels": False,
            "health_last": histories[-1]["health"],
        },
    )
    return checkpoint_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--winner-artifact",
        type=Path,
        help="signed selected downstream architecture; required when config has no model mapping",
    )
    parser.add_argument(
        "--allow-fixture-cache",
        action="store_true",
        help="test-only escape hatch; production V8 execution rejects fixture caches",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "config": str(args.config),
                    "fold": args.fold,
                    "labels": "forbidden",
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        override = None
        if args.winner_artifact is not None:
            cache = ScientificRecoveryV8CacheDataset(args.cache_manifest)
            override = _winner_endpoint_config(args.winner_artifact, channels=cache.shape[1])
        result = run_pretraining(
            config_path=args.config,
            cache_manifest=args.cache_manifest,
            fold=args.fold,
            output_dir=args.output_dir,
            device=args.device,
            allow_fixture_cache=args.allow_fixture_cache,
            endpoint_config_override=override,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"V8 JEPA pretraining failed closed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {"status": result["status"], "artifact_sha256": result["artifact_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

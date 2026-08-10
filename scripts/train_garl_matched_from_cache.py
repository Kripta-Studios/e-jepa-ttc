"""Train official Garl event-only from scratch on the exact cached matched screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from scripts.run_garl_matched_screen import EXPECTED_RELEASE_COMMIT  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _release_state(release_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=release_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=release_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


@dataclass(frozen=True)
class CachedBatch:
    data: torch.Tensor
    target: torch.Tensor
    visible_height: torch.Tensor
    sample_tokens: tuple[str, ...]
    sequence_ids: tuple[str, ...]

    def to(self, device: torch.device) -> CachedBatch:
        return CachedBatch(
            data=self.data.to(device, non_blocking=True),
            target=self.target.to(device, non_blocking=True),
            visible_height=self.visible_height.to(device, non_blocking=True),
            sample_tokens=self.sample_tokens,
            sequence_ids=self.sequence_ids,
        )


def _collate(rows: list[dict[str, Any]]) -> CachedBatch:
    return CachedBatch(
        data=torch.stack([cast(torch.Tensor, row["data"]) for row in rows]),
        target=torch.stack([cast(torch.Tensor, row["target"]) for row in rows]),
        visible_height=torch.stack(
            [cast(torch.Tensor, row["visible_height"]) for row in rows]
        ),
        sample_tokens=tuple(str(row["sample_token"]) for row in rows),
        sequence_ids=tuple(str(row["sequence_id"]) for row in rows),
    )


class GarlMatchedTensorCache(Dataset[dict[str, Any]]):
    """Lazy one-shard cache dataset with exact shard-contiguous index groups."""

    def __init__(self, manifest_path: Path, split: str) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = _read_json(self.manifest_path)
        if self.manifest.get("artifact_type") != (
            "garl_official_event_only_matched_preprocessing_cache_v1"
        ):
            raise ValueError("Matched tensor cache has the wrong artifact type.")
        if split not in {"train", "validation"}:
            raise ValueError("Matched tensor cache split must be train or validation.")
        self.split = split
        self.shards = [
            shard
            for shard in self.manifest.get("shards", [])
            if isinstance(shard, dict) and shard.get("split") == split
        ]
        if not self.shards:
            raise ValueError(f"Matched tensor cache has no {split} shards.")
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += int(shard["rows"])
        if total != int(self.manifest["split_counts"][split]):
            raise ValueError(f"Matched tensor cache {split} count disagrees with manifest.")
        self.total = total
        self._loaded_index: int | None = None
        self._loaded: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.total

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(range(offset, offset + int(shard["rows"])))
            for offset, shard in zip(self.offsets, self.shards, strict=True)
        )

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total:
            raise IndexError(index)
        for shard_index in range(len(self.shards) - 1, -1, -1):
            if index >= self.offsets[shard_index]:
                return shard_index, index - self.offsets[shard_index]
        raise IndexError(index)

    def _load(self, shard_index: int) -> dict[str, Any]:
        if self._loaded_index == shard_index and self._loaded is not None:
            return self._loaded
        shard = self.shards[shard_index]
        path = self.manifest_path.parent / str(shard["path"])
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError(f"Matched tensor shard must be a mapping: {path}")
        self._loaded_index = shard_index
        self._loaded = loaded
        return loaded

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        payload = self._load(shard_index)
        return {
            "data": payload["data"][local],
            "target": payload["target"][local],
            "visible_height": payload["visible_height"][local],
            "sample_token": payload["sample_tokens"][local],
            "sequence_id": payload["sequence_ids"][local],
        }


class ShardGroupedSampler(Sampler[int]):
    def __init__(self, groups: tuple[tuple[int, ...], ...], generator: torch.Generator) -> None:
        self.groups = groups
        self.generator = generator
        flattened = sorted(index for group in groups for index in group)
        if not groups or flattened != list(range(len(flattened))):
            raise ValueError("Matched shard groups must partition dataset indices.")

    def __iter__(self) -> Iterator[int]:
        for shard_index in torch.randperm(len(self.groups), generator=self.generator).tolist():
            group = self.groups[shard_index]
            for local_index in torch.randperm(len(group), generator=self.generator).tolist():
                yield group[local_index]

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups)


def _loader(
    dataset: GarlMatchedTensorCache,
    *,
    batch_size: int,
    train: bool,
    generator: torch.Generator | None,
) -> DataLoader[CachedBatch]:
    sampler = (
        ShardGroupedSampler(dataset.shard_index_groups(), cast(torch.Generator, generator))
        if train
        else None
    )
    return cast(
        DataLoader[CachedBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
        ),
    )


def _prediction(raw: torch.Tensor, delta_t_s: float) -> torch.Tensor:
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("Official height-ratio output must have shape [B,2].")
    return delta_t_s / (1.0 - raw[:, 0] / raw[:, 1])


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[CachedBatch],
    device: torch.device,
    delta_t_s: float,
) -> dict[str, Any]:
    model.eval()
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    tokens: list[str] = []
    sequences: list[str] = []
    started = time.perf_counter()
    for host_batch in loader:
        batch = host_batch.to(device)
        raw_output = model(batch.data)
        raw = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        if not isinstance(raw, torch.Tensor):
            raise TypeError("Official model returned a non-tensor output.")
        targets.append(batch.target.detach().cpu())
        predictions.append(_prediction(raw, delta_t_s).detach().cpu())
        tokens.extend(batch.sample_tokens)
        sequences.extend(batch.sequence_ids)
    elapsed = time.perf_counter() - started
    target = torch.cat(targets).numpy().astype(np.float64)
    prediction = torch.cat(predictions).numpy().astype(np.float64)
    sequence = np.asarray(sequences)
    return {
        "signed": signed_garl_metrics(target, prediction),
        "sequence_macro": sequence_macro_signed_metrics(target, prediction, sequence),
        "sample_tokens": tokens,
        "sequence_ids": sequences,
        "target_ttc_s": target.tolist(),
        "prediction_ttc_s": prediction.tolist(),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(target) / elapsed,
    }


def _selection(metrics: dict[str, Any]) -> tuple[float, float]:
    return (
        float(metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"]),
        float(metrics["signed"]["failure_rate_pct"]),
    )


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "loader_generator": generator.get_state(),
    }


def _restore_rng(payload: dict[str, Any], generator: torch.Generator) -> None:
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda"):
        torch.cuda.set_rng_state_all(payload["cuda"])
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    generator.set_state(payload["loader_generator"])


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", 0)
    return device


def train(
    *,
    release_root: Path,
    cache_manifest: Path,
    output_dir: Path,
    device_name: str = "cuda",
    seed: int = 7,
    epochs: int = 18,
    batch_size: int = 32,
    minimum_epochs: int = 8,
    early_stopping_patience: int = 5,
    maximum_runtime_hours: float = 4.5,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Train/select with public validation only and exact resumable state."""

    if not 1 <= minimum_epochs <= epochs:
        raise ValueError("minimum_epochs must lie in [1, epochs].")
    if early_stopping_patience <= 0 or not 0.0 < maximum_runtime_hours <= 4.5:
        raise ValueError("Invalid early stopping or runtime guard.")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must lie in [1, epochs].")
    state = _release_state(release_root)
    if state != {"commit": EXPECTED_RELEASE_COMMIT, "dirty": False}:
        raise RuntimeError(f"Official release is not the audited clean commit: {state}")
    manifest = _read_json(cache_manifest)
    if manifest.get("split_counts") != {"train": 2048, "validation": 2048}:
        raise ValueError("Matched cached training requires exact 2048/2048 rows.")
    config_path = Path(str(manifest["sources"]["materialized_config"]["path"]))
    if not config_path.is_absolute():
        config_path = cache_manifest.parent / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Materialized official config not found: {config_path}")
    release_resolved = str(release_root.resolve())
    if release_resolved not in sys.path:
        sys.path.insert(0, release_resolved)
    config_api = importlib.import_module("garl_ttc.config")
    model_api = importlib.import_module("garl_ttc.models")
    trainer_api = importlib.import_module("garl_ttc.engine.trainer")
    config = config_api.load_config(config_path)
    config["model"]["pretrained_ckpt_rgb"] = ""
    config["model"]["pretrained_ckpt_event"] = ""
    config["training_settings"]["batch_size"] = batch_size
    config["training_settings"]["total_epochs"] = epochs
    config_identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    protocol = {
        "release_commit": state["commit"],
        "cache_artifact_sha256": manifest.get("artifact_sha256"),
        "config_identity": config_identity,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "minimum_epochs": minimum_epochs,
        "early_stopping_patience": early_stopping_patience,
        "maximum_runtime_hours": maximum_runtime_hours,
    }
    protocol_identity = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")
    ).hexdigest()
    device = _resolve_device(device_name)
    trainer_api.seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    train_dataset = GarlMatchedTensorCache(cache_manifest, "train")
    validation_dataset = GarlMatchedTensorCache(cache_manifest, "validation")
    if not isinstance(train_dataset, Sized) or not isinstance(validation_dataset, Sized):
        raise TypeError("Matched cached datasets must expose length.")
    generator = torch.Generator().manual_seed(seed)
    train_loader = _loader(
        train_dataset, batch_size=batch_size, train=True, generator=generator
    )
    validation_loader = _loader(
        validation_dataset, batch_size=batch_size, train=False, generator=None
    )
    model = model_api.TTCNetwork(config, is_train=True).to(device)
    optimizer, scheduler = trainer_api.prepare_optimizer(model, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    last_path = state_dir / "last.pt"
    best_path = state_dir / "best.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_selection: tuple[float, float] | None = None
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    stale = 0
    elapsed_before = 0.0
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Matched resume state not found: {last_path}")
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("protocol_identity") != protocol_identity:
            raise ValueError("Matched resume protocol identity changed.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        history = list(saved["history"])
        best_selection = tuple(saved["best_selection"]) if saved["best_selection"] else None
        best_epoch = int(saved["best_epoch"])
        best_validation = saved["best_validation"]
        stale = int(saved["stale"])
        elapsed_before = float(saved["elapsed_seconds"])
        start_epoch = int(saved["epoch"]) + 1
        _restore_rng(saved["rng_state"], generator)
    started = time.perf_counter()
    status = "completed_max_epochs"
    delta_t_s = float(cast(Any, model).dT)
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        component_totals: dict[str, float] = {}
        examples = 0
        train_started = time.perf_counter()
        for host_batch in train_loader:
            batch = host_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, loss_dict, _ = model.forward_train(
                batch.data,
                batch.target,
                visible_height_target=batch.visible_height,
                mask_target=None,
                use_mask_supervison=None,
                epoch_idx=epoch,
            )
            if not loss_dict:
                raise RuntimeError("Official matched training returned no loss components.")
            loss_total = torch.stack(tuple(loss_dict.values())).sum()
            if not bool(torch.isfinite(loss_total)):
                raise FloatingPointError("Official matched training loss became non-finite.")
            loss_total.backward()
            optimizer.step()
            count = len(batch.target)
            examples += count
            component_totals["total"] = component_totals.get("total", 0.0) + float(
                loss_total.detach().cpu()
            ) * count
            for name, value in loss_dict.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(
                    value.detach().cpu()
                ) * count
        train_seconds = time.perf_counter() - train_started
        scheduler.step()
        validation = _evaluate(model, validation_loader, device, delta_t_s)
        selection = _selection(validation)
        eligible = epoch >= minimum_epochs
        improved = eligible and (best_selection is None or selection < best_selection)
        if improved:
            best_selection = selection
            best_epoch = epoch
            best_validation = validation
            stale = 0
        elif eligible:
            stale += 1
        epoch_record = {
            "epoch": epoch,
            "train": {
                **{name: value / examples for name, value in component_totals.items()},
                "elapsed_seconds": train_seconds,
                "samples_per_second": examples / train_seconds,
            },
            "validation": {
                "signed": validation["signed"],
                "sequence_macro": validation["sequence_macro"],
                "elapsed_seconds": validation["elapsed_seconds"],
                "samples_per_second": validation["samples_per_second"],
            },
            "selection": {
                "eligible": eligible,
                "improved": improved,
                "sequence_macro_MiD": selection[0],
                "failure_rate_pct": selection[1],
            },
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        elapsed = elapsed_before + time.perf_counter() - started
        payload = {
            "artifact_type": "garl_event_only_matched_cached_state_v1",
            "epoch": epoch,
            "config_identity": config_identity,
            "protocol_identity": protocol_identity,
            "protocol": protocol,
            "cache_artifact_sha256": manifest.get("artifact_sha256"),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_selection": list(best_selection) if best_selection else None,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "stale": stale,
            "elapsed_seconds": elapsed,
            "rng_state": _rng_state(generator),
        }
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        if elapsed >= maximum_runtime_hours * 3600.0:
            status = "stopped_runtime_guard"
            break
        if epoch >= minimum_epochs and stale >= early_stopping_patience:
            status = "completed_early_stopping"
            break
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            status = "interrupted_for_resume"
            break
    if best_validation is None or best_selection is None or not best_path.is_file():
        raise RuntimeError("Matched training produced no selectable validation checkpoint.")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    final_validation = _evaluate(model, validation_loader, device, delta_t_s)
    if _selection(final_validation) != best_selection:
        raise RuntimeError("Reloaded matched best checkpoint metrics changed.")
    predictions = pd.DataFrame(
        {
            "sample_token": final_validation["sample_tokens"],
            "sequence_id": final_validation["sequence_ids"],
            "target_ttc_s": final_validation["target_ttc_s"],
            "predicted_ttc_s": final_validation["prediction_ttc_s"],
        }
    )
    predictions_path = output_dir / "validation_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    model_path = output_dir / "model_best.pt"
    _atomic_torch_save(
        {
            "artifact_type": "garl_event_only_matched_cached_best_v1",
            "best_epoch": best_epoch,
            "best_selection": list(best_selection),
            "config_identity": config_identity,
            "protocol_identity": protocol_identity,
            "cache_artifact_sha256": manifest.get("artifact_sha256"),
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )
    total_elapsed = elapsed_before + time.perf_counter() - started
    report: dict[str, Any] = {
        "artifact_type": "garl_event_only_matched_cached_training_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "selection": {
            "source": "public_validation_only",
            "best_epoch": best_epoch,
            "sequence_macro_MiD": best_selection[0],
            "failure_rate_pct": best_selection[1],
            "minimum_epochs": minimum_epochs,
            "early_stopping_patience": early_stopping_patience,
        },
        "validation_metrics": {
            "signed": final_validation["signed"],
            "sequence_macro": final_validation["sequence_macro"],
        },
        "history": history,
        "timing": {
            "preprocessing_from_cache_manifest": manifest["timing"],
            "training_and_validation_elapsed_seconds": total_elapsed,
            "best_inference_elapsed_seconds": final_validation["elapsed_seconds"],
            "best_inference_samples_per_second": final_validation["samples_per_second"],
        },
        "resources": {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated(device) / (1024.0**2)
                if device.type == "cuda"
                else 0.0
            ),
            "checkpoint_size_bytes": model_path.stat().st_size,
        },
        "protocol": {
            "seed": seed,
            "epochs_maximum": epochs,
            "batch_size": batch_size,
            "precision": "fp32",
            "from_scratch": True,
            "pretrained_release_checkpoint_used": False,
            "official_model_loss_optimizer": True,
            "cached_official_preprocessing": True,
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "maximum_runtime_hours": maximum_runtime_hours,
            "bbox_oracle_crop_declared": True,
            "test_used_for_selection": False,
        },
        "artifacts": {
            "model_best": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": _sha256(predictions_path),
                "rows": len(predictions),
            },
        },
        "sources": {
            "release_commit": state["commit"],
            "cache_manifest": {
                "path": str(cache_manifest.resolve()),
                "sha256": _sha256(cache_manifest),
                "artifact_sha256": manifest.get("artifact_sha256"),
            },
            "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        },
        "sealed_sources": {
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
        },
        "sota_claim_authorized": False,
    }
    sign_artifact(report)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--maximum-runtime-hours", type=float, default=4.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = train(
            release_root=args.release_root,
            cache_manifest=args.cache_manifest,
            output_dir=args.output_dir,
            device_name=args.device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            minimum_epochs=args.minimum_epochs,
            early_stopping_patience=args.early_stopping_patience,
            maximum_runtime_hours=args.maximum_runtime_hours,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
        )
    except Exception as error:
        print(
            f"cached matched Garl training failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the unchanged official Garl network from a release-semantic cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garl_release_cache import (  # noqa: E402
    CACHE_SAMPLER_VERSION,
    GarlReleaseCacheDataset,
    GarlReleaseShardBatchSampler,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)

VARIANT_CONFIGS = {
    "event_only": "configs/ablation/event_lhr.yaml",
    "visual_only": "configs/ablation/visual_lhr.yaml",
    "rgbe_late_fusion": "configs/garl_ttc_eventdecoder.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _git_commit() -> str | None:
    """Return the checked-out commit without hiding a dirty worktree."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _cuda_usable() -> bool:
    """Handle partially masked CUDA environments used by CPU CI jobs."""

    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_usable():
        torch.cuda.manual_seed_all(seed)


def _resolve_variant_config(release_root: Path, variant: str) -> Path:
    try:
        relative = VARIANT_CONFIGS[variant]
    except KeyError as error:
        raise ValueError(f"Unsupported Garl cache variant: {variant!r}") from error
    path = (release_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official variant config is missing: {path}")
    return path


def _load_config(release_root: Path, variant: str) -> tuple[dict[str, Any], Path]:
    config_path = _resolve_variant_config(release_root, variant)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Official config is not a mapping: {config_path}")
    config = cast(dict[str, Any], payload)
    config["dataset"]["root"] = str((release_root / "data/eAP-dataset").resolve())
    config["dirs"]["output"] = str((release_root / "outputs/cache_training").resolve())
    model = config["model"]
    if isinstance(model, dict):
        for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
            value = model.get(key)
            if isinstance(value, str):
                model[key] = str((release_root / value).resolve())
    return config, config_path


def _cache_input(batch: Mapping[str, Any], variant: str) -> torch.Tensor:
    if variant == "event_only":
        return cast(torch.Tensor, batch["event_roi"])
    rgb_pair = cast(torch.Tensor, batch["rgb_pair"])
    rgb = rgb_pair.reshape(rgb_pair.shape[0], 6, 128, 128)
    if variant == "visual_only":
        return rgb
    if variant == "rgbe_late_fusion":
        event = cast(torch.Tensor, batch["event_roi"])
        return torch.cat((rgb, event), dim=1)
    raise ValueError(f"Unsupported Garl cache variant: {variant!r}")


def _cache_fields(variant: str) -> tuple[str, ...]:
    """Return only the compressed-cache arrays needed by one model variant."""

    common = {"ttc_s", "visible_height", "delta_t_s", "sequence_id", "sample_token"}
    if variant == "event_only":
        return tuple(sorted(common | {"event_q"}))
    if variant == "visual_only":
        return tuple(sorted(common | {"rgb_f16"}))
    if variant == "rgbe_late_fusion":
        return tuple(sorted(common | {"event_q", "rgb_f16"}))
    raise ValueError(f"Unsupported Garl cache variant: {variant!r}")


def _string_list(value: object, expected: int) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} string values, got {len(values)}.")
    return values


def _prediction_to_ttc(raw: torch.Tensor, mode: str, delta_t_s: float) -> np.ndarray:
    values = raw.detach().to(dtype=torch.float64, device="cpu").numpy().reshape(len(raw), -1)
    if mode == "height_ratio":
        ratio = values[:, 0] / values[:, 1]
        return delta_t_s / (1.0 - ratio)
    if mode == "height_ratio_direct":
        return delta_t_s / (1.0 - values[:, 0])
    if mode == "baseline":
        return values[:, 0]
    raise ValueError(f"Unsupported official prediction mode: {mode!r}")


def _evaluate_cache(
    model: torch.nn.Module,
    dataset: GarlReleaseCacheDataset,
    *,
    variant: str,
    device: torch.device,
    batch_size: int,
    max_validation_batches: int | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    config = cast(Any, model)
    mode = str(config.pred_mode)
    delta_t_s = float(config.dT)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            data = _cache_input(batch, variant).to(device, non_blocking=True)
            raw_output = model(data)
            raw = raw_output[0] if isinstance(raw_output, tuple) else raw_output
            if not isinstance(raw, torch.Tensor):
                raise TypeError("Official TTCNetwork returned a non-tensor prediction.")
            prediction = _prediction_to_ttc(raw, mode, delta_t_s)
            target = cast(torch.Tensor, batch["ttc_s"]).detach().cpu().numpy().reshape(-1)
            tokens = _string_list(batch["sample_token"], len(target))
            sequences = _string_list(batch["sequence_id"], len(target))
            rows.extend(
                {
                    "sample_token": token,
                    "sequence_id": sequence,
                    "target_ttc_s": float(target[index]),
                    "predicted_ttc_s": float(prediction[index]),
                }
                for index, (token, sequence) in enumerate(zip(tokens, sequences, strict=True))
            )
            if max_validation_batches is not None and batch_index + 1 >= max_validation_batches:
                break
    frame = pd.DataFrame(rows)
    target_values = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction_values = frame["predicted_ttc_s"].to_numpy(dtype=np.float64)
    metrics = {
        "signed_garl_metrics": signed_garl_metrics(target_values, prediction_values),
        "sequence_macro_signed_metrics": sequence_macro_signed_metrics(
            target_values,
            prediction_values,
            frame["sequence_id"].astype(str),
        ),
    }
    return metrics, frame


def train_one(
    *,
    release_root: Path,
    cache_manifest: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    device_name: str,
    max_batches: int | None,
    max_validation_batches: int | None,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or workers < 0:
        raise ValueError("epochs/batch_size must be positive and workers non-negative.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    if max_validation_batches is not None and max_validation_batches <= 0:
        raise ValueError("max_validation_batches must be positive when provided.")
    release_root = release_root.resolve()
    cache_manifest = cache_manifest.resolve()
    output_dir = output_dir.resolve()
    if not cache_manifest.is_file():
        raise FileNotFoundError(f"Cache manifest is missing: {cache_manifest}")
    started_at = _now()
    run_started = time.perf_counter()
    git_commit = _git_commit()
    config_hash = _sha256(_resolve_variant_config(release_root, variant))
    manifest_hash = _sha256(cache_manifest)

    sys.path.insert(0, str(release_root))
    from garl_ttc.engine.trainer import prepare_optimizer  # type: ignore[reportMissingImports]
    from garl_ttc.models import TTCNetwork  # type: ignore[reportMissingImports]

    _seed_everything(seed)
    config, config_path = _load_config(release_root, variant)
    config["training_settings"]["batch_size"] = batch_size
    config["training_settings"]["num_threads"] = workers
    config["training_settings"]["total_epochs"] = epochs
    cuda_available = _cuda_usable()
    device = torch.device(
        ("cuda" if device_name == "auto" and cuda_available else "cpu")
        if device_name == "auto"
        else device_name
    )
    if device.type == "cuda" and not cuda_available:
        raise RuntimeError("Requested CUDA for the cache trainer but CUDA is unavailable.")

    selected_fields = _cache_fields(variant)
    train_dataset = GarlReleaseCacheDataset(
        cache_manifest,
        split="train",
        fields=selected_fields,
    )
    validation_dataset = GarlReleaseCacheDataset(
        cache_manifest,
        split="validation",
        fields=selected_fields,
    )
    manifest_payload = train_dataset.manifest
    split_sha256 = manifest_payload.get("split_sha256")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    torch_cuda_version = torch.version.cuda if device.type == "cuda" else None
    train_sampler = GarlReleaseShardBatchSampler(train_dataset, batch_size=batch_size, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = TTCNetwork(config, is_train=True).to(device)
    optimizer, scheduler = prepare_optimizer(model, config)
    history: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_config.json",
        {
            "artifact_type": "garl_release_cache_training_config_v1",
            "experiment_id": f"garl_p0_public_train40_retraining_{variant}_seed{seed}",
            "run_id": output_dir.name,
            "variant": variant,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "device": str(device),
            "host": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch_cuda_version,
            "gpu_name": gpu_name,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "git_commit": git_commit,
            "start_time": started_at,
            "config_hash": config_hash,
            "dataset_manifest_hash": manifest_hash,
            "split_version": split_sha256,
            "protocol": "garl_signed_v1",
            "training_scope": "public_train40_retraining",
            "sampling_order_policy": CACHE_SAMPLER_VERSION,
            "release_root": release_root.as_posix(),
            "release_config": config_path.as_posix(),
            "release_config_sha256": config_hash,
            "cache_manifest": cache_manifest.as_posix(),
            "cache_manifest_sha256": manifest_hash,
            "cache_fields": list(selected_fields),
            "max_validation_batches": max_validation_batches,
            "uses_ttc_for_model_input": False,
            "uses_bbox_for_model_input": True,
            "bbox_protocol": "P0_oracle_bbox_roi",
        },
    )
    try:
        for epoch in range(1, epochs + 1):
            train_sampler.set_epoch(epoch - 1)
            model.train()
            loss_sum = 0.0
            batch_count = 0
            for batch_index, batch in enumerate(train_loader):
                data = _cache_input(batch, variant).to(device, non_blocking=True)
                target = cast(torch.Tensor, batch["ttc_s"]).to(device, non_blocking=True)
                visible_height = cast(torch.Tensor, batch["visible_height"]).to(
                    device, non_blocking=True
                )
                optimizer.zero_grad(set_to_none=True)
                net = cast(Any, model.module if hasattr(model, "module") else model)
                _, _, loss_dict, _ = net.forward_train(
                    data,
                    target,
                    visible_height_target=visible_height,
                    mask_target=None,
                    use_mask_supervison=[False] * len(target),
                    epoch_idx=epoch,
                )
                loss_values = [cast(torch.Tensor, value) for value in loss_dict.values()]
                if not loss_values:
                    raise RuntimeError("Official Garl forward_train returned no losses.")
                loss_total = torch.stack(loss_values).sum()
                if not torch.isfinite(loss_total):
                    raise FloatingPointError(f"Non-finite official loss at epoch {epoch}.")
                loss_total.backward()
                optimizer.step()
                loss_sum += float(loss_total.detach())
                batch_count += 1
                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss_mean": loss_sum / max(batch_count, 1),
                "train_batches": batch_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        checkpoint = output_dir / "ckpt.pth"
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, checkpoint)
        validation_metrics, predictions = _evaluate_cache(
            model,
            validation_dataset,
            variant=variant,
            device=device,
            batch_size=batch_size,
            max_validation_batches=max_validation_batches,
        )
        predictions_path = output_dir / "validation_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        result: dict[str, Any] = {
            "artifact_type": "garl_release_cache_training_run_v1",
            "status": (
                "completed_bounded_smoke"
                if max_batches is not None or max_validation_batches is not None
                else "completed"
            ),
            "variant": variant,
            "seed": seed,
            "epochs": epochs,
            "max_batches": max_batches,
            "max_validation_batches": max_validation_batches,
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": _sha256(checkpoint),
            "validation_predictions_path": predictions_path.as_posix(),
            "validation_predictions_sha256": _sha256(predictions_path),
            "validation_metrics": validation_metrics,
            "history": history,
            "experiment_id": f"garl_p0_public_train40_retraining_{variant}_seed{seed}",
            "run_id": output_dir.name,
            "release_root": release_root.as_posix(),
            "release_config": config_path.as_posix(),
            "release_config_sha256": config_hash,
            "cache_manifest": cache_manifest.as_posix(),
            "cache_manifest_sha256": manifest_hash,
            "device": str(device),
            "host": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch_cuda_version,
            "gpu_name": gpu_name,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "git_commit": git_commit,
            "start_time": started_at,
            "end_time": _now(),
            "config_hash": config_hash,
            "dataset_manifest_hash": manifest_hash,
            "split_version": split_sha256,
            "protocol": "garl_signed_v1",
            "training_scope": "public_train40_retraining",
            "sampling_order_policy": CACHE_SAMPLER_VERSION,
            "elapsed_seconds": time.perf_counter() - run_started,
            "uses_ttc_for_model_input": False,
            "uses_bbox_for_model_input": True,
            "bbox_protocol": "P0_oracle_bbox_roi",
            "negative_results_preserved": True,
        }
        _write_json(output_dir / "run.json", result)
        return result
    except BaseException as error:
        failure: dict[str, Any] = {
            "artifact_type": "garl_release_cache_training_failure_v1",
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "variant": variant,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "experiment_id": f"garl_p0_public_train40_retraining_{variant}_seed{seed}",
            "run_id": output_dir.name,
            "start_time": started_at,
            "end_time": _now(),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "dataset_manifest_hash": manifest_hash,
            "split_version": split_sha256 if "split_sha256" in locals() else None,
            "host": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if _cuda_usable() else None,
            "gpu_name": (torch.cuda.get_device_name(0) if _cuda_usable() else None),
            "sampling_order_policy": CACHE_SAMPLER_VERSION,
            "release_root": release_root.as_posix(),
            "release_config": config_path.as_posix(),
            "cache_manifest": cache_manifest.as_posix(),
            "max_batches": max_batches,
            "max_validation_batches": max_validation_batches,
            "elapsed_seconds": time.perf_counter() - run_started,
            "negative_result_preserved": True,
        }
        _write_json(output_dir / "FAILURE.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train_one(
        release_root=args.release_root,
        cache_manifest=args.cache_manifest,
        output_dir=args.output_dir,
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.num_workers,
        device_name=args.device,
        max_batches=args.max_batches,
        max_validation_batches=args.max_validation_batches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

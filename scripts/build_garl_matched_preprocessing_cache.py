"""Cache immutable official Garl preprocessing for the exact matched screen rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from scripts.run_garl_matched_screen import (  # noqa: E402
    EXPECTED_RELEASE_COMMIT,
    materialize_config,
)


class _OfficialDataset(Protocol):
    def __len__(self) -> int: ...

    def get_collate_fn(self) -> Callable[[list[Any]], Mapping[str, Any] | None]: ...


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


def _official_api(release_root: Path) -> tuple[Callable[..., dict[str, Any]], Callable[..., Any]]:
    resolved = str(release_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    config_module = importlib.import_module("garl_ttc.config")
    dataset_module = importlib.import_module("garl_ttc.datasets")
    return cast(Callable[..., dict[str, Any]], config_module.load_config), cast(
        Callable[..., Any], dataset_module.TTCEstimationDataset
    )


def _flatten_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence):
        raise TypeError("Official sample_token batch must be a sequence.")
    tokens: list[str] = []
    for item in value:
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, Sequence) and len(item) == 1 and isinstance(item[0], str):
            tokens.append(item[0])
        else:
            raise TypeError(f"Unsupported official sample token: {item!r}")
    return tokens


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_shard(
    *,
    destination: Path,
    split: str,
    shard_index: int,
    data: list[torch.Tensor],
    target: list[torch.Tensor],
    visible_height: list[torch.Tensor],
    tokens: list[str],
    sequences: list[str],
) -> dict[str, Any]:
    shard_dir = destination / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard-{shard_index:05d}.pt"
    payload = {
        "data": torch.cat(data, dim=0).contiguous(),
        "target": torch.cat(target, dim=0).contiguous(),
        "visible_height": torch.cat(visible_height, dim=0).contiguous(),
        "sample_tokens": tuple(tokens),
        "sequence_ids": tuple(sequences),
    }
    rows = len(tokens)
    if payload["data"].shape != (rows, 40, 128, 128):
        raise ValueError(f"Official cached input shape mismatch: {payload['data'].shape}")
    if payload["target"].shape != (rows,) or payload["visible_height"].shape != (rows, 2):
        raise ValueError("Official cached target shapes are invalid.")
    if not bool(torch.isfinite(payload["data"]).all()):
        raise ValueError("Official cached input contains non-finite values.")
    if not bool(torch.isfinite(payload["target"]).all()):
        raise ValueError("Official cached TTC target contains non-finite values.")
    if not bool(torch.isfinite(payload["visible_height"]).all()):
        raise ValueError("Official cached visible height contains non-finite values.")
    _atomic_torch_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or tuple(loaded["sample_tokens"]) != tuple(tokens):
        raise RuntimeError(f"Official preprocessing shard roundtrip failed: {path}")
    return {
        "path": path.relative_to(destination).as_posix(),
        "split": split,
        "rows": rows,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "torch_load_verified": True,
    }


def _materialize_split(
    *,
    dataset: _OfficialDataset,
    split: str,
    destination: Path,
    expected: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    shard_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if shard_size % batch_size != 0:
        raise ValueError("shard_size must be a multiple of batch_size.")
    token_to_sequence = dict(
        zip(expected["sample_token"].astype(str), expected["sequence_id"].astype(str), strict=True)
    )
    token_to_target = dict(
        zip(expected["sample_token"].astype(str), expected["ttc"].astype(float), strict=True)
    )
    loader = DataLoader(
        cast(Dataset[Any], dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    shard_data: list[torch.Tensor] = []
    shard_target: list[torch.Tensor] = []
    shard_height: list[torch.Tensor] = []
    shard_tokens: list[str] = []
    shard_sequences: list[str] = []
    shards: list[dict[str, Any]] = []
    observed_tokens: list[str] = []
    maximum_target_error = 0.0
    started = time.perf_counter()
    for batch in loader:
        if batch is None:
            raise RuntimeError(f"Official preprocessing returned an empty {split} batch.")
        data = cast(torch.Tensor, batch["data"]).detach().cpu().float()
        target = cast(torch.Tensor, batch["target"]).detach().cpu().float().reshape(-1)
        visible_height = (
            cast(torch.Tensor, batch["visible_height"]).detach().cpu().float().reshape(-1, 2)
        )
        tokens = _flatten_tokens(batch["sample_token"])
        if not (len(tokens) == len(data) == len(target) == len(visible_height)):
            raise ValueError("Official preprocessed batch row counts disagree.")
        unknown = sorted(set(tokens) - set(token_to_sequence))
        if unknown:
            raise ValueError(
                f"Official preprocessing emitted unknown {split} tokens: {unknown[:3]}"
            )
        expected_target = np.asarray([token_to_target[token] for token in tokens], dtype=np.float64)
        current_error = float(
            np.max(np.abs(target.numpy().astype(np.float64) - expected_target), initial=0.0)
        )
        maximum_target_error = max(maximum_target_error, current_error)
        if current_error > 1e-5:
            raise ValueError(f"Official preprocessing target mismatch in {split}: {current_error}")
        shard_data.append(data)
        shard_target.append(target)
        shard_height.append(visible_height)
        shard_tokens.extend(tokens)
        shard_sequences.extend(token_to_sequence[token] for token in tokens)
        observed_tokens.extend(tokens)
        if len(shard_tokens) >= shard_size:
            if len(shard_tokens) != shard_size:
                raise ValueError("Official preprocessing batch crossed a shard boundary.")
            shards.append(
                _write_shard(
                    destination=destination,
                    split=split,
                    shard_index=len(shards),
                    data=shard_data,
                    target=shard_target,
                    visible_height=shard_height,
                    tokens=shard_tokens,
                    sequences=shard_sequences,
                )
            )
            shard_data, shard_target, shard_height = [], [], []
            shard_tokens, shard_sequences = [], []
    if shard_tokens:
        shards.append(
            _write_shard(
                destination=destination,
                split=split,
                shard_index=len(shards),
                data=shard_data,
                target=shard_target,
                visible_height=shard_height,
                tokens=shard_tokens,
                sequences=shard_sequences,
            )
        )
    elapsed = time.perf_counter() - started
    expected_tokens = set(expected["sample_token"].astype(str))
    if len(observed_tokens) != len(expected_tokens) or set(observed_tokens) != expected_tokens:
        raise ValueError(
            f"Official preprocessing {split} token mismatch: observed={len(observed_tokens)}, "
            f"expected={len(expected_tokens)}"
        )
    if len(set(observed_tokens)) != len(observed_tokens):
        raise ValueError(f"Official preprocessing {split} emitted duplicate tokens.")
    return shards, {
        "rows": len(observed_tokens),
        "sequence_count": len(set(expected["sequence_id"].astype(str))),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(observed_tokens) / elapsed,
        "maximum_target_absolute_error_s": maximum_target_error,
    }


def build_cache(
    *,
    release_root: Path,
    official_config: Path,
    subset_manifest: Path,
    eap_root: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    shard_size: int = 64,
    seed: int = 7,
) -> dict[str, Any]:
    """Execute official preprocessing once and bind every tensor shard by hash."""

    state_before = _release_state(release_root)
    if state_before != {"commit": EXPECTED_RELEASE_COMMIT, "dirty": False}:
        raise RuntimeError(f"Official release is not the audited clean commit: {state_before}")
    manifest = _read_json(subset_manifest)
    if manifest.get("artifact_type") != "garl_event_only_matched_screen_subset_v1":
        raise ValueError("Matched subset manifest has the wrong artifact type.")
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_config = output_dir / "official_preprocessing_config.yaml"
    materialize_config(
        official_config=official_config,
        subset_manifest=subset_manifest,
        release_root=release_root,
        eap_root=eap_root,
        output_config=materialized_config,
        output_dir=output_dir / "unused_training_output",
        epochs=18,
        batch_size=batch_size,
        num_workers=num_workers,
        minimum_selection_epoch=8,
    )
    load_config, dataset_type = _official_api(release_root)
    config = load_config(materialized_config)
    roles = cast(dict[str, Any], manifest["roles"])
    all_shards: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    initialization_seconds: dict[str, float] = {}
    for output_split, official_split in (("train", "train"), ("validation", "test")):
        role = cast(dict[str, Any], roles[output_split])
        data_path = subset_manifest.parent / role["data"]["path"]
        labels_path = subset_manifest.parent / role["labels"]["path"]
        expected = pd.read_parquet(data_path)[["sample_token", "sequence_id"]].merge(
            pd.read_parquet(labels_path)[["sample_token", "ttc"]],
            on="sample_token",
            validate="one_to_one",
        )
        initialized = time.perf_counter()
        dataset = cast(
            _OfficialDataset,
            dataset_type(config, official_split, seed=seed, db_mode="window"),
        )
        initialization_seconds[output_split] = time.perf_counter() - initialized
        if len(dataset) != int(role["rows"]):
            raise ValueError(f"Official {output_split} dataset length differs from matched role.")
        shards, report = _materialize_split(
            dataset=dataset,
            split=output_split,
            destination=output_dir,
            expected=expected,
            batch_size=batch_size,
            num_workers=num_workers,
            shard_size=shard_size,
        )
        all_shards.extend(shards)
        split_reports[output_split] = report
    state_after = _release_state(release_root)
    if state_after != state_before:
        raise RuntimeError("Official release changed during preprocessing cache construction.")
    result: dict[str, Any] = {
        "artifact_type": "garl_official_event_only_matched_preprocessing_cache_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_train_validation_only",
        "format": "torch_sharded_tensor_batches_v1",
        "input_contract": {
            "shape": [40, 128, 128],
            "dtype": "float32",
            "modality": "event_only",
            "representation": "official_garl_timevolume20_two_endpoints",
            "bbox_oracle_crop_used_in_preprocessing": True,
            "bbox_stored_as_model_input": False,
            "rgb_stored": False,
            "target_stored_as_model_input": False,
        },
        "parameters": {
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shard_size": shard_size,
        },
        "timing": {
            "dataset_initialization_seconds": initialization_seconds,
            "preprocessing_by_split": split_reports,
        },
        "shards": all_shards,
        "split_counts": {split: report["rows"] for split, report in split_reports.items()},
        "sources": {
            "release_commit": state_before["commit"],
            "release_dirty": state_before["dirty"],
            "official_config": {
                "path": str(official_config.resolve()),
                "sha256": _sha256(official_config),
            },
            "official_dataset_source": {
                "path": str((release_root / "garl_ttc/datasets/ttc_dataset.py").resolve()),
                "sha256": _sha256(release_root / "garl_ttc/datasets/ttc_dataset.py"),
            },
            "official_representation_source": {
                "path": str(
                    (release_root / "garl_ttc/datasets/event_representation.py").resolve()
                ),
                "sha256": _sha256(
                    release_root / "garl_ttc/datasets/event_representation.py"
                ),
            },
            "subset_manifest": {
                "path": str(subset_manifest.resolve()),
                "sha256": _sha256(subset_manifest),
                "artifact_sha256": manifest.get("artifact_sha256"),
            },
            "materialized_config": {
                "path": str(materialized_config.resolve()),
                "sha256": _sha256(materialized_config),
            },
        },
        "checks": {
            "exact_tokens": True,
            "targets_equal_public_labels": True,
            "all_tensors_finite": True,
            "torch_roundtrip": True,
            "release_unchanged": True,
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
        },
    }
    sign_artifact(result)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(r"E:\Garl-TTC"))
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    release_root = args.release_root
    official_config = args.official_config or (
        release_root / "configs" / "ablation" / "event_lhr.yaml"
    )
    try:
        result = build_cache(
            release_root=release_root,
            official_config=official_config,
            subset_manifest=args.subset_manifest,
            eap_root=args.eap_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shard_size=args.shard_size,
            seed=args.seed,
        )
    except Exception as error:
        print(
            f"official preprocessing cache failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

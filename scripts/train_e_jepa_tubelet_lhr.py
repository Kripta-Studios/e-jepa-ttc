"""Train the event-only high-resolution Tubelet TTC model from raw eAP streams.

This trainer intentionally bypasses the multi-hundred-GiB dense cache.  It joins
the public Garl-TTC labels to raw eAP event windows, materializes only the active
batch, selects checkpoints on the sequence-disjoint validation split, and emits
a checkpoint directly supported by the label-free Table-VI runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.data.garlttc_eap import load_garlttc_train_index  # noqa: E402
from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows  # noqa: E402
from e_jepa_ttc.models.highres_factorized import (  # noqa: E402
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
)
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from scripts.screen_highres_real_garl import (  # noqa: E402
    RawHighResGarlDataset,
    _git_commit,
    _json_safe,
    _loader,
    _metrics,
    _sha256,
)


@dataclass(frozen=True)
class TubeletTrainConfig:
    """Resource-bounded controls for supervised raw-event training."""

    epochs: int = 8
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    precision: str = "bf16"
    max_grad_norm: float = 1.0
    minimum_epochs: int = 2
    early_stopping_patience: int = 3
    temporal_steps: int = 5
    width: int = 320
    height: int = 192
    max_samples_per_split: int | None = 2048
    seed: int = 7
    run_scope: str = "bounded_screen"
    require_clean_git: bool = False

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_workers + 1,
            self.minimum_epochs,
            self.early_stopping_patience + 1,
            self.temporal_steps,
            self.width,
            self.height,
        )
        if min(integers) <= 0:
            raise ValueError("Tubelet trainer integer controls must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.max_samples_per_split is not None and self.max_samples_per_split <= 0:
            raise ValueError("max_samples_per_split must be positive when provided")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if self.run_scope not in {"bounded_screen", "full_candidate"}:
            raise ValueError("run_scope must be bounded_screen or full_candidate")
        if self.run_scope == "full_candidate" and self.max_samples_per_split is not None:
            raise ValueError("full_candidate cannot cap max_samples_per_split")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return cast(dict[str, Any], value)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(dict(value)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _resolve_ref(owner: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Expected a YAML path reference in {owner}")
    for candidate in (owner.parent / value, ROOT / value):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {value!r} referenced by {owner}")


def _load_inherited(path: Path) -> dict[str, Any]:
    value = _read_yaml(path)
    base_ref = value.get("base")
    if base_ref is None:
        return value
    merged = _load_inherited(_resolve_ref(path, base_ref))
    merged.update({key: item for key, item in value.items() if key != "base"})
    return merged


def load_training_spec(
    experiment_path: Path,
) -> tuple[EJEPATubeletLHRConfig, TubeletTrainConfig, dict[str, Any]]:
    """Resolve model/train references and reject the not-yet-wired RGB branch."""

    experiment = _read_yaml(experiment_path)
    model_path = _resolve_ref(experiment_path, experiment.get("model"))
    model_value = _load_inherited(model_path)
    if bool(model_value.get("use_rgb", False)):
        raise NotImplementedError(
            "RGB-E fusion is not implemented by the event-only Tubelet trainer; "
            "use an event-only config instead of silently dropping RGB."
        )
    model_name = str(model_value.get("model", ""))
    if not model_name.startswith("e_jepa_tubelet_lhr"):
        raise ValueError(f"Unsupported Tubelet model declaration: {model_name!r}")
    model_fields = {field.name for field in fields(EJEPATubeletLHRConfig)}
    model_config = EJEPATubeletLHRConfig(
        **{key: item for key, item in model_value.items() if key in model_fields}
    )

    train_value: dict[str, Any] = {}
    if experiment.get("finetuning") is not None:
        train_value = _load_inherited(_resolve_ref(experiment_path, experiment["finetuning"]))
    train_fields = {field.name for field in fields(TubeletTrainConfig)}
    train_config = TubeletTrainConfig(
        **{key: item for key, item in train_value.items() if key in train_fields}
    )
    return (
        model_config,
        train_config,
        {
            "experiment_path": experiment_path.resolve().as_posix(),
            "model_path": model_path.as_posix(),
            "model_declaration": model_name,
            "event_only": True,
        },
    )


def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return value.sign() * torch.log1p(value.abs())


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def train_epoch(
    model: EJEPATubeletLHR,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
    gradient_accumulation_steps: int = 1,
    scaler: Any | None = None,  # noqa: ANN401
) -> float:
    """Train one signed-TTC epoch without retaining dataset tensors."""

    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)
    for batch_index, (inputs, targets, _, _) in enumerate(loader):
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        with _autocast(device, precision):
            prediction = model(inputs).ttc_mean_seconds
            loss = functional.smooth_l1_loss(
                _signed_log1p(prediction), _signed_log1p(targets), beta=0.05
            )
            backward_loss = loss / gradient_accumulation_steps
        update = (
            batch_index + 1
        ) % gradient_accumulation_steps == 0 or batch_index + 1 == batch_count
        if scaler is None:
            backward_loss.backward()
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            scaler.scale(backward_loss).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("Tubelet training epoch produced no batches")
    return float(np.mean(losses))


@torch.no_grad()
def _predict(
    model: EJEPATubeletLHR,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, str, str]],
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Return signed TTC predictions from the model's native regression head."""

    model.eval()
    rows: list[dict[str, object]] = []
    for inputs, targets, sequence_ids, sample_tokens in loader:
        output = model(inputs.to(device=device, non_blocking=True))
        predictions = output.ttc_mean_seconds.detach().cpu().numpy()
        target_values = targets.detach().cpu().numpy()
        rows.extend(
            {
                "sample_token": str(sample_token),
                "sequence_id": str(sequence_id),
                "target_ttc_s": float(target),
                "prediction_ttc_s": float(prediction),
            }
            for prediction, target, sequence_id, sample_token in zip(
                predictions,
                target_values,
                sequence_ids,
                sample_tokens,
                strict=True,
            )
        )
    return pd.DataFrame.from_records(rows)


def _load_pretrained(model: EJEPATubeletLHR, path: Path | None) -> dict[str, Any]:
    """Load only an exact Dense Level--Dynamics backbone transfer payload.

    SSL projection heads, EMA target state and the aligned-patch predictor are
    intentionally not inference state.  Accepting a partial key intersection here
    would make a random or incompatible initialization look pretrained.
    """

    if path is None:
        return {"used": False, "transferred_keys": []}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Pretraining checkpoint must contain a mapping")
    if payload.get("artifact_type") != "dense_level_dynamics_jepa_checkpoint_v1":
        raise ValueError(
            "Pretraining checkpoint is not a Dense Level-Dynamics JEPA v1 artifact; "
            "legacy pooled or generic model checkpoints cannot be transferred."
        )
    raw_state = payload.get("online_encoder_state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError("Pretraining checkpoint is missing online_encoder_state_dict")
    structural_config = payload.get("online_encoder_config")
    if not isinstance(structural_config, Mapping):
        structural_config = payload.get("backbone_structural_config")
    if not isinstance(structural_config, Mapping):
        raise ValueError(
            "Pretraining checkpoint is missing its exact online_encoder_config; "
            "cannot validate a backbone-only transfer."
        )
    report = model.load_exact_backbone_state_dict(raw_state, structural_config)
    return {
        "used": True,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        **report,
    }


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def train_tubelet_garl(
    *,
    eap_root: Path,
    garlttc_root: Path,
    split_path: Path,
    output_dir: Path,
    model_config: EJEPATubeletLHRConfig,
    train_config: TubeletTrainConfig,
    provenance: Mapping[str, Any],
    device_name: str = "auto",
    pretrained: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train and select a cache-free event-only Tubelet candidate."""

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    device = resolve_device(device_name)
    dirty_at_start = _git_dirty()
    if train_config.require_clean_git and dirty_at_start:
        raise RuntimeError("full_candidate requires a clean committed Git worktree")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_version = split.get("version", split.get("protocol", split.get("artifact_type")))
    roles = {
        str(sequence): role
        for role in ("train", "validation")
        for sequence in split["assignments"][role]
    }
    if set(split["assignments"]["train"]) & set(split["assignments"]["validation"]):
        raise ValueError("Train and validation sequence assignments overlap")
    index = load_garlttc_train_index(garlttc_root, sorted(roles))
    dataset_manifest_hash = _canonical_hash(
        {
            "garl_data_sha256": index.data_sha256,
            "garl_annotations_sha256": index.annotations_sha256,
            "split_sha256": _sha256(split_path),
        }
    )
    config_hash = _canonical_hash(
        {
            "model": asdict(model_config),
            "trainer": asdict(train_config),
            "provenance": dict(provenance),
        }
    )
    selected, selection_report = select_balanced_cache_rows(
        index.merged.sort_values(
            ["sequence_id", "timestamp_us", "track_id", "sample_token"], kind="mergesort"
        ),
        roles,
        seed=train_config.seed,
        max_samples_per_split=train_config.max_samples_per_split,
    )
    rows = {
        role: selected.loc[selected["sequence_id"].astype(str).map(roles.get) == role].copy()
        for role in ("train", "validation")
    }
    datasets = {
        role: RawHighResGarlDataset(
            rows[role],
            eap_root=eap_root,
            width=train_config.width,
            height=train_config.height,
            temporal_steps=train_config.temporal_steps,
        )
        for role in ("train", "validation")
    }
    validation_loader = _loader(
        datasets["validation"],
        batch_size=train_config.batch_size,
        shuffle=False,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )
    model = EJEPATubeletLHR(model_config).to(device)
    transfer = _load_pretrained(model, pretrained)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler(  # type: ignore[reportPrivateImportUsage]
        "cuda", enabled=(device.type == "cuda" and train_config.precision == "fp16")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "FAILURE.json"
    if not resume:
        failure_path.unlink(missing_ok=True)
    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    last_path = output_dir / "last.pt"
    if resume and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved.get("best_score", math.inf))
        best_epoch = int(saved.get("best_epoch", 0))
        stale = int(saved.get("stale", 0))
        saved_history = saved.get("history", [])
        if not isinstance(saved_history, list):
            raise ValueError("Resume checkpoint history must be a list")
        history = [dict(item) for item in saved_history if isinstance(item, Mapping)]

    try:
        for epoch in range(start_epoch, train_config.epochs + 1):
            # Epoch-derived ordering makes a resumed run reproduce the same
            # sample order without serializing DataLoader worker state.
            train_loader = _loader(
                datasets["train"],
                batch_size=train_config.batch_size,
                shuffle=True,
                seed=train_config.seed + epoch,
                num_workers=train_config.num_workers,
            )
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                precision=train_config.precision,
                max_grad_norm=train_config.max_grad_norm,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                scaler=scaler if scaler.is_enabled() else None,
            )
            predictions = _predict(model, validation_loader, device=device)
            metrics = _metrics(predictions)
            macro = metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"]
            score = float(macro)
            improved = math.isfinite(score) and score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            row = {
                "epoch": epoch,
                "train_signed_log_smooth_l1": train_loss,
                "validation_sequence_macro_paper_MiD_overall": score,
                "validation_paper_MiD_overall": metrics["paper_MiD_overall"],
                "validation_weighted_RTE_pct": metrics["weighted_RTE_pct"],
                "validation_failure_rate_pct": metrics["failure_rate_pct"],
            }
            history.append(cast(dict[str, Any], _json_safe(row)))
            checkpoint = {
                "artifact_type": "e_jepa_tubelet_lhr_checkpoint_v1",
                "architecture": "e_jepa_tubelet_lhr",
                "modality": "event_only",
                "model_config": asdict(model_config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "selection_metric": "validation_sequence_macro_paper_MiD_overall_signed_v1",
                "config_hash": config_hash,
                "dataset_manifest_hash": dataset_manifest_hash,
                "split_version": split_version,
                "seed": train_config.seed,
                "uses_evttc_for_selection": False,
                "uses_raw_eap_on_demand": True,
                "claim_eligible": False,
                "pretraining": transfer,
                "history": history,
            }
            _atomic_save(checkpoint, last_path)
            if improved:
                inference_checkpoint = dict(checkpoint)
                inference_checkpoint.pop("optimizer_state_dict")
                inference_checkpoint["validation_metrics"] = _json_safe(metrics)
                _atomic_save(inference_checkpoint, output_dir / "best.pt")
                predictions.to_csv(output_dir / "best_validation_predictions.csv", index=False)
            if (
                epoch >= train_config.minimum_epochs
                and stale >= train_config.early_stopping_patience
            ):
                break
    finally:
        for dataset in datasets.values():
            dataset.close()

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(
            "No checkpoint had finite sequence-macro MiD; increase validation coverage"
        )
    ended_at = datetime.now(UTC)
    summary_path = output_dir / "summary.json"
    summary = {
        "artifact_type": "e_jepa_tubelet_lhr_training_v1",
        "experiment_id": f"{output_dir.name}-seed{train_config.seed}",
        "run_name": output_dir.name,
        "status": "completed",
        "created_at": ended_at.isoformat(),
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": dirty_at_start,
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "split_version": split_version,
        "seed": train_config.seed,
        "host": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "provenance": dict(provenance),
        "split_path": split_path.resolve().as_posix(),
        "split_sha256": _sha256(split_path),
        "garl_data_sha256": index.data_sha256,
        "garl_annotations_sha256": index.annotations_sha256,
        "selection_report": selection_report,
        "split_counts": {role: len(frame) for role, frame in rows.items()},
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint": best_path.as_posix(),
        "best_checkpoint_sha256": _sha256(best_path),
        "checkpoint_path": best_path.as_posix(),
        "metrics_path": summary_path.as_posix(),
        "history": history,
        "uses_dense_disk_cache": False,
        "uses_evttc": False,
        "uses_labels_for_encoder_input": False,
        "modality": "event_only",
        "claim_eligible": False,
        "downstream_evaluation_eligible": (
            train_config.run_scope == "full_candidate" and not dirty_at_start
        ),
        "non_promotable_reason": (
            "bounded single-seed development screen; no official eAP test or EvTTC score"
            if train_config.run_scope == "bounded_screen"
            else "training-only candidate; requires multiseed freeze and external evaluation"
        ),
    }
    safe = cast(dict[str, Any], _json_safe(summary))
    summary_path.write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/splits/eap_pilot12_v1.json"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/e_jepa_garl_event_screen_v1.yaml"),
    )
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional second copy of the generated summary for a tracked metrics directory.",
    )
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        model_config, train_config, provenance = load_training_spec(args.config.resolve())
        overrides = {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "max_samples_per_split": args.max_samples_per_split,
        }
        if args.epochs is not None:
            overrides["minimum_epochs"] = min(train_config.minimum_epochs, args.epochs)
        train_config = replace(
            train_config,
            **{key: value for key, value in overrides.items() if value is not None},
        )
        result = train_tubelet_garl(
            eap_root=args.eap_root.resolve(),
            garlttc_root=args.garlttc_root.resolve(),
            split_path=args.split.resolve(),
            output_dir=args.output.resolve(),
            model_config=model_config,
            train_config=train_config,
            provenance=provenance,
            device_name=args.device,
            pretrained=args.pretrained.resolve() if args.pretrained else None,
            resume=args.resume,
        )
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "e_jepa_tubelet_lhr_training_failure_v1",
            "status": "failed",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

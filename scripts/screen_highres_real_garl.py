"""Screen S3/S4/S5 on raw eAP events joined to public Garl-TTC labels.

This is a bounded architecture screen, not a confirmation run.  It never opens
EvTTC and it never uses TTC, boxes, depth, categories, or labels to construct an
encoder input.  The public Garl TTC target is used only by the supervised screen
loss and by the signed validation metrics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.data.eap_representation import base_compatible_voxel, downsample_full_frame
from e_jepa_ttc.data.garlttc_eap import (
    load_garlttc_train_index,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.data.garlttc_lhr_cache import select_temporal_indices
from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows
from e_jepa_ttc.evaluation.garl_ttc_protocol import (
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig

ARMS: tuple[tuple[str, bool, str], ...] = (
    ("S3_R4_WINDOW_TEMPORAL", False, "block_causal"),
    ("S4_R4_WINDOW_MERGE_TEMPORAL", True, "block_causal"),
    ("S5_R4_WINDOW_MERGE_KDA", True, "kda"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _as_list(value: object) -> list[Any]:
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        value = as_py()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list-like field, got {type(value)!r}.")
    return list(value)


def _frame_ttc(row: Mapping[str, object], index: int) -> float:
    values = _as_list(row["frame_ttc"])
    value = float(str(values[index]))
    if not math.isfinite(value):
        raise ValueError("frame_ttc target is not finite.")
    if not -10.0 <= value <= 10.0:
        raise ValueError(f"frame_ttc target is outside [-10, 10]: {value}")
    return value


class RawHighResGarlDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    """Build high-resolution inputs directly from raw eAP HDF5 streams."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        eap_root: Path,
        width: int,
        height: int,
        temporal_steps: int,
    ) -> None:
        if temporal_steps < 2:
            raise ValueError("temporal_steps must be at least two.")
        self.rows = rows.reset_index(drop=True).copy(deep=True)
        self.eap_root = eap_root
        self.width = width
        self.height = height
        self.temporal_steps = temporal_steps
        self._readers: dict[Path, EAPEventReader] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _reader(self, events_path: str) -> EAPEventReader:
        resolved = resolve_eap_events_path(self.eap_root, events_path)
        reader = self._readers.get(resolved)
        if reader is None:
            reader = EAPEventReader(resolved)
            reader.open()
            self._readers[resolved] = reader
        return reader

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.rows.iloc[index].to_dict()
        frame_timestamps = [int(value) for value in _as_list(row["frame_timestamps_us"])]
        windows = normalize_event_windows_us(row["event_windows_us"])
        usable = min(len(frame_timestamps), len(windows))
        if usable < 2:
            raise ValueError("A high-resolution screen sample needs two aligned endpoints.")
        frame_timestamps = frame_timestamps[:usable]
        windows = windows[:usable]
        first, second, _ = select_temporal_indices(
            frame_timestamps,
            anchor_timestamp_us=int(row["timestamp_us"]),
            target_delta_t_s=0.1,
            tolerance_s=0.025,
            context_delta_t_s=0.1,
            context_tolerance_s=0.05,
        )
        start_us = int(windows[first][0])
        end_us = int(windows[second][1])
        if end_us <= start_us:
            raise ValueError("High-resolution screen event interval is not positive.")
        edges = np.linspace(start_us, end_us, self.temporal_steps + 1, dtype=np.int64)
        reader = self._reader(str(row["events_path"]))
        steps: list[torch.Tensor] = []
        for step in range(self.temporal_steps):
            step_start = int(edges[step])
            step_end = int(edges[step + 1])
            if step_end <= step_start:
                raise ValueError("Temporal subdivision produced an empty interval.")
            raw = reader.read_window(step_start, step_end)
            frame = downsample_full_frame(
                raw,
                sequence_id=str(row["sequence_id"]),
                start_us=step_start,
                end_us=step_end,
                width=self.width,
                height=self.height,
            )
            steps.append(base_compatible_voxel(frame, bins=5))
        inputs = torch.stack(steps).contiguous()
        target = torch.tensor(_frame_ttc(row, second), dtype=torch.float32)
        return inputs, target, str(row["sequence_id"]), str(row["sample_token"])

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self) -> None:
        self.close()


def _loader(
    dataset: RawHighResGarlDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[Any]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=False,
        generator=generator,
    )


def _model_config(*, merge: bool, temporal_mixer: str) -> EJEPATubeletLHRConfig:
    return EJEPATubeletLHRConfig(
        in_channels=21,
        embed_dim=32,
        patch_size=8,
        spatial_window=8,
        heads=4,
        spatial_depth=1,
        temporal_depth=1,
        temporal_mixer=temporal_mixer,
        merge_2x2=merge,
        # Dense candidate readout must consume the retained patches through a
        # fixed query set; historical mean-pool screens remain separate.
        pooling="query",
        global_attention=False,
        memory_budget_gb=12.0,
    )


def _run_epoch(
    model: EJEPATubeletLHR,
    head: nn.Module,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    head.train(training)
    losses: list[float] = []
    for inputs, targets, _, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            embedding = model(inputs).embedding
            prediction = head(embedding).squeeze(-1)
            loss = functional.smooth_l1_loss(prediction, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), max_grad_norm
                )
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("Screen epoch produced no batches.")
    return float(np.mean(losses))


@torch.no_grad()
def _predict(
    model: EJEPATubeletLHR,
    head: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    head.eval()
    rows: list[dict[str, object]] = []
    for inputs, targets, sequences, tokens in loader:
        prediction = head(model(inputs.to(device)).embedding).squeeze(-1).cpu().numpy()
        target_values = targets.numpy()
        rows.extend(
            {
                "sequence_id": str(sequence),
                "sample_token": str(sample_token),
                "target_ttc_s": float(target),
                "prediction_ttc_s": float(predicted),
            }
            for sequence, sample_token, target, predicted in zip(
                sequences, tokens, target_values, prediction, strict=True
            )
        )
    if not rows:
        raise RuntimeError("Screen prediction produced no rows.")
    return pd.DataFrame(rows)


def _finite_metric(value: object) -> float | None:
    number = float(str(value))
    return number if math.isfinite(number) else None


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return _finite_metric(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    target = predictions["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = predictions["prediction_ttc_s"].to_numpy(dtype=np.float64)
    sequence = predictions["sequence_id"].astype(str).to_numpy()
    signed = signed_garl_metrics(target, prediction)
    macro = sequence_macro_signed_metrics(target, prediction, sequence)
    signed["mae_s"] = float(np.mean(np.abs(target - prediction)))
    signed["median_ae_s"] = float(np.median(np.abs(target - prediction)))
    signed["sign_accuracy"] = float(np.mean(np.sign(target) == np.sign(prediction)))
    signed["sequence_macro"] = macro
    return signed


def _screen(
    *,
    eap_root: Path,
    garl_root: Path,
    split_path: Path,
    config_path: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    selection_cfg = config["selection"]
    seed = int(selection_cfg["seed"])
    max_samples = int(selection_cfg["max_samples_per_split"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    roles = {
        str(sequence): role
        for role in ("train", "validation")
        for sequence in split["assignments"][role]
    }
    index = load_garlttc_train_index(garl_root, sorted(roles))
    selected, selection_report = select_balanced_cache_rows(
        index.merged.sort_values(
            ["sequence_id", "timestamp_us", "track_id", "sample_token"], kind="mergesort"
        ),
        roles,
        seed=seed,
        max_samples_per_split=max_samples,
    )
    rows_by_role = {
        role: selected.loc[selected["sequence_id"].astype(str).map(roles.get) == role].copy()
        for role in ("train", "validation")
    }
    target_device = torch.device(device)
    rows: list[dict[str, Any]] = []
    for arm, merge, mixer in ARMS:
        torch.manual_seed(seed)
        model = EJEPATubeletLHR(_model_config(merge=merge, temporal_mixer=mixer)).to(target_device)
        torch.manual_seed(seed + 1)
        head = nn.Linear(32, 1).to(target_device)
        train_dataset = RawHighResGarlDataset(
            rows_by_role["train"],
            eap_root=eap_root,
            width=320,
            height=192,
            temporal_steps=int(selection_cfg["temporal_steps"]),
        )
        validation_dataset = RawHighResGarlDataset(
            rows_by_role["validation"],
            eap_root=eap_root,
            width=320,
            height=192,
            temporal_steps=int(selection_cfg["temporal_steps"]),
        )
        train_loader = _loader(
            train_dataset,
            batch_size=int(selection_cfg["batch_size"]),
            shuffle=True,
            seed=seed,
            num_workers=int(selection_cfg["num_workers"]),
        )
        validation_loader = _loader(
            validation_dataset,
            batch_size=int(selection_cfg["batch_size"]),
            shuffle=False,
            seed=seed,
            num_workers=int(selection_cfg["num_workers"]),
        )
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(selection_cfg["learning_rate"]),
            weight_decay=float(selection_cfg["weight_decay"]),
        )
        history: list[dict[str, float]] = []
        for epoch in range(int(selection_cfg["train_epochs"])):
            train_loss = _run_epoch(
                model,
                head,
                train_loader,
                optimizer=optimizer,
                device=target_device,
                max_grad_norm=float(selection_cfg["max_grad_norm"]),
            )
            validation = _predict(model, head, validation_loader, device=target_device)
            validation_metrics = _metrics(validation)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_smooth_l1": train_loss,
                    "validation_mae_s": float(validation_metrics["mae_s"]),
                }
            )
        predictions = _predict(model, head, validation_loader, device=target_device)
        predictions_path = output / f"{arm}.predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        metrics = _metrics(predictions)
        rows.append(
            {
                "arm": arm,
                "merge_2x2": merge,
                "temporal_mixer": mixer,
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "screen_head_parameters": sum(parameter.numel() for parameter in head.parameters()),
                "history": history,
                "validation_metrics": metrics,
                "prediction_count": int(len(predictions)),
                "prediction_sha256": _sha256(predictions_path),
                "global_attention_used": False,
                "readout": "query",
                "metrics_scope": "public_garl_eap_validation_short_screen",
                "selection_allowed": False,
            }
        )
        train_dataset.close()
        validation_dataset.close()
        del model, head, optimizer
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    s4 = next(item for item in rows if item["arm"] == "S4_R4_WINDOW_MERGE_TEMPORAL")
    s5 = next(item for item in rows if item["arm"] == "S5_R4_WINDOW_MERGE_KDA")
    s4_metrics = s4["validation_metrics"]
    s5_metrics = s5["validation_metrics"]
    s4_mid = _finite_metric(s4_metrics["paper_MiD_overall"])
    s5_mid = _finite_metric(s5_metrics["paper_MiD_overall"])
    s4_rte = _finite_metric(s4_metrics["weighted_RTE_pct"])
    s5_rte = _finite_metric(s5_metrics["weighted_RTE_pct"])
    s4_failure = float(s4_metrics["failure_rate_pct"])
    s5_failure = float(s5_metrics["failure_rate_pct"])
    if s4_mid is None or s5_mid is None or s4_rte is None or s5_rte is None:
        decision = "inconclusive_missing_signed_bucket"
    elif s5_mid <= s4_mid and s5_rte <= s4_rte and s5_failure <= s4_failure:
        decision = "no_regression_in_short_screen"
    else:
        decision = "regression_in_short_screen"
    result: dict[str, Any] = {
        "artifact_type": "highres_real_architecture_screen_v1",
        "schema_version": "v1",
        "evidence_type": "public_garl_eap_raw_event_screen",
        "code_commit": _git_commit(),
        "protocol_version": str(config["protocol_version"]),
        "protocol_sha256": _canonical_sha256(config),
        "created_at": datetime.now(UTC).isoformat(),
        "eap_root": eap_root.as_posix(),
        "garlttc_root": garl_root.as_posix(),
        "split_sha256": _sha256(split_path),
        "garl_data_sha256": index.data_sha256,
        "garl_annotations_sha256": index.annotations_sha256,
        "device": str(target_device),
        "input_source": "raw_eap_hdf5_downsample_full_frame",
        "input_resolution": [320, 192],
        "cache_resolution_not_used": [160, 90],
        "temporal_steps": int(selection_cfg["temporal_steps"]),
        "uses_evttc": False,
        "uses_ttc_for_encoder_input": False,
        "uses_boxes_for_encoder_input": False,
        "uses_depth_for_encoder_input": False,
        "uses_labels_for_training": True,
        "selection_allowed": False,
        "selection_report": selection_report,
        "split_counts": {role: int(len(rows_by_role[role])) for role in ("train", "validation")},
        "arms": rows,
        "s4_vs_s5": {
            "primary_metric": "paper_MiD_overall",
            "s4": s4_mid,
            "s5": s5_mid,
            "delta_s5_minus_s4": None if s4_mid is None or s5_mid is None else s5_mid - s4_mid,
            "decision": decision,
            "predeclared_rule": (
                "S5 must not worsen signed MiD, weighted RTE, or failure rate; "
                "this screen is not promotable until three seeds."
            ),
        },
        "status": "pass",
    }
    safe_result = _json_safe(result)
    if not isinstance(safe_result, dict):
        raise TypeError("Screen result did not normalize to a JSON object.")
    safe_result["artifact_sha256"] = _canonical_sha256(safe_result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "screen.json").write_text(
        json.dumps(safe_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return safe_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/splits/eap_pilot12_v1.json"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/highres_real_screen_v1.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/highres_real_screen_v1")
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = _screen(
            eap_root=args.eap_root.resolve(),
            garl_root=args.garlttc_root.resolve(),
            split_path=args.split.resolve(),
            config_path=args.config.resolve(),
            output=args.output.resolve(),
            device=args.device,
        )
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "highres_real_architecture_screen_failure_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

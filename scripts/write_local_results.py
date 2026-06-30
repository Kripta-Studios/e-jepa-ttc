"""Generate the local experiment summary from ignored metrics artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _metric(payload: dict[str, Any], split: str, metric: str) -> float:
    return float(payload["splits"][split]["metrics"][metric])


def _count(payload: dict[str, Any], split: str) -> int:
    split_payload = payload["splits"][split]
    for key in ("window_count", "count", "prediction_count", "label_count"):
        if key in split_payload:
            return int(split_payload[key])
    return 0


def _trivial_mean(payload: dict[str, Any], split: str, metric: str) -> float:
    return float(payload["predictors"]["mean_train_ttc"]["splits"][split]["metrics"][metric])


def _trivial_count(payload: dict[str, Any], split: str) -> int:
    return int(payload["predictors"]["mean_train_ttc"]["splits"][split]["count"])


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _tiny_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "path": path,
        "seed": int(payload["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "validation_mae": _metric(payload, "validation", "mae_s"),
        "test_mae": _metric(payload, "test", "mae_s"),
        "validation_rmse": _metric(payload, "validation", "rmse_s"),
        "test_rmse": _metric(payload, "test", "rmse_s"),
    }


def _reproduction_commands() -> tuple[str, str, str, str, str]:
    cache_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc cache voxel "
        "--manifest data/manifests/evttc_local.yaml "
        "--split data/splits/evttc_local.yaml "
        "--index data/cache/evttc_index.json "
        "--output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz "
        "--width 160 --height 90 --bins 5 --no-normalize --metadata-channels"
    )
    train_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc train tiny-cnn "
        "--cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz "
        "--output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 "
        "--epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto"
    )
    pretrain_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc pretrain jepa "
        "--cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz "
        "--output-dir artifacts/runs/jepa_voxel_160x90_b5_raw_meta_train_seed7 "
        "--epochs 160 --batch-size 128 --learning-rate 0.0005 --seed 7 --device auto "
        "--pretrain-splits train --validation-splits validation "
        "--variance-weight 1.0 --min-std 0.05"
    )
    finetune_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc train tiny-cnn "
        "--cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz "
        "--output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_jepa_seed7 "
        "--epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto "
        "--pretrained-encoder "
        "artifacts/runs/jepa_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt"
    )
    report_command = (
        ".\\.venv\\Scripts\\python.exe scripts/write_local_results.py "
        "--output docs/local_results.md"
    )
    return cache_command, train_command, pretrain_command, finetune_command, report_command


def build_report(root: Path = ROOT) -> str:
    metrics_dir = root / "artifacts" / "metrics"
    runs_dir = root / "artifacts" / "runs"
    features_dir = root / "artifacts" / "features"

    trivial = _load_json(metrics_dir / "trivial_baseline.json")
    event_rate = _load_json(metrics_dir / "event_rate_baseline.json")
    geometric = _load_json(metrics_dir / "geometric_baseline.json")
    normalized_cnn = _load_json(runs_dir / "tiny_cnn_voxel_160x90_b5_seed42" / "metrics.json")
    jepa_train = _load_json(runs_dir / "jepa_voxel_160x90_b5_raw_meta_train_seed7" / "metrics.json")
    jepa_train_ft = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_jepa_seed7" / "metrics.json"
    )
    jepa_train_ft_lr = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_jepa_seed7_lr1e4" / "metrics.json"
    )
    jepa_all = _load_json(runs_dir / "jepa_voxel_160x90_b5_raw_meta_all_seed7" / "metrics.json")
    jepa_all_ft = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_jepa_all_seed7" / "metrics.json"
    )
    cache_summary = _load_json(features_dir / "evttc_voxel_160x90_b5.summary.json")
    raw_cache_summary = _load_json(features_dir / "evttc_voxel_160x90_b5_raw_meta.summary.json")

    raw_meta_paths = sorted(
        runs_dir.glob("tiny_cnn_voxel_160x90_b5_raw_meta_seed*/metrics.json")
    )
    raw_meta_rows = [_tiny_row(path) for path in raw_meta_paths]
    raw_val_mean, raw_val_std = _mean_std([row["validation_mae"] for row in raw_meta_rows])
    raw_test_mean, raw_test_std = _mean_std([row["test_mae"] for row in raw_meta_rows])
    best_raw = min(raw_meta_rows, key=lambda row: row["validation_mae"])
    best_jepa_train_ft = min(
        [jepa_train_ft, jepa_train_ft_lr],
        key=lambda payload: _metric(payload, "validation", "mae_s"),
    )
    (
        cache_command,
        train_command,
        pretrain_command,
        finetune_command,
        report_command,
    ) = _reproduction_commands()

    lines = [
        "# Local Results",
        "",
        "Generated from local ignored artifacts under `artifacts/`.",
        "",
        "These numbers are local smoke evidence, not a benchmark claim. The split contains only",
        "three EvTTC `CCRs-1` speed sequences: train=`low-100`, validation=`medium-100`,",
        "test=`high-100`. Lower MAE is better.",
        "",
        "## Dataset And Caches",
        "",
        "- Indexed windows: 1230 total; train 335, validation 418, test 477.",
        "- Normalized voxel cache:",
        f"  `{cache_summary['output']}`, shape `{cache_summary['shape']}`,",
        f"  build time {_fmt(float(cache_summary['seconds']))} s.",
        "- Raw+metadata voxel cache:",
        f"  `{raw_cache_summary['output']}`, shape `{raw_cache_summary['shape']}`,",
        f"  build time {_fmt(float(raw_cache_summary['seconds']))} s.",
        f"- Mean events/window: {_fmt(float(raw_cache_summary['mean_events_per_window']))}.",
        "",
        "## Results",
        "",
        "| Method | Protocol | Validation MAE | Test MAE | Notes |",
        "| --- | --- | ---: | ---: | --- |",
        (
            "| Constant mean TTC | `ttc.csv` rows | "
            f"{_fmt(_trivial_mean(trivial, 'validation', 'mae_s'))} "
            f"(n={_trivial_count(trivial, 'validation')}) | "
            f"{_fmt(_trivial_mean(trivial, 'test', 'mae_s'))} "
            f"(n={_trivial_count(trivial, 'test')}) | Train split mean target. |"
        ),
        (
            "| Event-rate ridge | indexed windows | "
            f"{_fmt(_metric(event_rate, 'validation', 'mae_s'))} "
            f"(n={_count(event_rate, 'validation')}) | "
            f"{_fmt(_metric(event_rate, 'test', 'mae_s'))} "
            f"(n={_count(event_rate, 'test')}) | log count/rate features. |"
        ),
        (
            "| TinyCNN normalized voxel | indexed windows | "
            f"{_fmt(_metric(normalized_cnn, 'validation', 'mae_s'))} "
            f"(n={_count(normalized_cnn, 'validation')}) | "
            f"{_fmt(_metric(normalized_cnn, 'test', 'mae_s'))} "
            f"(n={_count(normalized_cnn, 'test')}) | seed 42. |"
        ),
        (
            "| TinyCNN raw+metadata | indexed windows | "
            f"{_fmt(raw_val_mean)} +/- {_fmt(raw_val_std)} | "
            f"{_fmt(raw_test_mean)} +/- {_fmt(raw_test_std)} | "
            f"5 seeds; best validation seed {best_raw['seed']}. |"
        ),
        (
            "| TinyCNN raw+metadata best-val | indexed windows | "
            f"{_fmt(best_raw['validation_mae'])} | {_fmt(best_raw['test_mae'])} | "
            f"seed {best_raw['seed']}, best epoch {best_raw['best_epoch']}. |"
        ),
        (
            "| JEPA train-only + TinyCNN | indexed windows | "
            f"{_fmt(_metric(best_jepa_train_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(best_jepa_train_ft, 'test', 'mae_s'))} | "
            "Self-supervised on train split only; best of lr 3e-4/1e-4. |"
        ),
        (
            "| JEPA all-splits + TinyCNN | indexed windows | "
            f"{_fmt(_metric(jepa_all_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_all_ft, 'test', 'mae_s'))} | "
            "Diagnostic only; uses validation/test events without labels. |"
        ),
        (
            "| Geometric bbox expansion | labeled frames only | "
            f"{_fmt(_metric(geometric, 'validation', 'mae_s'))} "
            f"(n={_count(geometric, 'validation')}) | "
            f"{_fmt(_metric(geometric, 'test', 'mae_s'))} "
            f"(n={_count(geometric, 'test')}) | Not directly comparable; uses bbox labels. |"
        ),
        "",
        "## JEPA Diagnostics",
        "",
        "| Pretrain scope | Best epoch | Best loss | Last train target std | "
        "Last validation target std | Downstream validation MAE | Downstream test MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| train only | "
            f"{jepa_train['best_epoch']} | {_fmt(float(jepa_train['best_loss']))} | "
            f"{_fmt(float(jepa_train['last']['train']['target_embedding_std']))} | "
            f"{_fmt(float(jepa_train['last']['validation']['target_embedding_std']))} | "
            f"{_fmt(_metric(best_jepa_train_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(best_jepa_train_ft, 'test', 'mae_s'))} |"
        ),
        (
            "| train+validation+test | "
            f"{jepa_all['best_epoch']} | {_fmt(float(jepa_all['best_loss']))} | "
            f"{_fmt(float(jepa_all['last']['train']['target_embedding_std']))} | "
            f"{_fmt(float(jepa_all['last']['validation']['target_embedding_std']))} | "
            f"{_fmt(_metric(jepa_all_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_all_ft, 'test', 'mae_s'))} |"
        ),
        "",
        "## Conclusion",
        "",
        "1. The geometric apparent-expansion baseline is the strongest local signal, but it uses",
        "   object labels and only evaluates labeled frames, so it is not a pure event-stream",
        "   model.",
        "2. On the indexed event-window protocol, the event-rate ridge baseline is the",
        "   strongest robust result on the held-out high-speed sequence.",
        "3. The CNN needs raw density information: normalized voxels underperform.",
        "   Raw+metadata improves sharply and can beat event-rate on validation for one seed,",
        "   but the five-seed mean remains behind event-rate on test and has high variance.",
        "4. JEPA/self-supervised pretraining is implemented and runs on GPU, but this local",
        "   train-only run does not improve downstream TTC. Even the all-splits diagnostic is",
        "   worse than the non-pretrained TinyCNN seed 7, so the limiting factor is not just",
        "   access to unlabeled validation/test event windows.",
        "5. With one training sequence, there is still not enough evidence to claim learned",
        "   visual event representations generalize across speeds. The next meaningful step is",
        "   a larger EvTTC subset and stronger multi-horizon JEPA rather than this tiny run.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        cache_command,
        train_command,
        pretrain_command,
        finetune_command,
        report_command,
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "local_results.md")
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(ROOT), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

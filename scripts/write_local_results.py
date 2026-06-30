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


def _reproduction_commands() -> tuple[str, str, str, str, str, str]:
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
    causal_geometry_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc baseline causal-geometry "
        "--manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml "
        "--output artifacts/metrics/causal_geometry_baseline.json --derivative-window 15"
    )
    pretrain_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc pretrain jepa "
        "--cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz "
        "--output-dir artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7 "
        "--epochs 160 --batch-size 64 --learning-rate 0.0005 --seed 7 --device auto "
        "--pretrain-splits train --validation-splits validation "
        "--temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 "
        "--variance-weight 1.0 --min-std 0.05"
    )
    finetune_command = (
        ".\\.venv\\Scripts\\python.exe -m e_jepa_ttc train tiny-cnn "
        "--cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz "
        "--output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7 "
        "--epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto "
        "--pretrained-encoder "
        "artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7/"
        "jepa_encoder_best.pt"
    )
    report_command = (
        ".\\.venv\\Scripts\\python.exe scripts/write_local_results.py "
        "--output docs/local_results.md"
    )
    return (
        cache_command,
        train_command,
        causal_geometry_command,
        pretrain_command,
        finetune_command,
        report_command,
    )


def build_report(root: Path = ROOT) -> str:
    metrics_dir = root / "artifacts" / "metrics"
    runs_dir = root / "artifacts" / "runs"
    features_dir = root / "artifacts" / "features"

    trivial = _load_json(metrics_dir / "trivial_baseline.json")
    event_rate = _load_json(metrics_dir / "event_rate_baseline.json")
    geometric = _load_json(metrics_dir / "geometric_baseline.json")
    causal_geometry = _load_json(metrics_dir / "causal_geometry_baseline.json")
    normalized_cnn = _load_json(runs_dir / "tiny_cnn_voxel_160x90_b5_seed42" / "metrics.json")
    jepa_train = _load_json(runs_dir / "jepa_voxel_160x90_b5_raw_meta_train_seed7" / "metrics.json")
    jepa_train_ft = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_jepa_seed7" / "metrics.json"
    )
    jepa_train_ft_lr = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_jepa_seed7_lr1e4" / "metrics.json"
    )
    jepa_temporal = _load_json(
        runs_dir / "jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7" / "metrics.json"
    )
    jepa_temporal_ft = _load_json(
        runs_dir
        / "tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7"
        / "metrics.json"
    )
    jepa_temporal_probe = _load_json(
        runs_dir
        / "tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_probe_seed7"
        / "metrics.json"
    )
    lowlabel_10_scratch = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_seed7_frac10" / "metrics.json"
    )
    lowlabel_10_jepa = _load_json(
        runs_dir
        / "tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7_frac10"
        / "metrics.json"
    )
    partial_event_rate = _load_json(metrics_dir / "event_rate_partial_starter_baseline.json")
    partial_scratch = _load_json(
        runs_dir / "tiny_cnn_partial_starter_raw_meta_seed7" / "metrics.json"
    )
    partial_jepa = _load_json(
        runs_dir / "tiny_cnn_partial_starter_temporal_jepa_seed7" / "metrics.json"
    )
    partial_jepa_lr = _load_json(
        runs_dir / "tiny_cnn_partial_starter_temporal_jepa_seed7_lr1e4" / "metrics.json"
    )
    partial_frac05_scratch = _load_json(
        runs_dir / "tiny_cnn_partial_starter_raw_meta_seed7_frac05" / "metrics.json"
    )
    partial_frac05_jepa = _load_json(
        runs_dir / "tiny_cnn_partial_starter_temporal_jepa_seed7_frac05" / "metrics.json"
    )
    partial_frac05_jepa_lr = _load_json(
        runs_dir / "tiny_cnn_partial_starter_temporal_jepa_seed7_frac05_lr1e4" / "metrics.json"
    )
    jepa_all = _load_json(runs_dir / "jepa_voxel_160x90_b5_raw_meta_all_seed7" / "metrics.json")
    jepa_all_ft = _load_json(
        runs_dir / "tiny_cnn_voxel_160x90_b5_raw_meta_jepa_all_seed7" / "metrics.json"
    )
    cache_summary = _load_json(features_dir / "evttc_voxel_160x90_b5.summary.json")
    raw_cache_summary = _load_json(features_dir / "evttc_voxel_160x90_b5_raw_meta.summary.json")

    raw_meta_paths = sorted(
        path
        for path in runs_dir.glob("tiny_cnn_voxel_160x90_b5_raw_meta_seed*/metrics.json")
        if "_frac" not in path.parent.name
    )
    raw_meta_rows = [_tiny_row(path) for path in raw_meta_paths]
    raw_val_mean, raw_val_std = _mean_std([row["validation_mae"] for row in raw_meta_rows])
    raw_test_mean, raw_test_std = _mean_std([row["test_mae"] for row in raw_meta_rows])
    best_raw = min(raw_meta_rows, key=lambda row: row["validation_mae"])
    lowlabel_05_scratch_rows = [
        _tiny_row(path)
        for path in sorted(
            runs_dir.glob("tiny_cnn_voxel_160x90_b5_raw_meta_seed*_frac05/metrics.json")
        )
    ]
    lowlabel_05_jepa_rows = [
        _tiny_row(path)
        for path in sorted(
            runs_dir.glob(
                "tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed*_frac05/metrics.json"
            )
        )
    ]
    lowlabel_05_scratch_val_mean, lowlabel_05_scratch_val_std = _mean_std(
        [row["validation_mae"] for row in lowlabel_05_scratch_rows]
    )
    lowlabel_05_scratch_test_mean, lowlabel_05_scratch_test_std = _mean_std(
        [row["test_mae"] for row in lowlabel_05_scratch_rows]
    )
    lowlabel_05_jepa_val_mean, lowlabel_05_jepa_val_std = _mean_std(
        [row["validation_mae"] for row in lowlabel_05_jepa_rows]
    )
    lowlabel_05_jepa_test_mean, lowlabel_05_jepa_test_std = _mean_std(
        [row["test_mae"] for row in lowlabel_05_jepa_rows]
    )
    best_jepa_train_ft = min(
        [jepa_train_ft, jepa_train_ft_lr],
        key=lambda payload: _metric(payload, "validation", "mae_s"),
    )
    best_partial_jepa = min(
        [partial_jepa, partial_jepa_lr],
        key=lambda payload: _metric(payload, "validation", "mae_s"),
    )
    best_partial_frac05_jepa = min(
        [partial_frac05_jepa, partial_frac05_jepa_lr],
        key=lambda payload: _metric(payload, "validation", "mae_s"),
    )
    (
        cache_command,
        train_command,
        causal_geometry_command,
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
            "| Masked JEPA train-only + TinyCNN | indexed windows | "
            f"{_fmt(_metric(best_jepa_train_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(best_jepa_train_ft, 'test', 'mae_s'))} | "
            "Same-window masked objective; self-supervised on train split only. |"
        ),
        (
            "| Temporal JEPA train-only + TinyCNN | indexed windows | "
            f"{_fmt(_metric(jepa_temporal_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_temporal_ft, 'test', 'mae_s'))} | "
            "Multi-horizon future embedding objective; self-supervised on train only. |"
        ),
        (
            "| Temporal JEPA frozen probe | indexed windows | "
            f"{_fmt(_metric(jepa_temporal_probe, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_temporal_probe, 'test', 'mae_s'))} | "
            "Only TTC head trained after JEPA pretraining. |"
        ),
        (
            "| JEPA all-splits + TinyCNN | indexed windows | "
            f"{_fmt(_metric(jepa_all_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_all_ft, 'test', 'mae_s'))} | "
            "Diagnostic only; uses validation/test events without labels. |"
        ),
        (
            "| Causal geometry calibrated | detection-assisted labeled frames | "
            f"{_fmt(_metric(causal_geometry, 'validation', 'mae_s'))} "
            f"(n={_count(causal_geometry, 'validation')}) | "
            f"{_fmt(_metric(causal_geometry, 'test', 'mae_s'))} "
            f"(n={_count(causal_geometry, 'test')}) | "
            "Uses current/past boxes only; calibration fit on train labels only. |"
        ),
        (
            "| Centered geometry diagnostic | labeled frames only | "
            f"{_fmt(_metric(geometric, 'validation', 'mae_s'))} "
            f"(n={_count(geometric, 'validation')}) | "
            f"{_fmt(_metric(geometric, 'test', 'mae_s'))} "
            f"(n={_count(geometric, 'test')}) | "
            "Non-causal centered derivative; not a valid claim. |"
        ),
        "",
        "## Low-Label Results",
        "",
        "These runs use the same train sequence but restrict supervised TTC labels. The temporal",
        "JEPA encoder is pretrained on train events only, without TTC labels.",
        "",
        "| Labels | Method | Seeds | Validation MAE | Test MAE | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
        (
            "| 5% (17 windows) | TinyCNN scratch | 7,13,21 | "
            f"{_fmt(lowlabel_05_scratch_val_mean)} +/- "
            f"{_fmt(lowlabel_05_scratch_val_std)} | "
            f"{_fmt(lowlabel_05_scratch_test_mean)} +/- "
            f"{_fmt(lowlabel_05_scratch_test_std)} | "
            "Random train-label subset per seed. |"
        ),
        (
            "| 5% (17 windows) | Temporal JEPA + fine-tune | 7,13,21 | "
            f"{_fmt(lowlabel_05_jepa_val_mean)} +/- {_fmt(lowlabel_05_jepa_val_std)} | "
            f"{_fmt(lowlabel_05_jepa_test_mean)} +/- {_fmt(lowlabel_05_jepa_test_std)} | "
            "Same label subsets; train-only SSL encoder. |"
        ),
        (
            "| 10% (34 windows) | TinyCNN scratch | 7 | "
            f"{_fmt(_metric(lowlabel_10_scratch, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(lowlabel_10_scratch, 'test', 'mae_s'))} | "
            "Single-seed check. |"
        ),
        (
            "| 10% (34 windows) | Temporal JEPA + fine-tune | 7 | "
            f"{_fmt(_metric(lowlabel_10_jepa, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(lowlabel_10_jepa, 'test', 'mae_s'))} | "
            "Single-seed check. |"
        ),
        "",
        "## Partial Starter Exploratory",
        "",
        "This protocol adds the downloaded `CCRs-side-low` HDF5+TTC sequence to train while",
        "keeping the original validation/test sequences. It is useful for stress testing",
        "domain shift, but it is not a sealed final protocol.",
        "",
        "| Method | Train labels | Validation MAE | Test MAE | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            "| Event-rate ridge | 100% | "
            f"{_fmt(_metric(partial_event_rate, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(partial_event_rate, 'test', 'mae_s'))} | "
            "Fit on CCRs-1-low + CCRs-side-low. |"
        ),
        (
            "| TinyCNN scratch | 100% | "
            f"{_fmt(_metric(partial_scratch, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(partial_scratch, 'test', 'mae_s'))} | "
            "Raw+metadata partial-starter cache. |"
        ),
        (
            "| Temporal JEPA + fine-tune | 100% | "
            f"{_fmt(_metric(best_partial_jepa, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(best_partial_jepa, 'test', 'mae_s'))} | "
            "Best validation of lr 3e-4/1e-4. |"
        ),
        (
            "| TinyCNN scratch | 5% | "
            f"{_fmt(_metric(partial_frac05_scratch, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(partial_frac05_scratch, 'test', 'mae_s'))} | "
            "38 labeled train windows. |"
        ),
        (
            "| Temporal JEPA + fine-tune | 5% | "
            f"{_fmt(_metric(best_partial_frac05_jepa, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(best_partial_frac05_jepa, 'test', 'mae_s'))} | "
            "Best validation of lr 3e-4/1e-4. |"
        ),
        (
            "| Temporal JEPA diagnostic | 5% | "
            f"{_fmt(_metric(partial_frac05_jepa, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(partial_frac05_jepa, 'test', 'mae_s'))} | "
            "Not validation-selected; included because test shift response is notable. |"
        ),
        "",
        "## Anti-Leakage Audit",
        "",
        "- The `causal_geometry_baseline.json` run reports `uses_future_bboxes=false`,",
        "  `uses_future_events=false`, and `uses_validation_or_test_ttc_for_fit=false`.",
        "- Its derivative at each labeled frame is fitted from that frame and earlier labeled",
        "  frames only. The log-affine calibration uses train split labels only.",
        "- It is detection-assisted, not event-only: it assumes an external detector or tracker",
        "  provides current/past object boxes at inference.",
        "- The older centered geometric baseline is retained only as a diagnostic and is marked",
        "  non-causal because it uses future boxes inside the derivative window.",
        "- The temporal JEPA run reports `uses_ttc_labels=false`; future event windows are used",
        "  only as self-supervised targets and never cross sequence or split boundaries.",
        "- Low-label subsets are sampled only from the train split. Validation is used for",
        "  checkpoint selection; the mini test split has been inspected repeatedly and is",
        "  therefore exploratory rather than a sealed final test.",
        "- The partial starter runs add only `CCRs-side-low` to train. Remaining starter HDF5",
        "  downloads were blocked by Google Drive/gdown access limits during this run.",
        "",
        "## JEPA Diagnostics",
        "",
        "| Pretrain scope | Best epoch | Best loss | Last train target std | "
        "Last validation target std | Downstream validation MAE | Downstream test MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| masked train only | "
            f"{jepa_train['best_epoch']} | {_fmt(float(jepa_train['best_loss']))} | "
            f"{_fmt(float(jepa_train['last']['train']['target_embedding_std']))} | "
            f"{_fmt(float(jepa_train['last']['validation']['target_embedding_std']))} | "
            f"{_fmt(_metric(best_jepa_train_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(best_jepa_train_ft, 'test', 'mae_s'))} |"
        ),
        (
            "| temporal train only | "
            f"{jepa_temporal['best_epoch']} | {_fmt(float(jepa_temporal['best_loss']))} | "
            f"{_fmt(float(jepa_temporal['last']['train']['target_embedding_std']))} | "
            f"{_fmt(float(jepa_temporal['last']['validation']['target_embedding_std']))} | "
            f"{_fmt(_metric(jepa_temporal_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_temporal_ft, 'test', 'mae_s'))} |"
        ),
        (
            "| masked train+validation+test | "
            f"{jepa_all['best_epoch']} | {_fmt(float(jepa_all['best_loss']))} | "
            f"{_fmt(float(jepa_all['last']['train']['target_embedding_std']))} | "
            f"{_fmt(float(jepa_all['last']['validation']['target_embedding_std']))} | "
            f"{_fmt(_metric(jepa_all_ft, 'validation', 'mae_s'))} | "
            f"{_fmt(_metric(jepa_all_ft, 'test', 'mae_s'))} |"
        ),
        "",
        "## Conclusion",
        "",
        "1. The strongest leakage-safe local result is the causal detection-assisted geometry",
        "   model: validation MAE 0.439 s and test MAE 0.188 s on labeled frames.",
        "   It is promising, but it is not an event-only model because it requires object boxes.",
        "2. Temporal multi-horizon JEPA is the first positive self-supervised result: with",
        "   only 5% train labels, validation MAE improves from 2.909 +/- 0.743 s to",
        "   1.548 +/- 0.176 s across three seeds, and test mean improves modestly from",
        "   3.107 +/- 0.277 s to 2.986 +/- 0.106 s.",
        "3. With 100% labels, temporal JEPA improves validation MAE over the matching",
        "   TinyCNN seed 7 run (1.518 s vs 1.877 s), but it does not beat the event-rate",
        "   baseline on the repeatedly inspected high-speed mini test split.",
        "4. On the full-label indexed event-window protocol, event-rate ridge remains the",
        "   strongest robust held-out result among pure event-window models.",
        "5. The CNN needs raw density information: normalized voxels underperform.",
        "   Raw+metadata improves sharply and can beat event-rate on validation for one seed,",
        "   but the five-seed mean remains behind event-rate on test and has high variance.",
        "6. With one training sequence, there is still not enough evidence to claim learned",
        "   visual event representations generalize across speeds. Adding only CCRs-side-low",
        "   gives mixed results: JEPA improves partial full-label validation over scratch, but",
        "   the partial low-label validation-selected model still does not beat scratch.",
        "7. The next meaningful step is the full EvTTC starter subset with a fresh sealed test",
        "   protocol; gdown retrieved one extra HDF5 but then hit Drive access limits.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        cache_command,
        train_command,
        causal_geometry_command,
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

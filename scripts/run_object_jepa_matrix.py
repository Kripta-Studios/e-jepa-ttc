"""Run a pre-registered matched scratch/Object-JEPA low-label matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.models.object_jepa import ObjectJEPAConfig
from e_jepa_ttc.training.object_jepa import (
    fine_tune_object_ttc,
    pretrain_object_event_jepa,
)
from e_jepa_ttc.utils.io import write_structured


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_provenance(summary_path: Path, expected_fingerprint: str | None = None) -> bool:
    try:
        import subprocess

        expected_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        )
        summary = _load(summary_path)
        if summary.get("git_commit") != expected_commit:
            return False

        if expected_fingerprint is not None:
            if summary.get("run_fingerprint") != expected_fingerprint:
                return False

        return True
    except Exception:
        return False


def _run_or_resume_pretraining(
    *,
    cache: Path,
    output: Path,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected_fingerprint = pretrain_object_event_jepa(
        cache_manifest_path=cache,
        output_dir=output,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        weight_decay=args.pretrain_weight_decay,
        seed=seed,
        device_name=args.device,
        embedding_dim=args.embedding_dim,
        feature_dim=args.feature_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        use_ego_actions=args.use_ego_actions,
        use_recurrence=args.use_recurrence,
        use_geometry=args.use_geometry,
        dry_run_fingerprint=True,
    )
    assert isinstance(expected_fingerprint, str)

    summary_path = output / "summary.json"
    if (
        args.resume
        and summary_path.is_file()
        and _check_provenance(summary_path, expected_fingerprint)
    ):
        return _load(summary_path)
    return pretrain_object_event_jepa(
        cache_manifest_path=cache,
        output_dir=output,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        weight_decay=args.pretrain_weight_decay,
        seed=seed,
        device_name=args.device,
        embedding_dim=args.embedding_dim,
        feature_dim=args.feature_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        use_ego_actions=args.use_ego_actions,
        use_recurrence=args.use_recurrence,
        use_geometry=args.use_geometry,
        dry_run_fingerprint=False,
    )


def _run_or_resume_finetuning(
    *,
    cache: Path,
    output: Path,
    seed: int,
    fraction: float,
    pretrained: Path | None,
    scratch_config: ObjectJEPAConfig | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected_fingerprint = fine_tune_object_ttc(
        cache_manifest_path=cache,
        output_dir=output,
        pretrained_checkpoint_path=pretrained,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.finetune_learning_rate,
        weight_decay=args.finetune_weight_decay,
        label_fraction=fraction,
        seed=seed,
        device_name=args.device,
        scratch_config=scratch_config,
        use_ego_actions=args.use_ego_actions,
        report_splits=tuple(args.report_splits),
        allow_final_test_evaluation=args.allow_final_test_evaluation,
        dry_run_fingerprint=True,
    )
    assert isinstance(expected_fingerprint, str)

    summary_path = output / "summary.json"
    if (
        args.resume
        and summary_path.is_file()
        and _check_provenance(summary_path, expected_fingerprint)
    ):
        return _load(summary_path)
    return fine_tune_object_ttc(
        cache_manifest_path=cache,
        output_dir=output,
        pretrained_checkpoint_path=pretrained,
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.finetune_learning_rate,
        weight_decay=args.finetune_weight_decay,
        label_fraction=fraction,
        seed=seed,
        device_name=args.device,
        scratch_config=scratch_config,
        use_ego_actions=args.use_ego_actions,
        report_splits=tuple(args.report_splits),
        allow_final_test_evaluation=args.allow_final_test_evaluation,
        dry_run_fingerprint=False,
    )


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    eval_split = summary.get("evaluation_split", "test")
    split_metrics = summary.get(eval_split, summary.get("validation", {}))
    regression = split_metrics["regression"]
    garl = split_metrics["garl_ttc"]
    conformal = split_metrics["conformal_90"]
    return {
        "initialization": summary["initialization"],
        "seed": int(summary["seed"]),
        "label_fraction": float(summary["label_fraction"]),
        "effective_label_count": int(summary["effective_label_count"]),
        "best_epoch": int(summary["best_epoch"]),
        "mae_s": float(regression["mae_s"]),
        "rmse_s": float(regression["rmse_s"]),
        "median_abs_error_s": float(regression["median_abs_error_s"]),
        "garl_weighted_mid": float(garl["weighted_mid"]),
        "garl_weighted_rte_pct": float(garl["weighted_rte_pct"]),
        "conformal_90_coverage": float(conformal["coverage"]),
        "conformal_90_mean_width_s": float(conformal["mean_width_s"]),
        "summary": str(summary["best_checkpoint"]).replace("object_ttc_best.pt", "summary.json"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["initialization"], row["label_fraction"]), []).append(row)
    payload: list[dict[str, Any]] = []
    metrics = (
        "mae_s",
        "rmse_s",
        "median_abs_error_s",
        "garl_weighted_mid",
        "garl_weighted_rte_pct",
        "conformal_90_coverage",
        "conformal_90_mean_width_s",
    )
    for (initialization, fraction), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "initialization": initialization,
            "label_fraction": fraction,
            "seeds": [int(row["seed"]) for row in group],
            "run_count": len(group),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            aggregate[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        payload.append(aggregate)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument(
        "--label-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.25, 1.0],
    )
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.05)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--predictor-depth", type=int, default=3)
    parser.add_argument("--predictor-heads", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-ego-actions", dest="use_ego_actions", action="store_false")
    parser.add_argument("--no-recurrence", dest="use_recurrence", action="store_false")
    parser.add_argument("--no-geometry", dest="use_geometry", action="store_false")
    parser.add_argument("--report-splits", nargs="+", default=["validation"])
    parser.add_argument("--allow-final-test-evaluation", action="store_true")
    parser.set_defaults(use_ego_actions=True, use_recurrence=True, use_geometry=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        pretrain_dir = args.output_dir / "pretrain" / f"seed-{seed}"
        pretraining = _run_or_resume_pretraining(
            cache=args.cache_manifest,
            output=pretrain_dir,
            seed=seed,
            args=args,
        )
        checkpoint = Path(pretraining["best_checkpoint"])
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scratch_config = ObjectJEPAConfig(**checkpoint_payload["model_config"])
        for fraction in args.label_fractions:
            fraction_name = f"{fraction:.4f}".rstrip("0").rstrip(".")
            for initialization, pretrained in (("scratch", None), ("jepa", checkpoint)):
                run_dir = (
                    args.output_dir
                    / "finetune"
                    / initialization
                    / f"fraction-{fraction_name}"
                    / f"seed-{seed}"
                )
                summary = _run_or_resume_finetuning(
                    cache=args.cache_manifest,
                    output=run_dir,
                    seed=seed,
                    fraction=fraction,
                    pretrained=pretrained,
                    scratch_config=scratch_config if pretrained is None else None,
                    args=args,
                )
                rows.append(_row(summary))
                write_structured(args.output_dir / "runs.json", {"runs": rows})
    result = {
        "protocol": "pre_registered_matched_low_label_object_jepa_v1",
        "cache_manifest": args.cache_manifest.as_posix(),
        "seeds": args.seeds,
        "label_fractions": args.label_fractions,
        "uses_ego_actions": args.use_ego_actions,
        "uses_recurrence": args.use_recurrence,
        "uses_geometry": args.use_geometry,
        "test_used_for_selection": False,
        "runs": rows,
        "aggregate": _aggregate(rows),
    }
    write_structured(args.output_dir / "matrix_summary.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

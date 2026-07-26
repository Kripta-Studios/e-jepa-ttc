"""Tests for the signed FlowMimic validation-pilot summary."""

import hashlib
import json
from pathlib import Path

from scripts.summarize_flowmimic_pilot import summarize_flowmimic_pilot


def _write_downstream(
    path: Path,
    *,
    fingerprint: str,
    mae: float,
    checkpoint_sha256: str | None,
) -> None:
    payload = {
        "cache_sha256": "a" * 64,
        "seed": 7,
        "run_fingerprint": fingerprint,
        "best_epoch": 3,
        "final_test_opened": False,
        "evaluation_splits": ["train", "validation"],
        "splits": {
            "validation": {
                "metrics": {
                    "mae_s": mae,
                    "mean_abs_relative_error_pct": mae * 10.0,
                    "rmse_s": mae * 1.2,
                    "median_abs_error_s": mae * 0.8,
                }
            }
        },
        "pretrained_encoder": (
            {"checkpoint_sha256": checkpoint_sha256} if checkpoint_sha256 is not None else None
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_flowmimic_summary_selects_validation_winner_and_checks_hashes(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch.json"
    _write_downstream(scratch, fingerprint="scratch", mae=0.4, checkpoint_sha256=None)
    variants: dict[str, tuple[Path, Path]] = {}
    for variant, mae, alignment, inverse in (
        ("E0", 0.35, 0.0, 0.0),
        ("E1", 0.25, 0.25, 0.0),
        ("E2", 0.32, 0.25, 0.1),
    ):
        checkpoint = tmp_path / f"{variant}.pt"
        checkpoint.write_bytes(variant.encode("ascii"))
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        pretrain = tmp_path / f"{variant}_pretrain.json"
        pretrain.write_text(
            json.dumps(
                {
                    "cache_sha256": "a" * 64,
                    "best_checkpoint": checkpoint.as_posix(),
                    "flowmimic_alignment_weight": alignment,
                    "flowmimic_inverse_ttc_weight": inverse,
                    "run_fingerprint": f"ssl-{variant}",
                    "best_epoch": 2,
                    "best_loss": 0.01,
                    "elapsed_seconds": 1.0,
                    "seed": 7,
                }
            ),
            encoding="utf-8",
        )
        downstream = tmp_path / f"{variant}_downstream.json"
        _write_downstream(
            downstream,
            fingerprint=f"downstream-{variant}",
            mae=mae,
            checkpoint_sha256=checkpoint_sha256,
        )
        variants[variant] = (pretrain, downstream)

    output = tmp_path / "summary.json"
    payload = summarize_flowmimic_pilot(
        scratch_metrics_path=scratch,
        variant_paths=variants,
        output_path=output,
    )

    assert payload["selection"]["best_validation_variant"] == "E1"
    assert payload["final_test_opened"] is False
    assert payload["promotable_claim"] is False
    assert len(payload["artifact_sha256"]) == 64
    row_by_id = {row["id"]: row for row in payload["rows"]}
    assert row_by_id["E1"]["mae_reduction_vs_e0_pct"] > 25.0

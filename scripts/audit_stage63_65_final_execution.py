"""Audit frozen outputs without training, model inference, or gate modification."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact  # noqa: E402
from e_jepa_ttc.artifacts.training_authorization import (  # noqa: E402
    read_signed,
    verify_output_bindings,
)
from e_jepa_ttc.evaluation.stage63_65 import validate_campaign_universe  # noqa: E402
from e_jepa_ttc.training.raw_time_residual import load_frozen_raw_endpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    output = root / "final_audit"
    output.mkdir(exist_ok=False)
    qa_evidence = {}
    qa_commands = {
        "pytest": [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "tests/test_stage63_final_reporting.py",
            "tests/test_stage63_essential_bundle.py",
        ],
        "ruff": [
            str(repo / ".venv/Scripts/ruff.exe"),
            "check",
            "scripts/audit_stage63_65_final_execution.py",
            "scripts/package_scientific_recovery_v9_stage63_65.py",
            "tests/test_stage63_final_reporting.py",
        ],
        "types": [
            str(repo / ".venv/Scripts/pyright.exe"),
            "--project",
            "pyright-stage63-65.json",
            "scripts/audit_stage63_65_final_execution.py",
            "scripts/package_scientific_recovery_v9_stage63_65.py",
        ],
        "diff_check": ["git", "diff", "--check"],
    }
    for name, command in qa_commands.items():
        qa_run = subprocess.run(
            command,
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        log_path = output / f"REPORTING_QA_{name}.log"
        log_path.write_text(qa_run.stdout + qa_run.stderr, encoding="utf-8")
        qa_evidence[name] = {
            "command": command,
            "exit_code": qa_run.returncode,
            "path": str(log_path),
            "sha256": compute_file_hash(str(log_path)),
        }
        if qa_run.returncode:
            raise RuntimeError(f"reporting QA failed: {log_path}")
    qa_path = output / "REPORTING_QA.json"
    qa_path.write_text(
        json.dumps(
            sign_artifact(
                {
                    "status": "passed",
                    "training_executed": False,
                    "evidence": qa_evidence,
                    "analysis_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                    ).strip(),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bound = [
        root / "CAMPAIGN_RESULT.json",
        root / "TRAINING_LOCK.json",
        root / "WALL_TIME_AMENDMENT.json",
        root / "RUN_LEDGER.jsonl",
    ]
    bound.append(qa_path)
    lock_sha = compute_file_hash(str(root / "TRAINING_LOCK.json"))
    results = {}
    for stage, path in (
        (64, root / "stage64/seed7/STAGE64_RESULT.json"),
        (65, root / "stage65/STAGE65_RESULT.json"),
    ):
        result = read_signed(path)
        verify_output_bindings(path.parent, result)
        assert result["training_lock_sha256"] == lock_sha
        results[stage] = result
        bound.append(path)
    checkpoints = []
    for path in sorted((root / "stage64/seed7").glob("outer*/S64-*/frozen_manifest.json")):
        _, checkpoint, manifest = load_frozen_raw_endpoint(path.parent, device=torch.device("cpu"))
        assert checkpoint["identity"]["training_lock_sha256"] == lock_sha
        checkpoints.append(
            {
                "path": manifest["checkpoint_path"],
                "sha256": manifest["checkpoint_sha256"],
                "updates": 3000,
            }
        )
        bound.append(path)
    assert len(checkpoints) == 12
    fits = read_signed(root / "stage65/ALL_RIDGE_FITS_FROZEN.json")["fits"]
    assert len(fits) == 6
    for fit in fits:
        path = Path(fit["path"])
        assert compute_file_hash(str(path)) == fit["sha256"]
        assert fit["training_lock_sha256"] == lock_sha
        with np.load(path, allow_pickle=False) as arrays:
            assert all(
                np.isfinite(arrays[name]).all()
                for name in arrays.files
                if arrays[name].dtype.kind in "fiu"
            )
        bound.append(path)
    ledger = [json.loads(line) for line in (root / "RUN_LEDGER.jsonl").read_text().splitlines()]
    last_freeze = max(v["time_ns"] for v in ledger if v["state"] == "stage64_endpoint_frozen")
    first_eval = min(v["time_ns"] for v in ledger if v["state"] == "stage64_arm_evaluating")
    assert last_freeze < first_eval
    assert results[65]["all_fits_frozen_before_outer_evaluation"] is True
    assert not any((root / f"stage64/seed{seed}").exists() for seed in (13, 23))
    canonical_path = args.stage61_worktree / (
        "artifacts/scientific_recovery_v9_stage61_stage62/stage61/aggregate_seed7/R2_oof.csv"
    )
    # Installed pandas supports round_trip; the older local pandas stubs omit it.
    canonical = pd.read_csv(canonical_path, float_precision=cast(Any, "round_trip"))
    bound.append(canonical_path)
    binding = pd.read_csv(root / "stage63/X3_RAW_BINDING_V2.csv")
    anchors = (
        binding.sort_values("window_id")
        .groupby("sample_token", as_index=False)
        .agg(
            target_anchor_us=("target_anchor_us", "last"),
            target_anchor_event_clock_us=("target_anchor_event_clock_us", "last"),
            observation_end_us=("observation_end_us", "max"),
        )
    )
    bucket_scores = []
    recomputed = {}
    for stage in (64, 65):
        stage_dir = root / ("stage64/seed7" if stage == 64 else "stage65")
        for path in sorted(stage_dir.glob("*_oof.csv")):
            frame = pd.read_csv(path, float_precision=cast(Any, "round_trip"))
            validate_campaign_universe(frame, canonical)
            assert not bool(np.asarray(frame["failure"]).any())
            target = frame.target_ttc_s.to_numpy(np.float64)
            prediction = frame.prediction_ttc_s.to_numpy(np.float64)
            assert np.isfinite(prediction).all()
            loss = 10000 * np.abs(np.log1p(-0.1 / target) - np.log1p(-0.1 / prediction))
            frame["prediction_phase"] = -np.log1p(-0.1 / prediction)
            frame["bucket"] = "uncovered"
            macro = 0.0
            for seq in sorted(frame.sequence_id.unique()):
                for name, lo, hi, weight in (
                    ("positive_small", 0, 3, 0.5),
                    ("positive_medium", 3, 6, 0.3),
                    ("positive_large", 6, 10, 0.1),
                    ("negative", -10, 0, 0.1),
                ):
                    mask = (frame.sequence_id.to_numpy() == seq) & (target > lo) & (target <= hi)
                    assert mask.any()
                    value = float(loss[mask].mean())
                    macro += weight * value / 9
                    frame.loc[mask, "bucket"] = name
                    bucket_scores.append(
                        {
                            "arm": path.stem.removesuffix("_oof"),
                            "sequence_id": seq,
                            "bucket": name,
                            "rows": int(mask.sum()),
                            "mean_mid": value,
                        }
                    )
            name = path.stem.removesuffix("_oof")
            expected = (
                results[64]["gates"]["scores"][name]["score"]
                if stage == 64
                else results[65]["scores_and_diagnostics"][name]["score"]["score"]
            )
            assert abs(macro - expected) < 1e-8
            recomputed[name] = {
                "score": macro,
                "declared": expected,
                "absolute_difference": abs(macro - expected),
            }
            frame = frame.merge(
                cast(pd.DataFrame, anchors), on="sample_token", validate="one_to_one"
            )
            if "fallback" not in frame:
                assert frame.selected_expert.isin([0, 1, 2]).all()
                frame["fallback"] = False  # Actual selector always chose a valid expert.
            frame.to_csv(output / path.name, index=False)
            for fold in range(3):
                frame.loc[frame.outer_fold == fold].to_csv(
                    output / f"{name}_outer{fold}.csv", index=False
                )
            bound.append(path)
    pd.DataFrame(bucket_scores).to_csv(output / "SCORES_BY_SEQUENCE_BUCKET.csv", index=False)
    draw_hashes = set()
    for stage, comparisons in (
        (64, results[64]["gates"]["comparisons"]),
        (65, results[65]["comparisons"]),
    ):
        for name, gate in comparisons.items():
            stats = gate["statistics"]
            draw_hashes.add(stats["draws_sha256"])
            threshold = (
                {"S64-STATE-L1": -1, "S64-COUNT-L1": -1, "S61-R2": -3}
                if stage == 64
                else {"S61-R2": -1, "S65-RISK8": -1, "RouterR": -3}
            )
            magnitude = stats["point_delta"] <= threshold.get(name, float("inf"))
            if stage == 64 and name == "S64-PERM-L1":
                magnitude = stats["point_delta"] < 0
            expected_gate = (
                magnitude and stats["ci95_high"] < 0 and stats["fraction_negative"] >= 0.95
            )
            assert gate["passed"] == expected_gate
            assert stats["valid_draws"] == 8192 and stats["validity_fraction"] >= 0.99
    assert len(draw_hashes) == 1
    source = repo / "scripts/run_scientific_recovery_v9_stage65.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    fit_lines = [
        n.lineno
        for n in ast.walk(run)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"
    ]
    replay_lines = [
        n.lineno
        for n in ast.walk(run)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_ce17_replay"
    ]
    assert min(fit_lines) < min(replay_lines)
    bound.extend(
        [
            source,
            args.handoff_root / "05_STAGE65_FULL_REGRET_ROUTER.md",
            args.handoff_root / "06_STATISTICS_AND_DATA_EXPOSURE.md",
            args.handoff_root / "10_RESULTS_CONTRACT.md",
        ]
    )
    audit = sign_artifact(
        {
            "status": "INTEGRITY_BLOCKED",
            "execution_status": "RISK_ROUTER_DEV_CANDIDATE",
            "training_executed_in_this_audit": False,
            "model_inference_in_this_audit": False,
            "scientific_final_updates_retained": 36000,
            "checkpoints": checkpoints,
            "stage65_fits_verified": 6,
            "independent_scores": recomputed,
            "all_endpoint_bytes_valid": True,
            "train_freeze_before_evaluate_verified": True,
            "gates_recomputed_from_frozen_statistics": True,
            "bootstrap_draws_sha256": next(iter(draw_hashes)),
            "acceptance_blockers": [
                "CE17/R2 equivalence replay occurred after ridge fits, contrary to document05.",
                "Mandatory pair-phase permutation/oracle/regret/sign diagnostics "
                "were absent at freeze.",
                "Sequence-only bootstrap and leave-one-sequence-out sensitivity "
                "were absent at freeze.",
            ],
            "code_evidence": {"fit_call_lines": fit_lines, "ce17_replay_call_lines": replay_lines},
            "interpretation": "Observed positive Stage65 gates do not establish "
            "protocol acceptance.",
            "not_repaired_retrospectively": True,
            "input_bindings": {
                str(p.resolve()): {"sha256": compute_file_hash(str(p)), "bytes": p.stat().st_size}
                for p in bound
            },
        }
    )
    (root / "FINAL_INTEGRITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"], "scores": recomputed}))


if __name__ == "__main__":
    main()

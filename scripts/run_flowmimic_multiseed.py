"""Run the frozen E0/E1 three-seed FlowMimic validation gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.utils.io import read_structured, write_structured


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def _require_clean_worktree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"]
    ).decode("utf-8")
    if status.strip():
        raise RuntimeError("Commit or remove worktree changes before starting the frozen gate.")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _validate_gate_inputs(config: dict[str, Any]) -> tuple[Path, str]:
    cache_path = Path(config["data"]["cache"])
    expected_hash = str(config["data"]["cache_sha256"])
    observed_hash = compute_file_hash(str(cache_path))
    if observed_hash != expected_hash:
        raise ValueError(f"Cache SHA-256 mismatch: expected {expected_hash}, got {observed_hash}.")
    protocol_version, protocol_sha256 = get_current_protocol_identity()
    if str(config["protocol"]["version"]) != protocol_version:
        raise ValueError("Configured protocol version differs from the verified protocol.")
    if str(config["protocol"]["sha256"]) != protocol_sha256:
        raise ValueError("Configured protocol SHA-256 differs from the verified protocol.")
    if config["protocol"].get("final_test_opened") is not False:
        raise ValueError("The validation gate must keep final_test_opened=false.")
    with np.load(cache_path, allow_pickle=False) as cache:
        physical_splits = set(cache["split"].astype(str).tolist())
        if "test" in physical_splits:
            raise ValueError("The gate cache physically contains test samples.")
        validation_sequences = set(
            cache["sequence_id"][cache["split"].astype(str) == "validation"].astype(str).tolist()
        )
    expected_validation = set(config["protocol"]["validation_sequences"])
    if validation_sequences != expected_validation:
        raise ValueError(
            f"Validation sequences differ: {sorted(validation_sequences)} != "
            f"{sorted(expected_validation)}."
        )
    return cache_path, observed_hash


def _pretrain_command(
    config: dict[str, Any],
    *,
    variant: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    pretrain = config["pretrain"]
    weights = config["experiment"]["variants"][variant]
    return [
        sys.executable,
        "-m",
        "e_jepa_ttc",
        "pretrain",
        "jepa",
        "--cache",
        str(config["data"]["cache"]),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(pretrain["epochs"]),
        "--batch-size",
        str(pretrain["batch_size"]),
        "--learning-rate",
        str(pretrain["learning_rate"]),
        "--seed",
        str(seed),
        "--device",
        str(pretrain["device"]),
        "--model",
        str(pretrain["model"]),
        "--navigation-mode",
        str(pretrain["navigation_mode"]),
        "--pretrain-splits",
        *[str(value) for value in config["data"]["train_splits"]],
        "--validation-splits",
        *[str(value) for value in config["data"]["validation_splits"]],
        "--temporal-horizons-ms",
        *[str(value) for value in pretrain["temporal_horizons_ms"]],
        "--max-target-slop-ms",
        str(pretrain["max_target_slop_ms"]),
        "--mask-ratio",
        str(pretrain["mask_ratio"]),
        "--block-count",
        str(pretrain["block_count"]),
        "--mask-mode",
        str(pretrain["mask_mode"]),
        "--ema-momentum",
        str(pretrain["ema_momentum"]),
        "--regularizer",
        str(pretrain["regularizer"]),
        "--variance-weight",
        str(pretrain["variance_weight"]),
        "--min-std",
        str(pretrain["min_std"]),
        "--dense-predictor",
        str(pretrain["dense_predictor"]),
        "--flowmimic-alignment-weight",
        str(weights["flowmimic_alignment_weight"]),
        "--flowmimic-inverse-ttc-weight",
        str(weights["flowmimic_inverse_ttc_weight"]),
        "--flowmimic-minimum-ttc-s",
        str(pretrain["flowmimic_minimum_ttc_s"]),
        "--flowmimic-maximum-ttc-s",
        str(pretrain["flowmimic_maximum_ttc_s"]),
    ]


def _downstream_command(
    config: dict[str, Any],
    *,
    seed: int,
    pretrained_checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    downstream = config["downstream"]
    command = [
        sys.executable,
        "-m",
        "e_jepa_ttc",
        "train",
        "tiny-cnn",
        "--cache",
        str(config["data"]["cache"]),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(downstream["epochs"]),
        "--batch-size",
        str(downstream["batch_size"]),
        "--learning-rate",
        str(downstream["learning_rate"]),
        "--seed",
        str(seed),
        "--device",
        str(downstream["device"]),
        "--model",
        str(downstream["model"]),
        "--navigation-mode",
        str(downstream["navigation_mode"]),
        "--pretrained-encoder",
        str(pretrained_checkpoint),
        "--train-splits",
        *[str(value) for value in config["data"]["train_splits"]],
        "--validation-splits",
        *[str(value) for value in config["data"]["validation_splits"]],
        "--evaluation-splits",
        *[str(value) for value in config["data"]["evaluation_splits"]],
    ]
    if bool(downstream["freeze_encoder"]):
        command.append("--freeze-encoder")
    return command


def _pretrain_complete(
    metrics_path: Path,
    *,
    commit: str,
    cache_sha256: str,
    seed: int,
    epochs: int,
    alignment_weight: float,
) -> bool:
    if not metrics_path.exists():
        return False
    payload = _read_json(metrics_path)
    checkpoint = Path(str(payload.get("best_checkpoint", "")))
    return bool(
        payload.get("git_commit") == commit
        and payload.get("cache_sha256") == cache_sha256
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("epochs", -1)) == epochs
        and float(payload.get("flowmimic_alignment_weight", -1.0)) == alignment_weight
        and checkpoint.is_file()
    )


def _downstream_complete(
    metrics_path: Path,
    *,
    commit: str,
    cache_sha256: str,
    seed: int,
    epochs: int,
    pretrained_sha256: str,
) -> bool:
    if not metrics_path.exists():
        return False
    payload = _read_json(metrics_path)
    checkpoint = Path(str(payload.get("best_checkpoint", "")))
    predictions = Path(str(payload.get("predictions_path", "")))
    pretrained = payload.get("pretrained_encoder") or {}
    return bool(
        payload.get("git_commit") == commit
        and payload.get("cache_sha256") == cache_sha256
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("epochs", -1)) == epochs
        and pretrained.get("checkpoint_sha256") == pretrained_sha256
        and checkpoint.is_file()
        and predictions.is_file()
        and payload.get("final_test_opened") is False
        and "test" not in set(payload.get("evaluation_splits", []))
    )


def _run(command: list[str], *, label: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] START {label}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)
    print(f"[{datetime.now(UTC).isoformat()}] DONE  {label}", flush=True)


def run_flowmimic_multiseed(
    *,
    config_path: Path,
    resume: bool,
    with_robustness: bool,
) -> None:
    """Execute or resume the frozen gate, then produce signed summaries."""

    _require_clean_worktree()
    config = read_structured(config_path)
    cache_path, cache_sha256 = _validate_gate_inputs(config)
    commit = _git_commit()
    variants = tuple(str(value) for value in config["experiment"]["variants"])
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    if variants != ("E0", "E1") or seeds != [7, 13, 21]:
        raise ValueError("Frozen runner requires E0/E1 with paired seeds 7/13/21.")
    run_root = Path(config["outputs"]["run_root"])
    ledger_path = run_root / "flowmimic_e0_e1_multiseed_30ep_ledger.json"
    ledger: dict[str, Any] = {
        "artifact_type": "flowmimic_multiseed_execution_ledger",
        "schema_version": "1.0",
        "started_at": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "config_path": config_path.as_posix(),
        "config_sha256": compute_file_hash(str(config_path)),
        "cache_path": cache_path.as_posix(),
        "cache_sha256": cache_sha256,
        "final_test_opened": False,
        "stages": [],
    }

    for variant in variants:
        for seed in seeds:
            stem = variant.lower()
            ssl_dir = run_root / f"flowmimic_full_{stem}_seed{seed}_ssl30"
            ft_dir = run_root / f"flowmimic_full_{stem}_seed{seed}_ft30"
            ssl_metrics = ssl_dir / "metrics.json"
            alignment_weight = float(
                config["experiment"]["variants"][variant]["flowmimic_alignment_weight"]
            )
            ssl_done = resume and _pretrain_complete(
                ssl_metrics,
                commit=commit,
                cache_sha256=cache_sha256,
                seed=seed,
                epochs=int(config["pretrain"]["epochs"]),
                alignment_weight=alignment_weight,
            )
            if not ssl_done:
                _run(
                    _pretrain_command(
                        config,
                        variant=variant,
                        seed=seed,
                        output_dir=ssl_dir,
                    ),
                    label=f"{variant} seed {seed} SSL",
                )
            ssl_payload = _read_json(ssl_metrics)
            ssl_checkpoint = Path(ssl_payload["best_checkpoint"])
            ssl_checkpoint_sha256 = compute_file_hash(str(ssl_checkpoint))
            ledger["stages"].append(
                {
                    "variant": variant,
                    "seed": seed,
                    "stage": "pretrain",
                    "status": "resumed" if ssl_done else "completed",
                    "metrics_path": ssl_metrics.as_posix(),
                    "run_fingerprint": ssl_payload["run_fingerprint"],
                    "checkpoint_sha256": ssl_checkpoint_sha256,
                }
            )
            write_structured(ledger_path, ledger)

            ft_metrics = ft_dir / "metrics.json"
            ft_done = resume and _downstream_complete(
                ft_metrics,
                commit=commit,
                cache_sha256=cache_sha256,
                seed=seed,
                epochs=int(config["downstream"]["epochs"]),
                pretrained_sha256=ssl_checkpoint_sha256,
            )
            if not ft_done:
                _run(
                    _downstream_command(
                        config,
                        seed=seed,
                        pretrained_checkpoint=ssl_checkpoint,
                        output_dir=ft_dir,
                    ),
                    label=f"{variant} seed {seed} downstream",
                )
            ft_payload = _read_json(ft_metrics)
            ledger["stages"].append(
                {
                    "variant": variant,
                    "seed": seed,
                    "stage": "downstream",
                    "status": "resumed" if ft_done else "completed",
                    "metrics_path": ft_metrics.as_posix(),
                    "run_fingerprint": ft_payload["run_fingerprint"],
                    "validation_mae_s": ft_payload["splits"]["validation"]["metrics"]["mae_s"],
                }
            )
            write_structured(ledger_path, ledger)

    summary_command = [
        sys.executable,
        str(Path("scripts") / "summarize_flowmimic_multiseed.py"),
        "--config",
        str(config_path),
        "--output",
        str(config["outputs"]["summary"]),
    ]
    _run(summary_command, label="paired multiseed summary")
    if with_robustness:
        robustness_command = [
            sys.executable,
            str(Path("scripts") / "evaluate_flowmimic_robustness.py"),
            "--config",
            str(config_path),
            "--output",
            str(config["outputs"]["robustness"]),
        ]
        _run(robustness_command, label="raw-event robustness suite")
        _run(
            [
                *summary_command,
                "--robustness",
                str(config["outputs"]["robustness"]),
            ],
            label="final paired multiseed summary",
        )
    ledger["completed_at"] = datetime.now(UTC).isoformat()
    ledger["status"] = "completed"
    ledger["robustness_complete"] = with_robustness
    write_structured(ledger_path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen E0/E1 30-epoch SSL and downstream gate."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--with-robustness", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    run_flowmimic_multiseed(
        config_path=args.config,
        resume=args.resume,
        with_robustness=args.with_robustness,
    )


if __name__ == "__main__":
    main()

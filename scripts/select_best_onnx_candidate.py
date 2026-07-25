import argparse
import datetime
import json
import math
import sys
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash, hash_dict, sign_artifact
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity


def _reject(run: Path, reason: str) -> None:
    print(
        json.dumps({"run": run.name, "status": "rejected", "reason": reason}),
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="artifacts/runs")
    parser.add_argument("--require-full-label", action="store_true")
    parser.add_argument("--require-commit", type=str, default=None)
    parser.add_argument("--require-navigation", type=str, default="enabled")
    parser.add_argument("--require-model-name", type=str, default="event-tubelet-transformer")
    args = parser.parse_args()

    actual_protocol_version, actual_protocol_hash = get_current_protocol_identity()

    runs_dir = Path(args.runs_dir)
    best_mae = float("inf")
    best_record = None

    if runs_dir.exists():
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir() or "recovery" not in run_path.name:
                continue

            metrics_path = run_path / "metrics.json"
            if not metrics_path.exists():
                # Some runs output summary.json instead, fallback
                metrics_path = run_path / "summary.json"
                if not metrics_path.exists():
                    continue

            with metrics_path.open("r", encoding="utf-8") as f:
                try:
                    metrics = json.load(f)
                except Exception:
                    print(
                        json.dumps(
                            {"run": run_path.name, "status": "rejected", "reason": "invalid_json"}
                        ),
                        file=sys.stderr,
                    )
                    continue

            # Check explicit reasons
            if (
                args.require_full_label
                and float(metrics.get("train_fraction", metrics.get("label_fraction", 0.0))) < 1.0
            ):
                _reject(run_path, "train_fraction_mismatch")
                continue
            if args.require_commit and metrics.get("git_commit") != args.require_commit:
                _reject(run_path, "commit_mismatch")
                continue

            run_prot_version = str(metrics.get("protocol_version", "unknown"))
            if run_prot_version != actual_protocol_version:
                _reject(run_path, "protocol_version_mismatch")
                continue

            nav_mode = metrics.get("navigation_mode")
            if nav_mode is None and "uses_ego_actions_in_student_predictor" in metrics:
                nav_mode = (
                    "enabled" if metrics["uses_ego_actions_in_student_predictor"] else "disabled"
                )

            if nav_mode != args.require_navigation:
                _reject(run_path, "navigation_mode_mismatch")
                continue

            # Extract model name handling differences in keys
            run_model = metrics.get("model_name", metrics.get("method"))
            if not isinstance(run_model, str):
                _reject(run_path, "model_name_missing")
                continue
            if "tiny_cnn" in run_model.lower():
                run_model = "tiny_cnn"
            elif "jepa" in run_model.lower() or "tubelet" in run_model.lower():
                run_model = "event-tubelet-transformer"

            if run_model != args.require_model_name:
                _reject(run_path, "model_name_mismatch")
                continue

            if (
                metrics.get("final_test_opened") is not False
                or metrics.get("evaluation_split") == "test"
            ):
                _reject(run_path, "final_test_exposure_or_undeclared_state")
                continue

            # Fetch mae
            mae = None
            if (
                "splits" in metrics
                and "validation" in metrics["splits"]
                and "metrics" in metrics["splits"]["validation"]
                and "mae_s" in metrics["splits"]["validation"]["metrics"]
            ):
                mae = float(metrics["splits"]["validation"]["metrics"]["mae_s"])
            elif "validation" in metrics and "mae_s" in metrics["validation"]:
                mae = float(metrics["validation"]["mae_s"])
            elif "best_validation_inverse_ttc_mae" in metrics:
                mae = float(metrics["best_validation_inverse_ttc_mae"])

            if mae is None or not math.isfinite(mae):
                _reject(run_path, "missing_or_nonfinite_validation_metrics")
                continue

            # Get checkpoint from run output
            ckpt_path_str = metrics.get("best_checkpoint", metrics.get("checkpoint_path"))
            if not ckpt_path_str:
                _reject(run_path, "missing_checkpoint_path_in_manifest")
                continue

            ckpt = Path(ckpt_path_str)
            if not ckpt.exists():
                _reject(run_path, "missing_checkpoint")
                continue

            # Path traversal check
            try:
                ckpt.resolve().relative_to(run_path.resolve())
            except ValueError:
                _reject(run_path, "checkpoint_outside_run_directory")
                continue

            cache_path = metrics.get(
                "cache_manifest", metrics.get("cache", metrics.get("cache_path"))
            )
            if not isinstance(cache_path, str) or not Path(cache_path).is_file():
                _reject(run_path, "cache_missing")
                continue
            cache_sha256 = compute_file_hash(cache_path)
            declared_cache_sha256 = metrics.get("cache_sha256", metrics.get("manifest_sha256"))
            if declared_cache_sha256 not in (None, cache_sha256):
                _reject(run_path, "cache_hash_mismatch")
                continue

            model_config_dict = metrics.get("run_fingerprint_payload", {}).get(
                "resolved_model_config", {}
            )
            if not model_config_dict:
                model_config_dict = metrics.get("model_config", {})

            cfg_sha256 = hash_dict(model_config_dict)

            run_protocol_hash = metrics.get("protocol_sha256", metrics.get("protocol_hash"))
            if not run_protocol_hash or run_protocol_hash == "unknown":
                _reject(run_path, "protocol_hash_missing")
                continue

            if run_protocol_hash != actual_protocol_hash:
                _reject(run_path, "protocol_hash_mismatch")
                continue

            git_commit = metrics.get("git_commit")
            if not isinstance(git_commit, str) or len(git_commit) != 40:
                _reject(run_path, "git_commit_missing_or_not_full_sha")
                continue

            split_manifest_sha256 = metrics.get("split_manifest_sha256")
            if not isinstance(split_manifest_sha256, str) or len(split_manifest_sha256) != 64:
                _reject(run_path, "split_manifest_hash_missing")
                continue

            if mae < best_mae:
                best_mae = mae
                best_record = {
                    "artifact_type": "onnx_candidate_v3",
                    "schema_version": "3.0",
                    "evidence_type": metrics.get("evidence_type", "validation_matrix"),
                    "code_commit": git_commit,
                    "protocol_version": actual_protocol_version,
                    "protocol_sha256": actual_protocol_hash,
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "checkpoint_path": ckpt.as_posix(),
                    "checkpoint_sha256": compute_file_hash(ckpt.as_posix()),
                    "cache_path": str(cache_path),
                    "cache_sha256": cache_sha256,
                    "model_config": model_config_dict,
                    "model_config_sha256": cfg_sha256,
                    "split_manifest_sha256": split_manifest_sha256,
                    "navigation_mode": nav_mode,
                    "selection_split": "validation",
                    "selection_metric": "mae_s",
                    "selection_metric_value": mae,
                    "run_id": run_path.name,
                    "model_name": run_model,
                    "seed": int(metrics.get("downstream_seed", metrics.get("seed", 0))),
                    "label_fraction": float(
                        metrics.get("train_fraction", metrics.get("label_fraction", 1.0))
                    ),
                    "train_sample_count": int(
                        metrics.get("effective_train_count", metrics.get("train_sample_count", 0))
                    ),
                    "validation_sample_count": int(metrics.get("validation_sample_count", 0)),
                    "final_test_opened": False,
                }

                for optional_hash in ("cache_sidecar_sha256", "normalization_sha256"):
                    value = metrics.get(optional_hash)
                    if isinstance(value, str) and len(value) == 64:
                        best_record[optional_hash] = value

    if best_record is None:
        print("Error: Could not find any valid trained checkpoint in validation.", file=sys.stderr)
        sys.exit(1)

    for key, val in best_record.items():
        if val == "unknown" or val == "missing" or val is None:
            print(
                json.dumps({"run": "best", "status": "rejected", "reason": f"missing_{key}"}),
                file=sys.stderr,
            )
            sys.exit(1)

    best_record = sign_artifact(best_record)
    print(json.dumps(best_record, indent=2))


if __name__ == "__main__":
    main()

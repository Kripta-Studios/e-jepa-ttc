import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="artifacts/runs")
    parser.add_argument(
        "--require-full-label",
        action="store_true",
        help="Only select models with train_fraction == 1.0",
    )
    parser.add_argument(
        "--require-commit", type=str, default=None, help="Only select models matching this commit"
    )
    parser.add_argument(
        "--require-protocol",
        type=str,
        default="2.0",
        help="Only select models matching this protocol version",
    )
    parser.add_argument(
        "--require-navigation",
        type=str,
        default="enabled",
        help="Only select models matching this navigation mode",
    )
    parser.add_argument(
        "--require-model-name",
        type=str,
        default="event-tubelet-transformer",
        help="Only select models matching this name",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    best_mae = float("inf")
    best_checkpoint = None
    best_run_path = None
    best_metrics = None

    if runs_dir.exists():
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir() or "recovery" not in run_path.name:
                continue

            metrics_path = run_path / "metrics.json"
            if not metrics_path.exists():
                continue

            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)

                # Use semantic completion from validation.py
                from e_jepa_ttc.experiments.validation import (
                    load_protocol,
                    verify_semantic_completion,
                )

                protocol_path = Path("configs/recovery_v3_protocol.yaml")
                schema_path = Path("schemas/recovery_v3_protocol.schema.json")
                if protocol_path.exists() and schema_path.exists():
                    protocol = load_protocol(protocol_path, schema_path)
                    if not verify_semantic_completion(
                        metrics_path, protocol, require_metrics=True
                    ):
                        continue

                # Strict filters
                if args.require_full_label and float(metrics.get("train_fraction", 0.0)) < 1.0:
                    continue
                if args.require_commit and metrics.get("git_commit") != args.require_commit:
                    continue
                if (
                    args.require_protocol
                    and metrics.get("protocol_version") != args.require_protocol
                ):
                    continue
                if (
                    args.require_navigation
                    and metrics.get("navigation_mode") != args.require_navigation
                ):
                    continue
                if (
                    args.require_model_name
                    and metrics.get("model_name", "event-tubelet-transformer")
                    != args.require_model_name
                ):
                    continue
                if metrics.get("final_test_opened") is True:
                    continue

                # Ensure we only use validation split
                if (
                    "splits" in metrics
                    and "validation" in metrics["splits"]
                    and "metrics" in metrics["splits"]["validation"]
                    and "mae_s" in metrics["splits"]["validation"]["metrics"]
                ):
                    mae = float(metrics["splits"]["validation"]["metrics"]["mae_s"])
                    if mae < best_mae:
                        ckpt = run_path / "tiny_cnn_best.pt"
                        if ckpt.exists():
                            best_mae = mae
                            best_checkpoint = ckpt.as_posix()
                            best_run_path = run_path
                            best_metrics = metrics
            except Exception:
                pass

    if best_checkpoint is None:
        print("Error: Could not find any valid trained checkpoint in validation.", file=sys.stderr)
        sys.exit(1)

    import hashlib

    def hash_file(path):
        if not path or not Path(path).exists():
            return "missing"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                h.update(b)
        return h.hexdigest()

    cache_path = best_metrics.get("cache", best_metrics.get("cache_path"))
    cfg_path = best_metrics.get("model", best_metrics.get("model_config_path", "unknown"))
    cache_sha256 = best_metrics.get("cache_sha256")
    cfg_sha256 = best_metrics.get("run_fingerprint", best_metrics.get("model_config_sha256", "unknown"))
    protocol_hash = best_metrics.get("protocol_version", best_metrics.get("protocol_hash", "unknown"))
    git_commit = best_metrics.get("git_commit")

    for name, val in [
        ("cache_path", cache_path), 
        ("cfg_path", cfg_path),
        ("cache_sha256", cache_sha256), 
        ("cfg_sha256", cfg_sha256), 
        ("protocol_hash", protocol_hash), 
        ("git_commit", git_commit)
    ]:
        if not val or val == "unknown":
            print(f"Error: Required provenance {name} is missing or unknown.", file=sys.stderr)
            sys.exit(1)

    selection_record = {
        "checkpoint_path": best_checkpoint,
        "checkpoint_sha256": hash_file(best_checkpoint),
        "cache_path": cache_path,
        "cache_sha256": cache_sha256,
        "model_config_path": cfg_path,
        "model_config_sha256": cfg_sha256,
        "protocol_hash": protocol_hash,
        "code_commit": git_commit,
    }

    print(json.dumps(selection_record, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="artifacts/runs")
    parser.add_argument("--require-full-label", action="store_true", help="Only select models with train_fraction == 1.0")
    parser.add_argument("--require-commit", type=str, default=None, help="Only select models matching this commit")
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

                # Strict filters
                if args.require_full_label and float(metrics.get("train_fraction", 0.0)) < 1.0:
                    continue
                if args.require_commit and metrics.get("git_commit") != args.require_commit:
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

    selection_manifest = {
        "status": "passed",
        "best_checkpoint": best_checkpoint,
        "validation_mae": best_mae,
        "train_fraction": float(best_metrics.get("train_fraction", 1.0)),
        "git_commit": best_metrics.get("git_commit", "unknown")
    }
    with open(best_run_path / "selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2)

    print(best_checkpoint)


if __name__ == "__main__":
    main()

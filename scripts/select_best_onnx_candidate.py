import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="artifacts/runs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    best_mae = float("inf")
    best_checkpoint = None

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

                # Ensure we only use validation split
                if (
                    "splits" in metrics
                    and "validation" in metrics["splits"]
                    and "metrics" in metrics["splits"]["validation"]
                    and "mae_s" in metrics["splits"]["validation"]["metrics"]
                ):
                    mae = float(metrics["splits"]["validation"]["metrics"]["mae_s"])
                    if mae < best_mae:
                        best_mae = mae
                        ckpt = run_path / "tiny_cnn_best.pt"
                        if ckpt.exists():
                            best_checkpoint = ckpt.as_posix()
            except Exception:
                pass

    if best_checkpoint is None:
        print("Error: Could not find any valid trained checkpoint in validation.", file=sys.stderr)
        sys.exit(1)

    print(best_checkpoint)


if __name__ == "__main__":
    main()

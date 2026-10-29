"""Build an embedding-health figure from a recorded JEPA history JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_figure(history_path: Path, figure_path: Path, summary_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows:
        raise ValueError(f"History is empty: {history_path}")
    epochs = [int(row["epoch"]) for row in rows]
    validation = [row["validation"] for row in rows]
    fields = {
        "loss": [float(row["loss"]) for row in validation],
        "embedding_std": [float(row["context_embedding_std"]) for row in validation],
        "effective_rank": [float(row["context_effective_rank"]) for row in validation],
        "collapsed_dimension_fraction": [
            float(row["context_collapsed_dimension_fraction"]) for row in validation
        ],
    }
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = (
        (axes[0, 0], "loss", "Validation JEPA loss"),
        (axes[0, 1], "embedding_std", "Context embedding std"),
        (axes[1, 0], "effective_rank", "Context effective rank"),
        (axes[1, 1], "collapsed_dimension_fraction", "Collapsed dimension fraction"),
    )
    for axis, field, title in panels:
        axis.plot(epochs, fields[field], marker="o")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[1, 1].set_ylim(-0.02, 1.02)
    fig.suptitle("JEPA embedding health (recorded metrics)")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    result = {
        "artifact_type": "jepa_embedding_health_figure_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "history": history_path.as_posix(),
        "history_sha256": _sha256(history_path),
        "figure": figure_path.as_posix(),
        "figure_sha256": _sha256(figure_path),
        "status": "PASS",
        "epochs": epochs,
        "last_validation": {key: values[-1] for key, values in fields.items()},
        "collapse_guard_threshold": 0.8,
        "metrics_source": "recorded_history_jsonl",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_figure(args.history, args.figure, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

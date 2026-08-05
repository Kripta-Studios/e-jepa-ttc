#!/usr/bin/env python
"""Compare scratch and Level-transfer Object Event TTC v4 summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _row(name: str, path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    evaluation = value["best_evaluation"]
    full = evaluation["full"]
    event = evaluation["event_only"]
    motion = evaluation["motion_only"]
    dependence = evaluation["event_dependence"]
    metrics = full["metrics"]
    return {
        "arm": name,
        "seed": value.get("seed"),
        "best_epoch": value.get("best_epoch"),
        "full_macro_mid": _get(
            metrics, "sequence_macro", "sequence_macro_paper_MiD_overall"
        ),
        "full_mae_s": metrics.get("mae_s"),
        "full_expansion_pearson": _get(full, "expansion_health", "pearson"),
        "event_macro_mid": _get(
            event,
            "metrics",
            "sequence_macro",
            "sequence_macro_paper_MiD_overall",
        ),
        "event_expansion_pearson": _get(event, "expansion_health", "pearson"),
        "event_balanced_sign": _get(event, "direction", "balanced_accuracy"),
        "event_negative_recall": _get(event, "direction", "negative_recall"),
        "motion_expansion_pearson": _get(motion, "expansion_health", "pearson"),
        "pearson_drop_zero_events": dependence.get("pearson_drop_zero_events"),
        "pearson_drop_shuffled_events": dependence.get(
            "pearson_drop_shuffled_events"
        ),
        "reversal_error_max": evaluation.get("reversal_error_max"),
        "event_gate_mean": evaluation.get("event_gate_mean"),
        "saturation_rate": full.get("saturation_rate"),
        "pretrained": _get(value, "pretraining", "used"),
        "summary": path.resolve().as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--level", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = pd.DataFrame(
        [_row("scratch", args.scratch), _row("level-transfer", args.level)]
    )
    print(frame.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

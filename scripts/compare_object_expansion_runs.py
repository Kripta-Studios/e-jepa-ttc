#!/usr/bin/env python
"""Compare scratch and JEPA-transfer Object-expansion summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("artifact_type") != "e_jepa_object_expansion_training_v2":
        raise ValueError(f"Unexpected summary artifact: {path}")
    return value


def _row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    evaluation = summary["best_evaluation"]
    metrics = evaluation["metrics"]
    macro = metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"]
    return {
        "arm": name,
        "best_epoch": summary["best_epoch"],
        "macro_mid": macro,
        "mae_s": metrics["mae_s"],
        "sign_accuracy": metrics["sign_accuracy"],
        "balanced_sign_accuracy": evaluation["direction"]["balanced_accuracy"],
        "inverse_pearson": evaluation["signed_inverse_health"]["pearson"],
        "log_ratio_pearson": evaluation["log_ratio_health"]["pearson"],
        "saturation_rate": evaluation["ttc_saturation_rate"],
        "pretrained": summary["pretraining"]["used"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        _row("scratch", _load(args.scratch.resolve())),
        _row("level-transfer", _load(args.transfer.resolve())),
    ]
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

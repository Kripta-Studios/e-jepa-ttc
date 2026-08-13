#!/usr/bin/env python
"""Paired sequence+track bootstrap for two neutral grouped-development arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from scripts.paired_cluster_bootstrap import run as run_legacy_pair  # noqa: E402

LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    first_path: Path,
    second_path: Path,
    output: Path,
    *,
    first_label: str,
    second_label: str,
    fold: int | None,
    resamples: int,
    seed: int,
    cluster_metadata: Path,
    protocol: Path,
) -> dict[str, Any]:
    """Compare two arms without assigning E-JEPA/Garl semantics to either side."""

    if not LABEL.fullmatch(first_label) or not LABEL.fullmatch(second_label):
        raise ValueError("comparison labels must be lowercase identifier strings")
    if first_label == second_label:
        raise ValueError("comparison labels must differ")
    if fold is not None and fold not in {0, 1, 2}:
        raise ValueError("fold must be 0, 1, 2, or omitted for the OOF aggregate")
    if not cluster_metadata.is_file() or not protocol.is_file():
        raise ValueError("cluster metadata and grouped protocol must exist")

    legacy = run_legacy_pair(
        first_path,
        second_path,
        output,
        resamples,
        seed,
        cluster_metadata,
        comparison_scope="diagnostic_only",
    )
    if not str(legacy["cluster_definition"]).endswith("verified_external_metadata"):
        raise ValueError("grouped comparison requires externally verified sequence+track clusters")

    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v5_grouped_paired_bootstrap_v1",
        "status": "completed_train_only_grouped_development",
        "created_at_utc": legacy["created_at_utc"],
        "comparison": {"first": first_label, "second": second_label},
        "fold": fold,
        "rows": legacy["rows"],
        "clusters": legacy["clusters"],
        "cluster_definition": "sequence_id+track_id_verified_external_metadata",
        "resamples": resamples,
        "seed": seed,
        "checks": {
            "exact_sample_tokens": True,
            "target_equality_atol_1e_5": True,
            "track_identity_verified": True,
            "paired_evaluation": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        },
        "sample_contract": legacy["sample_contract"],
        "sources": {
            "first_predictions": legacy["sources"]["ejepa_predictions"],
            "second_predictions": legacy["sources"]["garl_predictions"],
            "cluster_metadata": legacy["sources"]["cluster_metadata"],
            "protocol": {
                "path": str(protocol.resolve()),
                "sha256": _sha256(protocol),
            },
        },
        "first": legacy["ejepa"],
        "second": legacy["garl"],
        "delta_first_minus_second": legacy["delta_ejepa_minus_garl"],
        "bootstrap": {
            "sequence_macro_MiD_delta_first_minus_second": legacy["bootstrap"][
                "sequence_macro_MiD_delta_ejepa_minus_garl"
            ],
            "failure_pct_delta_first_minus_second": legacy["bootstrap"][
                "failure_pct_delta_ejepa_minus_garl"
            ],
            "probability_first_lower_MiD": legacy["bootstrap"][
                "probability_ejepa_lower_MiD"
            ],
        },
        "claim_scope": "development_only_not_sealed_not_sota",
    }
    sign_artifact(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-predictions", type=Path, required=True)
    parser.add_argument("--second-predictions", type=Path, required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--cluster-metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(
            args.first_predictions.resolve(strict=True),
            args.second_predictions.resolve(strict=True),
            args.output.resolve(),
            first_label=args.first_label,
            second_label=args.second_label,
            fold=args.fold,
            resamples=args.resamples,
            seed=args.seed,
            cluster_metadata=args.cluster_metadata.resolve(strict=True),
            protocol=args.protocol.resolve(strict=True),
        )
    except Exception as error:
        parser.exit(2, f"grouped paired bootstrap failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"output": str(args.output), "delta": report["delta_first_minus_second"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

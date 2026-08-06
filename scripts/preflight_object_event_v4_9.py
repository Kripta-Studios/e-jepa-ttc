#!/usr/bin/env python3
"""Fail-closed preflight for the v4.9 fixed event-only fusion screen."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REQUIRED_BASE = "488b433857090e525c447bf2974ac72639f25194"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v42-summary", type=Path, required=True)
    parser.add_argument("--v42-train-predictions", type=Path, required=True)
    parser.add_argument("--v42-validation-predictions", type=Path, required=True)
    parser.add_argument("--v48-summary", type=Path, required=True)
    parser.add_argument("--v48-train-predictions", type=Path, required=True)
    parser.add_argument("--v48-validation-predictions", type=Path, required=True)
    args = parser.parse_args()

    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_BASE, head], check=False
    ).returncode == 0
    if not ancestor:
        raise RuntimeError(f"HEAD {head} does not descend from {REQUIRED_BASE}")
    for path in vars(args).values():
        if not path.is_file():
            raise FileNotFoundError(path)
    v42 = json.loads(args.v42_summary.read_text(encoding="utf-8"))
    v48 = json.loads(args.v48_summary.read_text(encoding="utf-8"))
    if v42.get("artifact_type") != "object_event_v4_2_full_event_only_screen":
        raise RuntimeError("Unexpected v4.2 summary")
    if not bool(v42.get("screen_passed")):
        raise RuntimeError("v4.2 did not pass")
    if v48.get("artifact_type") != "object_event_v4_8_dense_foreground_motion":
        raise RuntimeError("Unexpected v4.8 summary")
    if v48.get("mode") != "screen" or not bool(v48.get("passed")):
        raise RuntimeError("v4.8 screen did not pass")
    print(
        json.dumps(
            {
                "status": "passed",
                "head": head,
                "required_base_ancestor": REQUIRED_BASE,
                "v4_2_summary": args.v42_summary.resolve().as_posix(),
                "v4_8_summary": args.v48_summary.resolve().as_posix(),
                "scientific_contract": {
                    "fixed_fusion_only": True,
                    "alpha_informed_by_prior_development_results": True,
                    "v4_9_does_not_fit_alpha": True,
                    "official_eap_test_not_opened": True,
                    "evttc_not_opened": True,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

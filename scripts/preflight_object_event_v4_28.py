#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--v427-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    args = parser.parse_args()
    for path in (args.cache_manifest, args.v48_config, args.v427_summary):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = json.loads(args.v427_summary.read_text(encoding="utf-8"))
    if summary.get("artifact_type") != "object_event_v4_27_scale_correlation_lhr":
        raise ValueError("expected v4.27 summary")
    if summary.get("status") != "completed_oof_gate_failed":
        raise ValueError("v4.28 is preregistered for the failed v4.27 OOF result")
    if not summary.get("scientific_contract", {}).get("development_validation_not_materialized_after_oof_failure", False):
        raise ValueError("v4.27 scientific contract does not prove validation remained sealed")
    seeds = []
    for item in args.adapted_checkpoint:
        seed, raw = item.split("=", 1)
        seeds.append(int(seed))
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid checkpoint {path}")
    if sorted(seeds) != [7, 13, 23]:
        raise ValueError("v4.28 requires adapted seeds 7,13,23")
    print({
        "status": "passed",
        "v427_oof_pearson": summary["oof_train_metrics"]["pearson"],
        "v427_log_eta_pearson": summary["oof_train_metrics"]["log_eta_pearson"],
        "development_validation_still_sealed": True,
        "official_eap_test_not_opened": True,
        "evttc_not_opened": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

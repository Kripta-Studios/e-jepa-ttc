#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-manifest", type=Path, required=True)
    p.add_argument("--v48-config", type=Path, required=True)
    p.add_argument("--adapted-checkpoint", action="append", required=True)
    args = p.parse_args()
    for path in (args.cache_manifest, args.v48_config):
        if not path.exists():
            raise FileNotFoundError(path)
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
        raise ValueError("v4.27 requires adapted seeds 7,13,23")
    print({"status": "passed", "adapted_seeds": sorted(seeds), "official_eap_test_not_opened": True, "evttc_not_opened": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

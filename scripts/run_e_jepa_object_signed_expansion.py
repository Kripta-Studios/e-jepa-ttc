#!/usr/bin/env python
"""Build/reuse the object cache and train signed-expansion TTC v3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_BUILDER = ROOT / "scripts" / "build_garlttc_object_lhr_cache.py"
TRAINER = ROOT / "scripts" / "train_e_jepa_object_signed_expansion.py"
PROFILES = {
    "screen": {
        "split": ROOT / "data" / "splits" / "eap_pilot12_v1.json",
        "experiment": (
            ROOT
            / "configs"
            / "experiment"
            / "e_jepa_garl_object_signed_expansion_screen_v3.yaml"
        ),
        "cache": ROOT / "artifacts" / "cache" / "garl_object_lhr_screen_v2",
        "output": ROOT / "artifacts" / "runs" / "e_jepa_garl_object_signed_expansion_screen_v3",
        "max_samples": 2048,
        "seeds": (7,),
    },
    "full": {
        "split": ROOT / "data" / "splits" / "eap_train40_v1.json",
        "experiment": (
            ROOT
            / "configs"
            / "experiment"
            / "e_jepa_garl_object_signed_expansion_full_v3.yaml"
        ),
        "cache": ROOT / "artifacts" / "cache" / "garl_object_lhr_full_v2",
        "output": ROOT / "artifacts" / "runs" / "e_jepa_garl_object_signed_expansion_full_v3",
        "max_samples": None,
        "seeds": (7, 13, 23),
    },
}


def _run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps(command, ensure_ascii=False))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="screen")
    parser.add_argument("--stages", nargs="+", choices=("cache", "train"), default=["train"])
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-workers", type=int, default=2)
    parser.add_argument("--train-workers", type=int)
    parser.add_argument("--preprocessing-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    split = (args.split or profile["split"]).resolve()
    cache_dir = (args.cache_dir or profile["cache"]).resolve()
    output_root = (args.output_root or profile["output"]).resolve()
    seeds = tuple(args.seeds or profile["seeds"])
    if args.profile == "full" and not set(seeds).issubset(set(profile["seeds"])):
        raise ValueError(f"Full profile seeds must be a subset of {sorted(profile['seeds'])}")
    if args.profile == "full" and args.pretrained is not None and len(seeds) != 1:
        raise ValueError("Full transfer requires one seed-matched SSL checkpoint per run")

    if "cache" in args.stages:
        if args.eap_root is None or args.garlttc_root is None:
            raise ValueError("cache stage requires --eap-root and --garlttc-root")
        command = [
            sys.executable,
            str(CACHE_BUILDER),
            "--eap-root",
            str(args.eap_root.resolve()),
            "--garlttc-root",
            str(args.garlttc_root.resolve()),
            "--split",
            str(split),
            "--output-dir",
            str(cache_dir),
            "--workers",
            str(args.cache_workers),
            "--preprocessing-device",
            args.preprocessing_device,
            "--storage-profile",
            "object_lhr_minimal",
            "--shard-size",
            "16",
        ]
        if profile["max_samples"] is not None:
            command.extend(("--max-samples-per-split", str(profile["max_samples"])))
        if args.resume:
            command.append("--resume")
        _run(command, dry_run=args.dry_run)

    if "train" in args.stages:
        manifest = cache_dir / "manifest.json"
        if not args.dry_run and not manifest.is_file():
            raise FileNotFoundError(f"Missing cache manifest {manifest}")
        arm = "level-transfer" if args.pretrained is not None else "scratch"
        for seed in seeds:
            command = [
                sys.executable,
                str(TRAINER),
                "--cache-manifest",
                str(manifest),
                "--config",
                str(profile["experiment"]),
                "--output-dir",
                str(output_root / arm / f"seed-{seed}"),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            if args.train_workers is not None:
                command.extend(("--workers", str(args.train_workers)))
            if args.pretrained is not None:
                command.extend(("--pretrained", str(args.pretrained.resolve())))
            if args.resume:
                command.append("--resume")
            _run(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

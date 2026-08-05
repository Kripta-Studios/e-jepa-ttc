#!/usr/bin/env python
"""Orchestrate Object Event TTC v4 cache, scratch and Level-transfer runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_assignment_split(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    assignments = value.get("assignments") if isinstance(value, dict) else None
    return (
        isinstance(assignments, dict)
        and isinstance(assignments.get("train"), list)
        and isinstance(assignments.get("validation"), list)
        and bool(assignments["train"])
        and bool(assignments["validation"])
    )


def _resolve_assignment_split(candidate: Path) -> Path:
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    if _is_assignment_split(candidate):
        return candidate

    legacy_manifest = (
        ROOT / "artifacts" / "cache" / "garl_object_lhr_screen_v2" / "manifest.json"
    )
    if legacy_manifest.is_file():
        value = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        fallback_value = value.get("split_path")
        if isinstance(fallback_value, str) and fallback_value.strip():
            fallback = Path(fallback_value).expanduser()
            if not fallback.is_absolute():
                fallback = (ROOT / fallback).resolve()
            if _is_assignment_split(fallback):
                print(
                    json.dumps(
                        {
                            "warning": "requested split is not an assignments artifact",
                            "requested_split": candidate.as_posix(),
                            "resolved_split": fallback.as_posix(),
                            "source_manifest": legacy_manifest.as_posix(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return fallback

    top_level_keys: list[str] = []
    if candidate.is_file():
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                top_level_keys = sorted(str(key) for key in value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    raise ValueError(
        "Object Event v4 requires a sequence split with "
        "assignments.train and assignments.validation. "
        f"Candidate={candidate}; top_level_keys={top_level_keys}. "
        "The eap_level_dynamics_v1 manifest is a subset/training manifest, "
        "not a sequence assignment split."
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps(command, ensure_ascii=False), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("screen", "full"), default="screen")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("preflight", "cache", "scratch", "level"),
        default=("preflight", "cache", "scratch", "level"),
    )
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "artifacts" / "manifests" / "eap_level_dynamics_v1.json",
    )
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts" / "cache")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_root / f"garl_object_event_common_roi_{args.profile}_v4"
    manifest = cache_dir / "manifest.json"
    config = ROOT / "configs" / "experiment" / f"e_jepa_garl_object_event_{args.profile}_v4.yaml"
    run_root = args.output_root / f"e_jepa_garl_object_event_{args.profile}_v4"
    python = sys.executable

    if "preflight" in args.stages:
        command = [
            python,
            str(ROOT / "scripts" / "preflight_object_event_v4.py"),
        ]
        if manifest.is_file():
            command.extend(["--manifest", str(manifest)])
        _run(command, dry_run=args.dry_run)

    if "cache" in args.stages:
        if args.eap_root is None or args.garlttc_root is None:
            raise ValueError("--eap-root and --garlttc-root are required for stage cache")
        split_path = _resolve_assignment_split(args.split)
        command = [
            python,
            str(ROOT / "scripts" / "build_eap_object_event_v4_cache.py"),
            "--eap-root",
            str(args.eap_root),
            "--garlttc-root",
            str(args.garlttc_root),
            "--split",
            str(split_path),
            "--output-dir",
            str(cache_dir),
            "--profile",
            args.profile,
            "--workers",
            str(args.workers),
        ]
        if args.resume:
            command.append("--resume")
        _run(command, dry_run=args.dry_run)

    common = [
        python,
        str(ROOT / "scripts" / "train_e_jepa_object_event_v4.py"),
        "--cache-manifest",
        str(manifest),
        "--config",
        str(config),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.resume:
        common.append("--resume")

    if "scratch" in args.stages:
        _run(
            common
            + [
                "--output-dir",
                str(run_root / "scratch" / f"seed-{args.seed}"),
            ],
            dry_run=args.dry_run,
        )
    if "level" in args.stages:
        if args.pretrained is None:
            raise ValueError("--pretrained is required for stage level")
        _run(
            common
            + [
                "--output-dir",
                str(run_root / "level-transfer" / f"seed-{args.seed}"),
                "--pretrained",
                str(args.pretrained),
            ],
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

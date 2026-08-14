#!/usr/bin/env python
# ruff: noqa: E501
"""Plan or execute frozen V8 B1/B2 seed-7 temporal frontend folds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (  # noqa: E402
    V8IntegrityError,
    verify_frozen_inputs,
)
from e_jepa_ttc.training.scientific_recovery_v8_jobs import (  # noqa: E402
    V8JobIntegrityError,
    build_fold_jobs,
    execute_jobs,
    plan_to_json,
)


def _configs(frozen: object, arm: str) -> list[Path]:
    manifest = frozen.manifest
    entries = manifest["enabled_seed7_configs"]
    selected = [
        ROOT / str(entry["path"])
        for name, entry in entries.items()
        if name.startswith(f"{arm}_fold") and name.endswith("_seed7")
    ]
    if len(selected) != 3:
        raise V8JobIntegrityError(f"frozen manifest lacks exactly three {arm} seed-7 fold configs")
    return sorted(selected)


def _root(argument: Path | None, environment: str, pinned: str, label: str) -> Path:
    if argument is not None:
        return argument.resolve()
    value = os.environ.get(environment)
    if value:
        return Path(value).resolve()
    fallback = Path(pinned)
    if fallback.is_dir():
        return fallback.resolve()
    raise V8JobIntegrityError(
        f"{label} cache prerequisites are absent: pass its root argument or set {environment}; "
        f"the local default {pinned!r} does not exist."
    )


def _cache_command(
    *, arm: str, eap_root: Path, garlttc_root: Path, protocol: Path, output: Path
) -> list[str]:
    representation = "timevol20" if arm == "timevol20_3" else "exp6"
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/build_scientific_recovery_v8_cache.py",
        "--eap-root",
        str(eap_root),
        "--garlttc-root",
        str(garlttc_root),
        "--protocol",
        str(protocol),
        "--output-dir",
        str(output),
        "--representation",
        representation,
        "--steps",
        "3",
        "--resume",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "artifacts/scientific_recovery_v8"
    )
    parser.add_argument("--arm", choices=("all", "timevol20_3", "exp6_3"), default="all")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        frozen = verify_frozen_inputs(args.protocol, args.manifest)
        arms = ("timevol20_3", "exp6_3") if args.arm == "all" else (args.arm,)
        cache_commands: list[list[str]] = []
        for arm in arms:
            cache = ROOT / "artifacts/scientific_recovery_v8/cache" / arm / "manifest.json"
            if not cache.is_file():
                eap_root = _root(args.eap_root, "EAP_ROOT", "E:/eAP_dataset", "eAP")
                garl_root = _root(
                    args.garlttc_root, "GARLTTC_ROOT", "E:/GarlTTC_dataset", "GarlTTC"
                )
                cache_commands.append(
                    _cache_command(
                        arm=arm,
                        eap_root=eap_root,
                        garlttc_root=garl_root,
                        protocol=args.protocol.resolve(),
                        output=cache.parent,
                    )
                )
        jobs = tuple(
            job
            for arm in arms
            for job in build_fold_jobs(
                configs=_configs(frozen, arm),
                output_root=args.results_root / "runs",
                device=args.device,
                max_parallel=args.max_parallel,
            )
        )
        if args.dry_run:
            plan = plan_to_json(jobs)
            plan["cache_jobs"] = cache_commands
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        for command in cache_commands:
            if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
                raise V8JobIntegrityError("V8 temporal cache materialization failed")
        outputs = execute_jobs(
            jobs,
            protocol_hash=str(frozen.protocol["artifact_sha256"]),
            manifest_hash=str(frozen.manifest["artifact_sha256"]),
            dry_run=False,
            max_parallel=args.max_parallel,
        )
        print(json.dumps({"status": "executed", "jobs": outputs}, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, V8IntegrityError, V8JobIntegrityError) as error:
        parser.exit(2, f"V8 temporal stage failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

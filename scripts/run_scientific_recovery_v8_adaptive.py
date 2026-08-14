#!/usr/bin/env python
# ruff: noqa: E501
"""Fail-closed V8 C1 adaptive stage; it never manufactures conditional configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (  # noqa: E402
    V8IntegrityError,
    assert_adaptive_gate,
    verify_frozen_inputs,
)
from e_jepa_ttc.training.scientific_recovery_v8_jobs import (  # noqa: E402
    V8JobIntegrityError,
    build_fold_jobs,
    execute_jobs,
    plan_to_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditional_fold_configs(template: dict[str, object]) -> list[Path]:
    """Resolve exactly three signed C1 configs; never synthesize conditional arms."""

    entries = template.get("fold_configs")
    if not isinstance(entries, list) or len(entries) != 3:
        raise V8IntegrityError("enabled C1 template requires exactly three signed fold_configs")
    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise V8IntegrityError("C1 fold config entry lacks a relative path")
        raw = Path(str(entry["path"]))
        if raw.is_absolute():
            raise V8IntegrityError("C1 fold config path must be repository-relative")
        path = ROOT / raw
        if not path.is_file() or entry.get("sha256") != _sha256(path):
            raise V8IntegrityError(f"C1 frozen config hash mismatch: {raw}")
        paths.append(path)
    return sorted(paths)


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
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        frozen = verify_frozen_inputs(args.protocol, args.manifest)
        assert_adaptive_gate(results_root=args.results_root, frozen=frozen)
        templates = frozen.manifest.get("conditional_templates", {})
        gated = templates.get("gated_exp6_3") if isinstance(templates, dict) else None
        if not isinstance(gated, dict) or gated.get("enabled") is not True:
            raise V8IntegrityError(
                "C1 gate opened but no signed conditional gated_exp6_3 fold configs are frozen; "
                "regenerate the protocol contract before training."
            )
        configs = conditional_fold_configs(gated)
        jobs = build_fold_jobs(
            configs=configs,
            output_root=args.results_root / "adaptive" / "runs",
            device=args.device,
            max_parallel=args.max_parallel,
        )
        if args.dry_run:
            print(json.dumps(plan_to_json(jobs), indent=2, sort_keys=True))
            return 0
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
        parser.exit(2, f"V8 adaptive stage failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

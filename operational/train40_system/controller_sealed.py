"""Avoid remerging thousands of preparation receipts after complete immutable admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import Lease, digest
from operational.train40_system import controller, controller_resources8
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT

_refresh = controller.refresh_complete_inputs


def refresh_complete_inputs(output: Path) -> bool:
    """Reuse the admitted manifest; live consumers continue checking every source fragment."""
    progress_path = output / "INPUT_PREPARATION_PROGRESS.json"
    manifest_path = output / "INPUT_MANIFEST.json"
    models_path = output / "MODELS_FREEZE.json"
    if all(path.exists() for path in (progress_path, manifest_path, models_path)):
        progress, manifest, models = read(progress_path), read(manifest_path), read(models_path)
        if progress["status"] == "COMPLETE" and manifest["status"] == "COMPLETE_VERIFIED":
            if (
                progress["completed_rows"] != 88744
                or manifest["row_count"] != 88744
                or manifest["sequence_count"] != 40
                or digest(manifest_path) != models["input_manifest_sha256"]
            ):
                raise ValueError("Preserve the sealed complete TRAIN40 input contract")
            return True
    return _refresh(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    args = parser.parse_args()
    verify_sources(read(args.output / "SAFE_PREPARATION_FREEZE.json"))
    verify_sources(read(args.output / "FEATURES_EIGHT_WORKER_FREEZE.json"))
    verify_sources(read(args.output / "CONTROLLER_SEALED_FREEZE.json"))
    controller.refresh_complete_inputs = refresh_complete_inputs
    controller.active = controller_resources8.active
    controller.launch = controller_resources8.launch
    with Lease((args.output / "controller").resolve()):
        controller.run(args.output.resolve(), args.raw_root.resolve(), args.teacher_path.resolve())

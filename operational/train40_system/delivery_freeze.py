"""Freeze independently tested postprocessing while preserving the scientific source freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files


def run(output: Path, code_root: Path) -> None:
    """Bind native inference, signed diagnostics, packaging and executed context QA sources."""
    checks = {
        "POSTPROCESS_PYRIGHT.txt": "0 errors",
        "POSTPROCESS_RUFF.txt": "All checks passed",
        "POSTPROCESS_PYTEST.txt": "[100%]",
    }
    for name, marker in checks.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError("Required independent postprocessing QA failed: " + name)
    context = read(output / "CONTEXT_AUDIT.json")
    archived = output / "admission_sources/context_audit_executed.py"
    if context["status"] != "PASSED" or digest(archived) != context["source_sha256"]:
        raise ValueError("Executed complete causal support audit source required")
    files = dependency_files(
        [
            ROOT / "operational/train40_system" / (name + ".py")
            for name in (
                "training_diagnostics",
                "context_audit",
                "garl_predictions",
                "comparison_metrics",
                "delivery",
                "followup",
                "delivery_freeze",
            )
        ]
    )
    native = [
        {"path": path.relative_to(code_root).as_posix(), "sha256": digest(path)}
        for path in sorted((code_root / "garl_ttc").rglob("*.py"))
    ]
    contract = {
        "schema": "train40_postprocessing_freeze_v1",
        "files": files,
        "native_code_root": str(code_root),
        "native_source_files": native,
        "public_checkpoint_sha256": digest(output / "public_garl/paper_event_only_lhr.pth"),
        "config_sha256": digest(output / "public_garl/configs/ablation/event_lhr.yaml"),
        "context_audit_sha256": digest(output / "CONTEXT_AUDIT.json"),
        "QA": {name: digest(output / name) for name in checks},
        "new_optimizer_updates_authorized": 0,
        "public_comparison_role": "TRAIN_FIT_DIAGNOSTIC_NOT_GENERALIZATION",
        "scientific_engineering_freeze_unchanged": digest(output / "ENGINEERING_FREEZE.json"),
    }
    path = output / "DELIVERY_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing independent postprocessing freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.code_root.resolve())

"""Admit a resource-only H8 variant without altering the frozen mathematical kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

REPLACEMENTS = (
    (
        "ProcessPoolExecutor(max_workers=4, initializer=worker_init)",
        "ProcessPoolExecutor(max_workers=8, initializer=worker_init)",
    ),
    ("min(row + 4, len(jobs))", "min(row + 8, len(jobs))"),
    (
        'read(args.output / "FEATURES_FREEZE.json")',
        'read(args.output / "FEATURES_EIGHT_WORKER_FREEZE.json")',
    ),
    ('choices=("PAIR", "H8")', 'choices=("H8",)'),
)


def expected_variant(source: str) -> str:
    """Change exactly the pool, bounded lookahead, admission file and H8-only entrypoint."""
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise ValueError("The frozen resource site changed: " + before)
        source = source.replace(before, after, 1)
    return source


def generate() -> None:
    """Preserve the original source and create the explicitly reviewable resource variant."""
    original = ROOT / "operational/train40_system/history_features.py"
    variant = original.with_name("history_resources8.py")
    expected = expected_variant(original.read_text(encoding="utf-8"))
    if variant.exists():
        if variant.read_text(encoding="utf-8") != expected:
            raise ValueError("Preserve existing resource variant; unexpected implementation change")
    else:
        variant.write_text(expected, encoding="utf-8")


def freeze(output: Path) -> None:
    """Freeze executed QA and exact source changes before any H8 feature generation."""
    generate()
    baseline = read(output / "FEATURES_FREEZE.json")
    verify_sources(baseline)
    checks = {
        "FEATURE_RESOURCES_PYTEST.txt": "[100%]",
        "FEATURE_RESOURCES_RUFF.txt": "All checks passed",
        "FEATURE_RESOURCES_PYRIGHT.txt": "0 errors",
    }
    for name, marker in checks.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError("Resource admission QA did not pass: " + name)
    if (output / "h8_feature_fragments/BINDING.json").exists():
        raise ValueError("Resource admission must precede H8 feature generation")
    directory = ROOT / "operational/train40_system"
    contract = {
        **baseline,
        "schema": "train40_H8_resource_only_eight_worker_v1",
        "files": dependency_files(
            [
                directory / "history_resources8.py",
                directory / "controller_resources8.py",
                directory / "feature_resources.py",
            ]
        ),
        "source_sha256": digest(directory / "history_resources8.py"),
        "baseline_feature_freeze_sha256": digest(output / "FEATURES_FREEZE.json"),
        "baseline_kernel_sha256": digest(directory / "history_features.py"),
        "resource_only_source_replacements": [list(pair) for pair in REPLACEMENTS],
        "raw_workers": 8,
        "maximum_pending_raw_queries": 8,
        "CPU_threads_per_raw_worker": 1,
        "resource_QA": {name: digest(output / name) for name in checks},
        "resource_QA_source_sha256": digest(ROOT / "tests/unit/test_train40_feature_resources.py"),
        "math_precision_rows_order_targets_and_checkpoint_parents_unchanged": True,
        "original_RAM_disk_deadline_and_one_heavy_writer_guards_retained": True,
        "optimizer_updates_by_admission": 0,
        "acceleration_measured": False,
    }
    path = output / "FEATURES_EIGHT_WORKER_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve published resource admission")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only:
        generate()
    else:
        freeze(args.output.resolve())

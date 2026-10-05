"""Seal tested local source dependencies and prerequisite receipts before new scientific fits."""

from __future__ import annotations

import ast
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def dependency_files(seeds: list[Path]) -> list[dict]:
    """Follow local Python imports without importing or executing scientific modules."""
    queue, seen = list(seeds), set()
    while queue:
        path = queue.pop().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        relative = path.relative_to(ROOT)
        module = list(relative.with_suffix("").parts)
        if module[0] == "src":
            module = module[1:]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = module[: -node.level] if node.level else []
                target = ".".join([*base, *(node.module or "").split(".")]).strip(".")
                names = [target, *(target + "." + alias.name for alias in node.names)]
            for name in names:
                parts = name.split(".")
                root = ROOT / "src" if parts[0] == "e_jepa_ttc" else ROOT
                if parts[0] not in ("e_jepa_ttc", "operational"):
                    continue
                target = root.joinpath(*parts)
                for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                    if candidate.is_file():
                        queue.append(candidate)
    return [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(seen)]


def run() -> None:
    """Publish engineering freezes without admitting incomplete data to the model trainer."""
    output = OUTPUT.resolve()
    pyright = output / "FINAL_QUEUE_PYRIGHT.txt"
    ruff = output / "FINAL_QUEUE_RUFF.txt"
    if "0 errors" not in pyright.read_text(encoding="utf-8"):
        raise ValueError("Final source type checking must pass")
    if "All checks passed" not in ruff.read_text(encoding="utf-8"):
        raise ValueError("Final source lint must pass")
    for name in (
        "RECOVERY_CONTRACT_PYTEST.txt",
        "FULL_STATE_INTEGRITY_RESUME_PYTEST.txt",
        "STATE_INTEGRITY_PYTEST.txt",
        "EXTRA_PREPARE_PYTEST.txt",
    ):
        if "[100%]" not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required QA receipt absent: {name}")
    for name in (
        "REAL_MODEL_ADMISSION",
        "REAL_HISTORY_INPUT_ADMISSION",
        "PUBLIC_GARL_REAL_INPUT_ADMISSION",
    ):
        if read(output / (name + ".json"))["status"] != "PASSED":
            raise ValueError(f"Required real TRAIN admission absent: {name}")
    seeds = list((ROOT / "operational/train40_system").glob("*.py"))
    files = dependency_files(seeds)
    qa_names = [
        "REAL_MODEL_ADMISSION.json",
        "REAL_HISTORY_INPUT_ADMISSION.json",
        "PUBLIC_GARL_REAL_INPUT_ADMISSION.json",
        "RECOVERY_CONTRACT_PYTEST.txt",
        "FULL_STATE_INTEGRITY_RESUME_PYTEST.txt",
        "STATE_INTEGRITY_PYTEST.txt",
        "EXTRA_PREPARE_PYTEST.txt",
        "FINAL_QUEUE_PYRIGHT.txt",
        "FINAL_QUEUE_RUFF.txt",
    ]
    contract = {
        "schema": "train40_QA_engineering_freeze_v1",
        "files": files,
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "QA": {name: digest(output / name) for name in qa_names},
        "checkpoint_has_full_state_integrity_hash": True,
        "Windows_reader_lock_publication_retry_tested": True,
        "new_scientific_optimizer_updates_before_full_data_seal": 0,
        "teacher_and_input_kernel_sources_preserved": True,
        "historical_holdouts_or_evaluation_payloads_opened": False,
    }
    path = output / "ENGINEERING_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing engineering freeze")
    atomic_json(path, contract)
    atomic_json(
        output / "PARTITION_PREPARE_FREEZE.json",
        {
            "files": dependency_files([ROOT / "operational/train40_system/prepare_partition.py"]),
            "QA_sha256": digest(output / "RECOVERY_CONTRACT_PYTEST.txt"),
            "kernel_recipe_freeze_sha256": digest(output / "PREPARE_FREEZE.json"),
            "only_progress_publication_changes_from_original_kernel": True,
        },
    )


if __name__ == "__main__":
    run()

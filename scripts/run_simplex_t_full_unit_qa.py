"""Accounted CPU-only companion unit suite with explicit resource and update limits."""

from __future__ import annotations

import argparse
import functools
import json
import os
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json
from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import TechnicalBudget, admitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    work = Path.cwd().resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("new QA output under companion T0 required")
    if args.other_reserved_bytes < 0:
        raise ValueError("nonnegative outstanding output reservation required")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    import pytest
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    def boundary() -> None:
        snapshot = admitted([work])
        if not snapshot["has_headroom"] or not shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 1_073_741_824
        ):
            raise InterruptedError("RESOURCE_PAUSE during CPU unit QA")

    boundary()
    tests = sorted((work / "tests/unit").glob("test_simplex*.py"))
    pins = {str(path.relative_to(work)): sha256(path) for path in tests}
    code_pins = {
        str(path.relative_to(work)): sha256(path)
        for folder in ("src", "scripts")
        for path in (work / folder).rglob("*.py")
    }
    budget = work / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    TechnicalBudget(budget).reserve(args.operation_id, 20)
    output.mkdir(parents=True)
    write_new_json(
        output / "CONTRACT.json",
        {
            "test_files": pins,
            "code_files": code_pins,
            "operation_id": args.operation_id,
            "reserved_optimizer_updates": 20,
            "budget_sha256": sha256(budget),
            "device": "cpu",
            "threads": 4,
            "interop_threads": 2,
            "cuda_visible_devices": "-1",
            "runner_sha256": sha256(Path(__file__)),
        },
    )
    counters = {"attempted_optimizer_updates": 0, "completed_optimizer_updates": 0}
    original_step = torch.optim.AdamW.step

    @functools.wraps(original_step)
    def counted_step(
        self: torch.optim.AdamW, closure: Callable[[], torch.Tensor | float] | None = None
    ) -> torch.Tensor | float | None:
        boundary()
        if counters["attempted_optimizer_updates"] >= 20:
            raise RuntimeError("technical suite attempted to exceed reserved 20 updates")
        counters["attempted_optimizer_updates"] += 1
        atomic_json(output / "UPDATE_PROGRESS.json", counters)
        result = original_step(self, closure)
        counters["completed_optimizer_updates"] += 1
        atomic_json(output / "UPDATE_PROGRESS.json", counters)
        return result

    class Results:
        def __init__(self) -> None:
            self.reports = []
            self.collected = []

        def pytest_collection_finish(self, session: pytest.Session) -> None:
            self.collected = [item.nodeid for item in session.items]
            write_new_json(output / "COLLECTED.json", {"nodeids": self.collected})

        def pytest_runtest_setup(self, item: pytest.Item) -> None:
            try:
                boundary()
            except InterruptedError as error:
                pytest.exit(str(error), returncode=3)

        def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
            record = {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "duration": report.duration,
                "longrepr": str(report.longrepr) if report.failed else None,
            }
            self.reports.append(record)
            with (output / "reports.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    plugin = Results()
    torch.optim.AdamW.step = counted_step
    try:
        exit_code = pytest.main(
            [
                *[str(path) for path in tests],
                "-q",
                "--basetemp",
                str(output / "test_tmp"),
                "--junitxml",
                str(output / "QA.xml"),
            ],
            plugins=[plugin],
        )
    finally:
        torch.optim.AdamW.step = original_step
    if any(sha256(work / path) != digest for path, digest in (pins | code_pins).items()):
        raise ValueError("unit QA source changed during execution")
    failed = sorted({r["nodeid"] for r in plugin.reports if r["outcome"] == "failed"})
    result = {
        "status": "UNIT_QA_COMPLETED_NOT_SCIENTIFIC_ADMISSION"
        if exit_code == 0
        else "UNIT_QA_NOT_PASSED",
        "exit_code": int(exit_code),
        "collected": len(plugin.collected),
        "failed_nodeids": failed,
        **counters,
        "scientific_updates": 0,
        "reports_sha256": sha256(output / "reports.jsonl"),
        "not_covered": ["real GPU H16 parity", "complete data-pool integration", "scientific fits"],
    }
    write_new_json(output / "RESULT.json", result)
    print(json.dumps(result))
    if exit_code != 0:
        raise SystemExit(int(exit_code))


if __name__ == "__main__":
    main()

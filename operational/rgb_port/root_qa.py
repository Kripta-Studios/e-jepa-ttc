"""Run CPU admission QA with durable accounting of actual optimizer steps."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import (
    atomic_write_json,
    read_json_shared,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--allow-updates", type=int, default=0)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    if not 0 <= args.allow_updates <= 500:
        raise ValueError("Technical allowance must be within the campaign cap")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import pytest
    import torch

    torch.set_num_threads(2)
    path = args.run / "TECHNICAL_JOURNAL.json"
    journal: dict[str, Any] = (
        read_json_shared(path)
        if path.exists()
        else {"schema": "rgb_port_technical_journal_v1", "completed": 0, "pending_upper": 0}
    )
    # A process that died inside a step conservatively consumes that allowance.
    journal["completed"] += int(journal["pending_upper"])
    journal["pending_upper"] = 0
    before = int(journal["completed"])
    atomic_write_json(path, journal)
    methods: list[tuple[type, Any]] = []

    def instrument(original: Callable[..., object]) -> Callable[..., object]:
        def step(self: torch.optim.Optimizer, *positional: object, **keyword: object) -> object:
            completed = int(journal["completed"])
            if completed - before >= args.allow_updates or completed >= 500:
                raise RuntimeError(
                    "QA optimizer update is outside its admitted technical allowance"
                )
            journal["pending_upper"] = 1
            atomic_write_json(path, journal)
            result = original(self, *positional, **keyword)
            journal["completed"] += 1
            journal["pending_upper"] = 0
            atomic_write_json(path, journal)
            return result

        return step

    for optimizer in (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
        original = optimizer.step
        methods.append((optimizer, original))
        optimizer.step = instrument(original)
    try:
        code = pytest.main(["-q", *args.tests])
    finally:
        for optimizer, original in methods:
            optimizer.step = original
        receipt = {
            "schema": "rgb_port_root_qa_v1",
            "scope": "CPU scientific contracts and recovery; no GPU admission",
            "technical_updates_before": before,
            "technical_updates_after": int(journal["completed"]),
            "pending_upper": int(journal["pending_upper"]),
            "tests": {str(Path(name)): sha256_file(Path(name)) for name in args.tests},
            "torch_version": torch.__version__,
            "cuda_hidden": True,
        }
        atomic_write_json(args.run / "ROOT_QA_RECEIPT.json", receipt)
    receipt["exit_code"] = int(code)
    atomic_write_json(args.run / "ROOT_QA_RECEIPT.json", receipt)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

"""Admit grouped native contexts under existing bounded shared GPU policy."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from operational.multimodal_validation.common import read, sha, write

from . import common, shared_run
from .guard_scan import wrap


def main() -> None:
    assert read(common.OUT / "GUARD_SCAN_PARITY.json")["source_sha256"] == sha(
        Path(__file__).with_name("guard_scan.py")
    )
    assert read(common.OUT / "GROUP_CROSSFIT_NATIVE_PARITY.json")["context_source_sha256"] == sha(
        Path(__file__).with_name("crossfit_context.py")
    )
    receipt_path = common.OUT / f"GROUP_CROSSFIT_EXECUTION_{os.getpid()}.json"
    receipt = dict(
        utc_start=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        status="RUNNING",
        source_sha256=sha(__file__),
        guard_adapter_sha256=sha(Path(__file__).with_name("guard_scan.py")),
        admission_policy_source_sha256=sha(Path(shared_run.__file__)),
        trainer_source_sha256=sha(Path(__file__).with_name("crossfit_context.py")),
        guards="Native limits and ownership unchanged; skip discarded non-Python metadata",
        max_own_cuda_allocation_bytes=1024**3,
    )
    write(receipt_path, receipt)
    try:
        with (
            patch.object(common, "guard", wrap(common.guard)),
            patch.dict(shared_run.ACTIONS, {"group_crossfit": ("crossfit_context", [])}),
            patch.object(sys, "argv", ["shared_run", "group_crossfit"]),
        ):
            shared_run.main()
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt["status"] = "STOPPED"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["utc_end"] = datetime.now(UTC).isoformat()
        write(receipt_path, receipt)


if __name__ == "__main__":
    main()

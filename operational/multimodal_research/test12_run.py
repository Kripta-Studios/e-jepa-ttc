"""Run frozen official-input inference under the unchanged bounded shared admission."""

import sys
from pathlib import Path
from unittest.mock import patch

from operational.multimodal_validation.common import read, sha

from . import common, shared_run
from .guard_scan import wrap


def main() -> None:
    assert read(common.OUT / "GUARD_SCAN_PARITY.json")["source_sha256"] == sha(
        Path(__file__).with_name("guard_scan.py")
    )
    with (
        patch.object(common, "guard", wrap(common.guard)),
        patch.dict(shared_run.ACTIONS, {"test12_frozen": ("test12_infer", [])}),
        patch.object(sys, "argv", ["shared_run", "test12_frozen"]),
    ):
        shared_run.main()


if __name__ == "__main__":
    main()

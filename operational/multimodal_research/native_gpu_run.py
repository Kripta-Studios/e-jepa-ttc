"""Run native-precision transfer under the existing bounded shared admission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

from operational.multimodal_validation.common import read, sha, write

from . import shared_run
from .common import OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("FCWD",))
    args = parser.parse_args()
    extension = dict(
        source_sha256=sha(__file__),
        admission_source_sha256=sha(Path(shared_run.__file__)),
        extractor_source_sha256=sha(Path(__file__).with_name("external_states_gpu.py")),
        change="Additional frozen-inference action; resource limits unchanged",
        hard_own_allocator_limit_bytes=1024**3,
        optimizer_updates=0,
    )
    path = OUT / "admission" / ("NATIVE_TRANSFER_" + sha(__file__) + ".json")
    if path.exists():
        assert read(path) == extension
    else:
        write(path, extension)
    action = "native_gpu_" + args.cohort
    with (
        patch.dict(shared_run.ACTIONS, {action: ("external_states_gpu", [args.cohort])}),
        patch.object(sys, "argv", ["shared_run", action]),
    ):
        shared_run.main()


if __name__ == "__main__":
    main()

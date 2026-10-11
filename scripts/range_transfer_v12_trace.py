"""Load historical V12 modules in an isolated process; write only into V14."""

# ruff: noqa: E402 -- establish the historical module search path first.
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--v12-root", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
v12 = args.v12_root.resolve(strict=True)
v14 = Path(__file__).resolve().parents[1]
run = args.run.resolve()
if not run.is_relative_to(v14) or v12 == v14:
    raise ValueError("Require historical V12 inputs and separate V14 output")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path[:0] = [str(v12 / "src"), str(v12)]
import operational

operational.__path__.append(str(v14 / "operational"))
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from operational.range_transfer.common import write

try:
    from operational.range_transfer.raw_traces import trace_raw

    result = trace_raw(v12, run)
    print(result, flush=True)
except Exception as error:
    write(
        run / "v12_raw_attribution/FAILURE.json",
        {"type": type(error).__name__, "message": str(error), "optimizer_updates": 0},
    )
    raise

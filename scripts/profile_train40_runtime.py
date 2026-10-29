"""Profile the frozen oracle without modifying its predictions or checkpoints."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch  # noqa: E402

from scripts.final_train40_transfer import (  # noqa: E402
    FinalTransferPredictor,
    read,
    verify_candidate,
)


def main() -> None:
    """Write a bounded, explicitly instrumented oracle profile to a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 200, 553])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    paths, binding = verify_candidate(args.candidate)
    predictor = FinalTransferPredictor(paths, "cuda")
    rows = read(args.manifest)["rows"]
    results = []
    for index in args.indices:
        row = rows[index]
        torch.cuda.synchronize()
        profile = cProfile.Profile()
        start = time.perf_counter()
        result = profile.runcall(predictor.predict, row)
        torch.cuda.synchronize()
        result.update(
            index=index, query_id=row["query_id"], wall_seconds=time.perf_counter() - start
        )
        results.append(result)
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(70)
        (args.output / f"profile_{index}.txt").write_text(stream.getvalue(), encoding="utf-8")
        print(json.dumps(result), flush=True)
    payload = {
        "scope": "INSTRUMENTED_CPROFILE_NOT_ACCEPTANCE_LATENCY",
        "binding": binding,
        "results": results,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
